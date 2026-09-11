"""
RSMA grafted onto a frozen open-weight base model.

The base (a Hugging Face causal LM) is frozen. Between selected decoder blocks a self-referential
fast-weight sublayer is inserted via forward hooks, starting near identity so the base's abilities
are preserved. A self-model gates and forecasts as in the from-scratch RSMA. Low-rank adapters on
the base's attention projections give the model room to take on new voices during training and
during sleep consolidation. Only the fast layers, self-model and adapters are trained.

forward(input_ids, state, targets, ref_chunk_loss) returns (logits, new_state, aux) with the same
aux keys as RSMA, so training, evaluation, runtime and consolidation code is shared.
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RSMAConfig
from .fastweights import SelfReferentialFastWeights
from .selfmodel import SelfModel


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=16, alpha=32):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x):
        lora = F.linear(F.linear(x.to(self.A.dtype), self.A), self.B) * self.scale
        return self.base(x) + lora.to(x.dtype)


class GraftedRSMA(nn.Module):
    def __init__(self, base_name="Qwen/Qwen2.5-0.5B-Instruct", cfg: RSMAConfig = None, lora_r=16,
                 lora_targets=("q_proj", "k_proj", "v_proj", "o_proj"), dtype=torch.float32):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.base_name = base_name
        self.tokenizer = AutoTokenizer.from_pretrained(base_name)
        self.base = AutoModelForCausalLM.from_pretrained(base_name, dtype=dtype)
        for p in self.base.parameters():
            p.requires_grad = False
        bc = self.base.config
        cfg = cfg or RSMAConfig()
        cfg.d_model = bc.hidden_size
        cfg.vocab_size = bc.vocab_size
        cfg.n_layers = bc.num_hidden_layers
        if cfg.d_model % cfg.fast_heads != 0 or cfg.d_model // cfg.fast_heads != 64:
            cfg.fast_heads = max(1, cfg.d_model // 64)  # 64-wide heads regardless of the base's width
        self.cfg = cfg
        self.lora_r, self.lora_targets = lora_r, tuple(lora_targets)
        layers = self.base.model.layers
        # adapters
        if lora_r > 0:
            for layer in layers:
                for name in lora_targets:
                    lin = getattr(layer.self_attn, name)
                    setattr(layer.self_attn, name, LoRALinear(lin, r=lora_r))
        # fast layers
        self.fast_layer_ids = sorted(i for i in cfg.fast_layers if i < len(layers))
        self.fast = nn.ModuleList([SelfReferentialFastWeights(cfg, fi) for fi in range(len(self.fast_layer_ids))])
        fl = self.fast[0]
        self.selfmodel = SelfModel(cfg, fl.R, fl.d, len(self.fast_layer_ids))
        self._hooks = []
        for fi, li in enumerate(self.fast_layer_ids):
            self._hooks.append(layers[li].register_forward_hook(self._make_hook(fi)))
        self._state_in = None
        self._state_out = None
        self._aux = None

    # ---- hooks ---------------------------------------------------------------------------
    def _make_hook(self, fi):
        def hook(module, args, output):
            if self._state_in is None:
                return output
            h = output[0] if isinstance(output, tuple) else output
            y, new_delta, aux = self.fast[fi](h.float(), self._state_in[fi], self.selfmodel)  # fast path runs in fp32
            self._state_out[fi] = new_delta
            self._aux[fi] = aux
            h2 = h + y.to(h.dtype)
            return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2
        return hook

    # ---- API shared with RSMA -------------------------------------------------------------
    @property
    def fast_layers(self):
        return list(self.fast)

    def init_state(self, batch, device):
        return [fl.init_state(batch, device) for fl in self.fast]

    @staticmethod
    def clone_state(state):
        return None if state is None else [d.detach().clone() for d in state]

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def n_params(self, trainable_only=False):
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def trainable_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith("base.") or ".A" in k[-2:] or ".B" in k[-2:] or k.endswith(".A") or k.endswith(".B")}

    def load_trainable_state_dict(self, sd):
        missing, unexpected = self.load_state_dict(sd, strict=False)
        unexpected = [k for k in unexpected]
        return missing, unexpected

    def forward(self, input_ids, state=None, targets=None, ref_chunk_loss=None, attention_mask=None):
        B, T = input_ids.shape
        C = self.cfg.chunk_size
        assert T % C == 0, "sequence length must be a multiple of chunk_size"
        n_chunks = T // C
        if state is None:
            state = self.init_state(B, input_ids.device)
        self._state_in = state
        self._state_out = [None] * len(self.fast)
        self._aux = [None] * len(self.fast)
        out = self.base(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
        self._state_in = None
        logits = out.logits
        h = out.hidden_states[-1].float()
        new_state, fast_aux = self._state_out, self._aux

        aux = {}
        if targets is not None:
            per_tok = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), reduction="none", ignore_index=-100).view(B, T)
            valid = (targets != -100).float()
            aux["lm_loss"] = (per_tok * valid).sum() / valid.sum().clamp(min=1)
            aux["per_token_loss"] = per_tok
            chunk_loss = (per_tok * valid).view(B, n_chunks, C).sum(-1) / valid.view(B, n_chunks, C).sum(-1).clamp(min=1)
            aux["chunk_loss"] = chunk_loss

        aux["gates"] = [a["gates"] for a in fast_aux]
        aux["delta_norms"] = torch.stack([a["delta_norm"] for a in fast_aux])
        encs = torch.stack([a["encs"] for a in fast_aux], dim=2)  # (B, n_chunks+1, n_fast, hid)
        pooled = h.view(B, n_chunks, C, -1).mean(2)
        pooled_prev = torch.cat([torch.zeros_like(pooled[:, :1]), pooled], dim=1)
        pred = self.selfmodel.predict_benefit(encs.detach(), pooled_prev.detach().float())
        aux["pred_benefit"] = pred
        aux["pooled_last"] = pooled[:, -1].detach().float()
        if targets is not None:
            tgt = (ref_chunk_loss - chunk_loss).detach() if ref_chunk_loss is not None else torch.zeros_like(chunk_loss)
            aux["benefit"] = tgt
            aux["sm_loss"] = F.huber_loss(pred[:, :n_chunks], tgt, delta=0.5)
            aux["loss"] = aux["lm_loss"] + self.cfg.selfmodel_loss_weight * aux["sm_loss"]
        return logits, new_state, aux

    @torch.no_grad()
    def forecast(self, state, pooled_last):
        encs = torch.stack([self.selfmodel.encode(d, i) for i, d in enumerate(state)], dim=1)
        return self.selfmodel.predict_benefit(encs[:, None], pooled_last[:, None])[:, 0]

    @torch.no_grad()
    def generate(self, input_ids, state=None, max_new=200, temperature=0.7, top_p=0.9, stop_ids=(), repetition_penalty=1.15):
        """Sample with the fast weights frozen at `state`. The prompt is left-padded to a
        multiple of chunk_size with the pad token so the fast layers see whole chunks."""
        self.eval()
        C = self.cfg.chunk_size
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        ids = input_ids
        for _ in range(max_new):
            ctx = ids[:, -self.cfg.seq_len:]
            pad = (-ctx.shape[1]) % C
            attn = torch.ones_like(ctx)
            if pad:
                ctx = torch.cat([torch.full((ctx.shape[0], pad), pad_id, dtype=ctx.dtype, device=ctx.device), ctx], dim=1)
                attn = torch.cat([torch.zeros(ctx.shape[0], pad, dtype=attn.dtype, device=attn.device), attn], dim=1)
            logits, _, _ = self(ctx, state, attention_mask=attn)
            logits = logits[:, -1].float()
            if repetition_penalty != 1.0:
                prev = ids[:, input_ids.shape[1]:]  # tokens generated so far
                if prev.numel():
                    sc = logits.gather(1, prev)
                    sc = torch.where(sc > 0, sc / repetition_penalty, sc * repetition_penalty)
                    logits.scatter_(1, prev, sc)
            logits = logits / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)
            sp, si = probs.sort(descending=True)
            keep = (sp.cumsum(-1) - sp) < top_p
            sp = sp * keep
            nxt = si.gather(-1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1))
            ids = torch.cat([ids, nxt], dim=1)
            if nxt.item() in stop_ids:
                break
        return ids

    # ---- persistence ----------------------------------------------------------------------
    def save(self, path, extra=None):
        torch.save({"base_name": self.base_name, "cfg": self.cfg.to_dict(), "lora_r": self.lora_r,
                    "lora_targets": self.lora_targets, "trainable": self.trainable_state_dict(), **(extra or {})}, path)

    @classmethod
    def load(cls, path, device="cpu", dtype=torch.float32):
        ck = torch.load(path, map_location="cpu")
        cfg = RSMAConfig.from_dict(ck["cfg"])
        m = cls(ck["base_name"], cfg, lora_r=ck["lora_r"], lora_targets=ck["lora_targets"], dtype=dtype)
        m.load_trainable_state_dict(ck["trainable"])
        return m.to(device), ck

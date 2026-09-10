"""
RSMA: a decoder-only transformer whose selected blocks carry a self-referential fast-weight
sublayer, plus a self-model that gates the self-modification and predicts its consequences.

forward(tokens, state) -> logits, new_state, aux
  state: list of fast deltas, one per fast layer, or None to start fresh
  aux:   lm_loss, sm_loss, chunk_loss (B,C), pred_benefit (B,C+1), benefit (B,C), gates, delta_norms
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RSMAConfig
from .fastweights import SelfReferentialFastWeights
from .selfmodel import SelfModel


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q, k, v = (t.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg, fast_index=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)
        self.fast = SelfReferentialFastWeights(cfg, fast_index) if fast_index is not None else None

    def forward(self, x, delta=None, selfmodel=None):
        x = x + self.attn(self.ln1(x))
        aux = None
        if self.fast is not None:
            y, delta, aux = self.fast(x, delta, selfmodel)
            x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, delta, aux


class RSMA(nn.Module):
    def __init__(self, cfg: RSMAConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        fast_set = set(cfg.fast_layers) if cfg.use_fast else set()
        self.fast_layer_ids = sorted(i for i in fast_set if i < cfg.n_layers)
        blocks = []
        for i in range(cfg.n_layers):
            fi = self.fast_layer_ids.index(i) if i in fast_set else None
            blocks.append(Block(cfg, fi))
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        if self.fast_layer_ids:
            fl = self.blocks[self.fast_layer_ids[0]].fast
            self.selfmodel = SelfModel(cfg, fl.R, fl.d, len(self.fast_layer_ids))
        else:
            self.selfmodel = None
        self.apply(self._init)
        # apply() above overwrote the submodules' deliberate inits; restore them
        if self.selfmodel is not None:
            self.selfmodel.reset_special_init()
        for fl in self.fast_layers:
            fl.reset_special_init()

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    # ---- fast state -------------------------------------------------------------------------
    @property
    def fast_layers(self):
        return [self.blocks[i].fast for i in self.fast_layer_ids]

    def init_state(self, batch: int, device):
        return [fl.init_state(batch, device) for fl in self.fast_layers]

    @staticmethod
    def clone_state(state):
        return None if state is None else [d.detach().clone() for d in state]

    def n_params(self, trainable_only=False):
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    # ---- forward ----------------------------------------------------------------------------
    def forward(self, tokens, state=None, targets=None, ref_chunk_loss=None):
        """
        ref_chunk_loss: optional (B, n_chunks) per-chunk loss of the same tokens with the fast
        state reset. When given, the self-model is trained to forecast ref - actual, the benefit
        of the carried state. When absent the benefit target is zero (true for a fresh state).
        """
        B, T = tokens.shape
        dev = tokens.device
        C = self.cfg.chunk_size
        n_chunks = T // C
        if state is None:
            state = self.init_state(B, dev)
        pos = torch.arange(T, device=dev)
        x = self.drop(self.tok_emb(tokens) + self.pos_emb(pos)[None])
        new_state, fast_aux = [], []
        for i, blk in enumerate(self.blocks):
            if blk.fast is not None:
                fi = blk.fast.fast_index
                x, d, aux = blk(x, state[fi], self.selfmodel)
                new_state.append(d)
                fast_aux.append(aux)
            else:
                x, _, _ = blk(x)
        h = self.ln_f(x)
        logits = self.head(h)

        aux = {}
        if targets is not None:
            per_tok = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none").view(B, T)
            aux["lm_loss"] = per_tok.mean()
            aux["per_token_loss"] = per_tok
            chunk_loss = per_tok.view(B, n_chunks, C).mean(-1)  # (B, n_chunks)
            aux["chunk_loss"] = chunk_loss

        if fast_aux:
            aux["gates"] = [a["gates"] for a in fast_aux]
            aux["delta_norms"] = torch.stack([a["delta_norm"] for a in fast_aux])
            if self.selfmodel is not None:
                # encodings of fast state at the start of chunk c (c = 0..n_chunks), all layers
                encs = torch.stack([a["encs"] for a in fast_aux], dim=2)  # (B, n_chunks+1, n_fast, hid)
                # pooled hidden of the previous chunk; zeros before chunk 0
                pooled = h.view(B, n_chunks, C, -1).mean(2)
                pooled_prev = torch.cat([torch.zeros_like(pooled[:, :1]), pooled], dim=1)  # (B, n_chunks+1, D)
                pred = self.selfmodel.predict_benefit(encs.detach(), pooled_prev.detach())  # (B, n_chunks+1)
                aux["pred_benefit"] = pred      # pred[:, c]: forecast benefit of the state on chunk c; pred[:, -1]: forecast for what comes next
                aux["pooled_last"] = pooled[:, -1].detach()
                if targets is not None:
                    tgt = (ref_chunk_loss - chunk_loss).detach() if ref_chunk_loss is not None else torch.zeros_like(chunk_loss)
                    aux["benefit"] = tgt
                    aux["sm_loss"] = F.huber_loss(pred[:, :n_chunks], tgt, delta=0.5)

        if targets is not None:
            total = aux["lm_loss"]
            if "sm_loss" in aux:
                total = total + self.cfg.selfmodel_loss_weight * aux["sm_loss"]
            aux["loss"] = total
        return logits, new_state, aux

    @torch.no_grad()
    def forecast(self, state, pooled_last):
        """Self-model forecast (B,) of the benefit of an arbitrary fast state on the next chunk.
        Used to compare a modification against the state before it."""
        if self.selfmodel is None:
            return None
        encs = torch.stack([self.selfmodel.encode(d, i) for i, d in enumerate(state)], dim=1)  # (B, n_fast, hid)
        return self.selfmodel.predict_benefit(encs[:, None], pooled_last[:, None])[:, 0]

    @torch.no_grad()
    def generate(self, tokens, state=None, max_new=64, temperature=1.0):
        """Greedy or sampled generation. Fast weights are frozen during generation for simplicity."""
        self.eval()
        for _ in range(max_new):
            ctx = tokens[:, -self.cfg.seq_len:]
            pad = (-ctx.shape[1]) % self.cfg.chunk_size
            if pad:
                ctx = torch.cat([torch.zeros(ctx.shape[0], pad, dtype=ctx.dtype, device=ctx.device), ctx], dim=1)
            logits, _, _ = self(ctx, state)
            probs = F.softmax(logits[:, -1] / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, nxt], dim=1)
        return tokens

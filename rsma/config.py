from dataclasses import dataclass, field, asdict
from typing import Tuple


@dataclass
class RSMAConfig:
    # transformer
    vocab_size: int = 64
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    seq_len: int = 256
    dropout: float = 0.0

    # self-referential fast weights
    use_fast: bool = True
    fast_layers: Tuple[int, ...] = (1, 3, 5)  # blocks that carry a fast-weight sublayer
    fast_heads: int = 4
    chunk_size: int = 16          # tokens per fast-weight update step
    fast_lr_scale: float = 1.0    # multiplies sigmoid(beta)
    fast_beta_init: float = -1.0  # bias on beta; sigmoid(-1) ~ 0.27 initial rate
    fast_w0_std: float = 0.1      # init scale of the slow fast-weight matrix W0
    fast_max_norm: float = 8.0    # per-head Frobenius clip on the fast delta
    fast_direct_value: bool = True  # write target includes a direct projection of the input

    # self-model
    use_gate: bool = True
    summary_dim: int = 32         # random-projection size of each head's delta
    selfmodel_hidden: int = 128
    selfmodel_loss_weight: float = 0.1

    stream_len: int = 1           # consecutive windows per training stream; state carries across them

    # persistence: 1 = reset per sequence, 2 = persist across sequences, 3 = consolidate into slow weights
    tier: int = 1

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        if "fast_layers" in d:
            d["fast_layers"] = tuple(d["fast_layers"])
        return cls(**d)

from dataclasses import dataclass
import warnings
import cs336_basics.nn.attention  as attn_mod
from cs336_basics.nn.transformer import TransformerLM
from cs336_systems.flash_attn.fa_triton import FlashAttentionTritonFunc

@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 768
    d_ff: int = 3072
    num_layers: int = 12
    num_heads: int = 12
    context_len: int = 128
    vocab_size: int = 10000
    batch_size: int = 4
    lr: float = 3e-4
    annotated:bool = False

def flash_attention_wrapper(Q, K, V, mask=None):
    if mask is not None:
        warnings.warn(
            "FlashAttention wrapper currently ignores the provided mask. "
            "Results may be incorrect if the mask is not strictly causal.",
            RuntimeWarning,
        )
        return FlashAttentionTritonFunc.apply(Q, K, V, True)
    return FlashAttentionTritonFunc.apply(Q, K, V, False)

def build_model(cfg: ModelConfig) -> TransformerLM:

    attn_mod.scaled_dot_product_attention = flash_attention_wrapper
    model = TransformerLM(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_len,
        num_layers=cfg.num_layers,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        rope_theta=10000,
    )
    return model
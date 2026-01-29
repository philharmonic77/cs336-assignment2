from __future__ import annotations

from dataclasses import dataclass
from timeit import default_timer as timer
from pathlib import Path
from statistics import mean, stdev
import argparse
import json

import torch
import torch.cuda.nvtx as nvtx
from torch import Tensor
from jaxtyping import Int

from cs336_basics.nn.transformer import TransformerLM
from cs336_basics.losses import cross_entropy
from cs336_basics.optim import AdamW
import cs336_basics.nn.attention  as attn_mod

from jaxtyping import Float, Int, Bool
import math
from einops import einsum
from contextlib import nullcontext


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




def _build_model(cfg: ModelConfig) -> TransformerLM:
    if cfg.annotated:
        attn_mod.scaled_dot_product_attention = annotated_scaled_dot_product_attention
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


def _generate_data_batch(
    cfg: ModelConfig, device: torch.device
) -> tuple[Int[Tensor, "B S"], Int[Tensor, "B S"]]:
    x = torch.randint(
        0, cfg.vocab_size,
        (cfg.batch_size, cfg.context_len + 1),
        device=device,
        dtype=torch.long,
    )
    y = x[:, 1:]
    x = x[:, :-1]
    return x, y


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _resolve_dtype(dtype_arg: str) -> torch.dtype:
    if dtype_arg == "fp32":
        return torch.float32
    if dtype_arg == "fp16":
        return torch.float16
    if dtype_arg == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {dtype_arg}")


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def run_once(
    cfg: ModelConfig,
    *,
    warm_up: int,
    nsteps: int,
    mode: str,  # "forward_only" | "train_step"
    use_bf16: bool = False,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, float]:
    
    model = _build_model(cfg).to(device=device, dtype=dtype)
    optimizer = AdamW(model.parameters(), lr=cfg.lr)

    if mode == "forward_only":
        model.eval()
    elif mode == "train_step":
        model.train()
    else:
        raise ValueError(f"Unknown mode: {mode!r}")
    
    if use_bf16:
        amp_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    else:
        amp_ctx = nullcontext()

    # -----------------
    # Warmup (sync each step)
    # -----------------
    with nvtx.range("warmup"):
        for _ in range(warm_up):
            x, y = _generate_data_batch(cfg, device)

            if mode == "forward_only":
                with amp_ctx:
                    with torch.no_grad():
                        _ = model(x)
            else:
                optimizer.zero_grad(set_to_none=True)
                with amp_ctx:
                    logits = model(x)
                    loss = cross_entropy(logits, y)
                loss.backward()
                optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize()

    # -----------------
    # Measure (timer includes GPU completion)
    # -----------------
    times: list[float] = []
    with nvtx.range("measure"):
        for _ in range(nsteps):
            x, y = _generate_data_batch(cfg, device)

            start = timer()

            if mode == "forward_only":
                with nvtx.range("forward"):
                    with amp_ctx:
                        with torch.no_grad():
                            _ = model(x)

            else:
                optimizer.zero_grad(set_to_none=True)

                with nvtx.range("forward"):
                    with amp_ctx:
                        logits = model(x)

                with nvtx.range("loss"):
                    with amp_ctx:
                        loss = cross_entropy(logits, y)

                with nvtx.range("backward"):
                    loss.backward()

                with nvtx.range("optimizer_step"):
                    optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = timer()
            times.append(end - start)

    avg = mean(times)
    sd = stdev(times) if len(times) > 1 else 0.0
    return avg, sd

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... n d_k"],
    K: Float[Tensor, " ... m d_k"],
    V: Float[Tensor, " ... m d_v"],
    mask: Bool[Tensor, " ... n m"] | None = None,       
) -> Float[Tensor, "... n d_v"]:
    d_k = Q.shape[-1]
    assert d_k == K.shape[-1] 
    assert K.shape[-2] == V.shape[-2] 

    with nvtx.range("computing attention scores"):
        scores: Float[Tensor, "... n m"] = einsum(Q, K, "... n d_k, ... m d_k -> ... n m") / math.sqrt(d_k)

    if mask is not None:
        assert mask.shape[-1] == K.shape[-2]
        assert mask.shape[-2] == Q.shape[-2]

        scores: Float[Tensor, "... n m"] = scores.masked_fill(~mask, float("-inf")) 

    with nvtx.range("computing softmax"):
        scores: Float[Tensor, "... n m"] = softmax(scores, dim=-1)

    with nvtx.range("final matmul"):
        result: Float[Tensor, "... n d_v"] = einsum(scores, V, "... n m, ... m d_v -> ... n d_v")
    
    return result

def softmax(
    x: Float[Tensor, "... d_model"],
    dim: int = -1
) -> Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    exp_x  = torch.exp(x - x_max)
    
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="nsys profile + timer + JSONL")

    p.add_argument("--model-tag", type=str, required=True)
    p.add_argument("--mode", type=str, default="forward_only",
                   choices=["forward_only", "train_step"])

    # Model hyperparameters
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--d-ff", type=int, default=3072)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--context-len", type=int, default=128)
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=4)

    # Optimizer
    p.add_argument("--lr", type=float, default=3e-4)

    # Benchmark
    p.add_argument("--warm-up", type=int, default=5)
    p.add_argument("--nsteps", type=int, default=10)

    # Runtime
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])

    # Output
    p.add_argument("--output", type=str, default="results/nsys_profile_times.jsonl")

    # use annotated self-attn or not
    p.add_argument("--annotated", action="store_true")

    # use mixed precision or not
    p.add_argument("--use-bf16", action="store_true")

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = ModelConfig(
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        context_len=args.context_len,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        lr=args.lr,
        annotated=args.annotated
    )

    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype)

    avg_s, std_s = run_once(
        cfg,
        warm_up=args.warm_up,
        nsteps=args.nsteps,
        mode=args.mode,
        use_bf16=args.use_bf16,
        device=device,
        dtype=dtype,
    )

    record = {
        "model_tag": args.model_tag,
        "mode": args.mode, 
        "num_layers": cfg.num_layers,
        "d_model": cfg.d_model,
        "d_ff": cfg.d_ff,
        "num_heads": cfg.num_heads,
        "context_len": cfg.context_len,
        "batch_size": cfg.batch_size,
        "vocab_size": cfg.vocab_size,
        "warm_up": args.warm_up,
        "nsteps": args.nsteps,
        "use_bf16": args.use_bf16,
        "device": str(device),
        "dtype": str(dtype),
        "mean_s": avg_s,
        "std_s": std_s,
    }
    append_jsonl(Path(args.output), record)

    print(json.dumps(record))


if __name__ == "__main__":
    main()

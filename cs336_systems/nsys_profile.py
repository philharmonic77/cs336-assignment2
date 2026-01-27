from dataclasses import dataclass
from timeit import default_timer as timer
from cs336_basics.nn.transformer import TransformerLM
from cs336_basics.losses import cross_entropy
from cs336_basics.optim import AdamW
import torch
from torch import Tensor
from jaxtyping import Int
import argparse
import torch.cuda.nvtx as nvtx
from enum import Enum
from statistics import stdev, mean
from pathlib import Path 
import json


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


class StepMode(Enum):
    FORWARD_ONLY = "forward_only"
    TRAIN_STEP = "train_step"


def run_profiling(
    cfg: ModelConfig,
    warm_up: int,
    nsteps: int,
    mode: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, float]:
    model = _build_model(cfg)
    model.to(device=device, dtype=dtype)

    if mode == StepMode.FORWARD_ONLY.value:
        model.eval()
    else:
        model.train()

    optimizer = AdamW(model.parameters(), lr=cfg.lr)

    # -----------------
    # Warmup (sync each step)
    # -----------------
    with nvtx.range("warmup"):
        for _ in range(warm_up):
            x, y = _generate_data_batch(cfg, device=device)

            if mode == StepMode.FORWARD_ONLY.value:
                with torch.no_grad():
                    _ = model(x)
            else:
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = cross_entropy(logits, y)
                loss.backward()
                optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize()

    # -----------------
    # Measure
    # -----------------
    times: list[float] = []
    with nvtx.range("measure"):
        for _ in range(nsteps):
            x, y = _generate_data_batch(cfg, device=device)
            start = timer()

            if mode == StepMode.FORWARD_ONLY.value:
                with nvtx.range("forward"):
                    with torch.no_grad():
                        _ = model(x)
                # sync OUTSIDE range so NVTX "forward" doesn't include sync wait
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end = timer()

            else:
                optimizer.zero_grad(set_to_none=True)

                with nvtx.range("forward"):
                    logits = model(x)

                with nvtx.range("loss"):
                    loss = cross_entropy(logits, y)

                with nvtx.range("backward"):
                    loss.backward()

                with nvtx.range("optimizer_step"):
                    optimizer.step()

                if device.type == "cuda":
                    torch.cuda.synchronize()
                end = timer()

            times.append(end - start)

    return mean(times), stdev(times) if len(times) > 1 else 0.0


def _build_model(cfg: ModelConfig) -> TransformerLM:
    return TransformerLM(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_len,
        num_layers=cfg.num_layers,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        rope_theta=10000,
    )


def _generate_data_batch(
    cfg: ModelConfig, device: torch.device
) -> tuple[Int[Tensor, "B S"], Int[Tensor, "B S"]]:
    input_ids = torch.randint(
        low=0,
        high=cfg.vocab_size,
        size=(cfg.batch_size, cfg.context_len + 1),
        device=device,
        dtype=torch.long,
    )
    target_ids = input_ids[:, 1:]
    input_ids = input_ids[:, :-1]
    return input_ids, target_ids


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="time + NVTX annotate forward/train_step")

    p.add_argument("--model-tag", type=str, default=None)
    p.add_argument("--mode", type=str, default="forward_only", choices=["forward_only", "train_step"])

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

    return p.parse_args()


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

def append_jsonl(output_path: Path , record: dict):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'a', encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

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
    )

    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype)

    avg_s, std_s = run_profiling(
        cfg=cfg,
        warm_up=args.warm_up,
        nsteps=args.nsteps,
        mode=args.mode,
        device=device,
        dtype=dtype,
    )

    record = {
        "model_tag": args.model_tag,
        "mode": args.mode.value,
        "num_layers": cfg.num_layers,
        "d_model": cfg.d_model,
        "d_ff": cfg.d_ff,
        "num_heads": cfg.num_heads,
        "context_len": cfg.context_len,
        "batch_size": cfg.batch_size,
        "vocab_size": cfg.vocab_size,
        "warm_up": args.warm_up,
        "nsteps": args.nsteps,
        "device": str(device),
        "dtype": str(dtype),
        "mean_s": avg_s,
        "std_s": std_s,        
    }

    append_jsonl(Path(args.output), record)

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"device={device} ({gpu_name}), dtype={dtype}, mode={args.mode.value}")
    else:
        print(f"device={device}, dtype={dtype}, mode={args.mode.value}")

    print(f"avg time: {avg_s:.2f} s, std time: {std_s:.2f} s")


if __name__ == "__main__":
    main()
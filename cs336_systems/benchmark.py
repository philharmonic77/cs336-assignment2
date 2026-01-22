from timeit import default_timer as timer
from dataclasses import dataclass
from cs336_basics.nn.transformer import TransformerLM
from cs336_basics.losses import cross_entropy
import torch
from torch import Tensor
from jaxtyping import Int
from enum import Enum
from statistics import stdev, mean
import argparse


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 768
    d_ff: int = 3072
    num_layers: int = 12
    num_heads: int = 12
    context_len: int = 128
    vocab_size: int = 10000
    batch_size: int = 4

class StepMode(Enum):
    FORWARD = "forward"
    FORWARD_BACKWARD = "forward_backward"

def run_benchmark(
        cfg: ModelConfig,
        warm_up: int,
        nsteps: int,
        mode:  StepMode,
        *,
        device: torch.device,
        dtype: torch.dtype
) -> tuple[float, float]:
    
    model = _build_model(cfg)
    model.to(device=device, dtype=dtype)

    for _ in range(warm_up):
        x, y = _generate_data_batch(cfg, device=device)
        model.zero_grad(set_to_none=True)

        logits = model(x)
        if mode == StepMode.FORWARD_BACKWARD:
            loss = cross_entropy(logits, y)
            loss.backward()

        if device.type == "cuda":
            torch.cuda.synchronize()

    times = []
    for _ in range(nsteps):
        x, y = _generate_data_batch(cfg, device=device)
        model.zero_grad(set_to_none=True)

        start = timer()
        logits = model(x)

        if mode == StepMode.FORWARD:
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = timer()  

        elif mode == StepMode.FORWARD_BACKWARD:
            loss = cross_entropy(logits, y)
            loss.backward()

            if device.type == "cuda":
                torch.cuda.synchronize()
            end = timer()
        else:
            raise ValueError(f"mode can only be forward or forward_backward, but now it's {mode}")
        elapsed = end - start
        times.append(elapsed)

    return mean(times), stdev(times)


def _build_model(
        cfg: ModelConfig,
) -> TransformerLM:

    model = TransformerLM(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_len,
        num_layers=cfg.num_layers,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        rope_theta=10000
    )
    return model

def _generate_data_batch(
        cfg:ModelConfig,
        device: torch.device) -> tuple[Int[Tensor, "B S"], Int[Tensor, "B S"]]:
    
    input_ids = torch.randint(
        low=0,
        high=cfg.vocab_size,
        size=(cfg.batch_size, cfg.context_len + 1),
        device=device,
        dtype=torch.long
    )

    target_ids = input_ids[:, 1:]
    input_ids = input_ids[:, :-1]
    return input_ids, target_ids

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="run benchmark")
    
    # Model hyperparameters
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--d-ff", type=int, default=3072)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--context-len", type=int, default=128)
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=4)

    # Benchmark hyperparameters
    p.add_argument("--warm-up", type=int, default=5)
    p.add_argument("--nsteps", type=int, default=30)
    p.add_argument("--mode", type=str, default="forward_backward",
                  choices=["forward", "forward_backward"])

    # Runtime
    p.add_argument("--device", type=str, default="auto",
                  choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", type=str, default="fp32",
                  choices=["fp32", "fp16", "bf16"])

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


def _resolve_mode(mode_arg: str) -> StepMode:
    return StepMode.FORWARD if mode_arg == "forward" else StepMode.FORWARD_BACKWARD


def main():
    args = _parse_args()

    cfg = ModelConfig(
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        context_len=args.context_len,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
    )

    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype)
    mode = _resolve_mode(args.mode)

    avg_time, std_time = run_benchmark(
        cfg=cfg,
        warm_up=args.warm_up,
        nsteps=args.nsteps,
        mode=mode,
        device=device,
        dtype=dtype,
    )

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"device={device} ({gpu_name}), dtype={dtype}, mode={mode.value}")
    else:
        print(f"device={device}, dtype={dtype}, mode={mode.value}")

    print(f"avg time: {avg_time * 1000:.2f} ms, std time: {std_time * 1000:.2f} ms")

if __name__ == "__main__":
    main()

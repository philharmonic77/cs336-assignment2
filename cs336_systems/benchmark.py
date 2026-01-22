from timeit import default_timer as timer
from dataclasses import dataclass
from cs336_basics.nn.transformer import TransformerLM
from cs336_basics.losses import cross_entropy
import torch
from torch import Tensor
from jaxtyping import Int
from enum import Enum
from statistics import stdev, mean


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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = StepMode.FORWARD_BACKWARD
    avg_time, std_time = run_benchmark(
        cfg=ModelConfig(),
        warm_up=5,
        nsteps=30,
        mode=mode,
        device=device,
        dtype=torch.float32
    )
    print(f"mode: {mode}, avg time: {avg_time * 1000:.2f} ms, std time: {std_time * 1000:.2f} ms")

if __name__ == "__main__":
    main()

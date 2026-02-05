from jaxtyping import Float, Int, Bool
import torch
import math
from torch import Tensor
from einops import einsum
from timeit import default_timer as timer
from statistics import stdev, mean
import argparse
from pathlib import Path
import json



def run_once(
        d_model: int,
        context_len: int,
        warm_up: int,
        nsteps: int,
        use_torch_comile: Bool,
        batch_size: int=8,
        device: str = "cuda"
) -> tuple[float, ...]:
    
    Q = torch.randn(batch_size, context_len, d_model, device=device, requires_grad=True)
    K = torch.randn(batch_size, context_len, d_model, device=device, requires_grad=True)
    V = torch.randn(batch_size, context_len, d_model, device=device, requires_grad=True)

    mask = torch.tril(torch.ones(context_len, context_len, device=device, dtype=torch.bool))

    if use_torch_comile:
        f = torch.compile(scaled_dot_product_attention)
    else:
        f = scaled_dot_product_attention

    for _ in range(warm_up):
        out = f(Q, K, V, mask)
        loss = out.sum()
        loss.backward()
        Q.grad = None; K.grad = None; V.grad = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    forward_times: list[float] = []
    backward_times: list[float] = []
    before_backward_mems: list[float] = []
    peak_mems: list[float] = []

    for _ in range(nsteps):

        forward_start = timer()

        out = f(Q, K, V, mask)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        forward_end = timer()
        torch.cuda.reset_peak_memory_stats()
        before_backward_mem = torch.cuda.memory_allocated()

        loss = out.sum()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        backward_start = timer()
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        backward_end = timer()
        peak_mem = torch.cuda.max_memory_allocated()

        forward_times.append(forward_end - forward_start)
        backward_times.append(backward_end - backward_start)
        before_backward_mems.append(before_backward_mem)
        peak_mems.append(peak_mem)

        Q.grad = None; K.grad = None; V.grad = None

    return mean(forward_times), mean(backward_times), \
        mean(before_backward_mems) / 2**20, mean(peak_mems) / 2**20

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... n d_k"],
    K: Float[Tensor, " ... m d_k"],
    V: Float[Tensor, " ... m d_v"],
    mask: Bool[Tensor, " ... n m"] | None = None,       
) -> Float[Tensor, "... n d_v"]:
    d_k = Q.shape[-1]
    assert d_k == K.shape[-1] 
    assert K.shape[-2] == V.shape[-2] 

    scores: Float[Tensor, "... n m"] = einsum(Q, K, "... n d_k, ... m d_k -> ... n m") / math.sqrt(d_k)
    if mask is not None:
        assert mask.shape[-1] == K.shape[-2]
        assert mask.shape[-2] == Q.shape[-2]

        scores: Float[Tensor, "... n m"] = scores.masked_fill(~mask, float("-inf")) 

    scores: Float[Tensor, "... n m"] = softmax(scores, dim=-1)
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
    p = argparse.ArgumentParser(description="run benchmark")

    p.add_argument("--d-model", type=int)
    p.add_argument("--context-len", type=int)
    p.add_argument("--warm-up", type=int)
    p.add_argument("--nsteps", type=int)
    p.add_argument("--use-torch-compile", action="store_true")

    # Output
    p.add_argument("--output", type=str, default="results/attn/attn_benchmark.jsonl")

    return p.parse_args()

def append_jsonl(output_path: Path , record: dict):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'a', encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def main():

    args = _parse_args()

    print(f"d_model={args.d_model}, context_len={args.context_len}")

    try:
        forward_time, backward_time, \
            before_backward_mem, peak_mem = run_once(
                args.d_model,
                args.context_len,
                args.warm_up,
                args.nsteps,
                args.use_torch_compile
            )
        record = {
            "d_model": args.d_model,
            "context_len": args.context_len,  
            "compiled": bool(args.use_torch_compile),
            "forward_time(s)": forward_time,
            "backward_time(s)": backward_time, 
            "before_backward_mem(M)": before_backward_mem,
            "peak_mem(M)": peak_mem, 
            "status": "OK"        
        }

        print(f"forward_time={forward_time:.4f}, backward_time={backward_time:.4f}, \
          before_backward_mem={before_backward_mem:.3f}, peak_mem={peak_mem:.3f}")
        
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        record = {
            "d_model": args.d_model,
            "context_len": args.context_len,
            "compiled": bool(args.use_torch_compile),
            "status": "OOM",
        }  
        print("OOM Error!")      

    append_jsonl(Path(args.output), record)

if __name__ == "__main__":
    main()

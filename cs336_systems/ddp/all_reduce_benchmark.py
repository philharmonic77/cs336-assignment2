import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from timeit import default_timer as timer

os.environ["GLOO_LOG_LEVEL"] = "ERROR"


def bytes_to_numel(num_bytes, dtype=torch.float32):
    return num_bytes // torch.tensor([], dtype=dtype).element_size()

def generate_data(size_mb, device):
    num_bytes = size_mb * 1024 * 1024
    numel = bytes_to_numel(num_bytes)
    return torch.randn(numel, dtype=torch.float32, device=device)

def setup(rank: int, world_size: int, backend: str) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def benchmark(rank: int, world_size: int, size_mb: int, backend: str):
    # init
    setup(rank, world_size, backend)
    device = torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    data = generate_data(size_mb, device)

    # warmup
    for _ in range(5):
        dist.all_reduce(data, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # measure
    niters = 10
    start = timer()
    for _ in range(niters):
        dist.all_reduce(data, async_op=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = timer() 

    elapsed = (end - start) / niters

    # gather
    times: list[float] = [0.0] * world_size
    dist.all_gather_object(times, elapsed)

    if rank == 0:
        print(f"backend={backend}, size={size_mb}MB, world_size={world_size}")
        print(f"mean={sum(times) / world_size:.6f}s, max={max(times):.6f}s\n")

    dist.destroy_process_group()


def main():

    sizes_mb = [1, 10, 100, 1024] 
    world_sizes = [2, 4, 6]
    if torch.cuda.is_available():
        backend = "nccl"
    else:
        backend = "gloo"

    for size_mb in sizes_mb:
        for world_size in world_sizes:
            mp.spawn(
                fn=benchmark,
                args=(world_size, size_mb, backend),
                nprocs=world_size,
                join=True,
            )

if __name__ == "__main__":
    main()
import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
from contextlib import contextmanager
from pathlib import Path
from cs336_systems.flash_attn.model_builder import build_model, ModelConfig
from cs336_basics.losses import cross_entropy
from cs336_basics.optim import AdamW
from timeit import default_timer as timer
from cs336_systems.ddp.overlap import DDP_BUCKETED

# close mixed precision、use torch.compile


@contextmanager
def cuda_memory_profile(enabled: bool, out_path: str):
    if not enabled:
        yield
        return

    torch.cuda.memory._record_memory_history(
        max_entries=1_000_000,
        stacks="all",
        context="all",
    )
    try:
        yield
    finally:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(out_path)
        torch.cuda.memory._record_memory_history(enabled=None)


def run_bucket_ddp(
    rank,
    world_size,
    backend,
    cfg,
    use_flash,
    warmup,
    nsteps,
    bucket_size_mb,
    seed=123,
    profile_out: str = "",
):
    torch.manual_seed(seed) # for model param init
    setup(rank, world_size, backend)

    device = torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

    should_profile = (device.type == "cuda") and (rank == 0) and bool(profile_out)

    # ---- memory stats ----
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with cuda_memory_profile(should_profile, profile_out):
        model = build_model(cfg, use_flash=use_flash)
        model.to(device)
        model = DDP_BUCKETED(model, bucket_size_mb)

        if device.type == "cuda":
            model = torch.compile(model)

        optimizer = AdamW(model.parameters(), lr=cfg.lr)

        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

        # ---- warmup (no logging) ----
        for _ in range(warmup):
            x, y = generate_and_scatter_data(rank, world_size, cfg, device, gen)
            _ = train_step(model, optimizer, x, y)

        # ---- measure ----
        total_time_acc = 0.0
        last_loss = None

        for _ in range(nsteps):
            x, y = generate_and_scatter_data(rank, world_size, cfg, device, gen)
            total_time, loss = train_step(model, optimizer, x, y)
            total_time_acc += total_time
            last_loss = loss

    total_tensor = torch.tensor(total_time_acc / nsteps, device=device)
    loss_tensor = torch.tensor(last_loss, device=device)

    dist.all_reduce(total_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    loss_tensor /= world_size

    peak_mem = 0.0
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024**3)  # GiB
    peak_tensor = torch.tensor(peak_mem, device=device)
    dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)

    dist.barrier()
    if rank == 0:
        # torch.save(model.state_dict(), "ddp.pt")

        print(f"Avg step time: {total_tensor.item():.4f}s")
        print(f"Last loss: {loss_tensor.item():.6f}")
        print(f"Peak mem: {peak_tensor.item():.2f} GiB")
    dist.barrier()
    dist.destroy_process_group()

    return model 


def generate_data_batch(cfg, device, generator):
    x = torch.randint(
        0, cfg.vocab_size,
        (cfg.batch_size, cfg.context_len + 1),
        device=device,
        dtype=torch.long,
        generator=generator
    )
    y = x[:, 1:]
    x = x[:, :-1]
    return x, y 

def generate_and_scatter_data(rank, world_size, cfg, device, generator):
    assert cfg.batch_size % world_size == 0

    if rank == 0:
        global_x, global_y = generate_data_batch(cfg, device, generator)

        # x_chunks = list(global_x.chunk(world_size, dim=0))
        # y_chunks = list(global_y.chunk(world_size, dim=0))
        x_chunks = [c.contiguous() for c in global_x.chunk(world_size, dim=0)]
        y_chunks = [c.contiguous() for c in global_y.chunk(world_size, dim=0)]
    else:
        x_chunks = None
        y_chunks = None

    x = torch.empty(cfg.batch_size // world_size, cfg.context_len, dtype=torch.long, device=device)
    y = torch.empty(cfg.batch_size // world_size, cfg.context_len, dtype=torch.long, device=device)
    dist.scatter(x, x_chunks, src=0)
    dist.scatter(y, y_chunks, src=0)

    return x, y

def setup(rank: int, world_size: int, backend: str) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
            

def train_step(model, optimizer, x, y):
    start_total = timer()

    model.start_train_batch()

    logits = model(x)
    loss = cross_entropy(logits, y)
    loss.backward()

    model.finish_gradient_synchronization()
    
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    total_time = timer() - start_total

    return total_time, loss.item()

       

def main(bucket_size, profile_out: str = ""):

    world_size = 2
    if torch.cuda.is_available():
        backend = "nccl"
    else:
        backend = "gloo"
    
    cfg = ModelConfig()
    warmup = 5
    nsteps = 10
    seed = 123

    mp.spawn(
        fn=run_bucket_ddp,
        args=(world_size, backend, cfg, True, warmup, nsteps, bucket_size, seed, profile_out),
        nprocs=world_size,
        join=True,
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDP bucketed benchmark")
    parser.add_argument(
        "--bucket-size-mb",
        type=int,
        required=True,
        help="Bucket size in MB.",
    )
    parser.add_argument(
        "--profile-out",
        type=str,
        default="",
        help="Output snapshot path. If empty, profiling is disabled.",
    )
    args = parser.parse_args()

    main(args.bucket_size_mb, profile_out=args.profile_out)

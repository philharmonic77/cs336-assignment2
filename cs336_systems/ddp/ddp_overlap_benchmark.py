import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_systems.flash_attn.model_builder import build_model, ModelConfig
from cs336_basics.losses import cross_entropy
from cs336_basics.optim import AdamW
from timeit import default_timer as timer
from cs336_systems.ddp.overlap import DDP

# close mixed precision、use torch.compile



def run_overlap_ddp(rank, world_size, backend, cfg, use_flash, warmup, nsteps, seed=123):
    torch.manual_seed(seed) # for model param init
    setup(rank, world_size, backend)

    device = torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

    model = build_model(cfg, use_flash=use_flash)
    model.to(device)
    model = DDP(model)

    if device.type == "cuda":
        model = torch.compile(model)

    optimizer = AdamW(model.parameters(), lr=cfg.lr)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # ---- memory stats ----
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # ---- warmup (no logging) ----
    for _ in range(warmup):

        x, y = generate_and_scatter_data(rank, world_size, cfg, device, gen)
        _ = train_step(model, optimizer, x, y, device, world_size)

    # ---- measure ----
    total_time_acc = 0.0
    comm_time_acc = 0.0
    pack_time_acc = 0.0
    allreduce_time_acc = 0.0
    unpack_time_acc = 0.0
    last_loss = None

    for _ in range(nsteps):
        x, y = generate_and_scatter_data(rank, world_size, cfg, device, gen)
        total_time, loss = train_step(
            model, optimizer, x, y, device, world_size
        )
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
        torch.save(model.state_dict(), "ddp.pt")

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

    logits = model(x)
    loss = cross_entropy(logits, y)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    total_time = timer() - start_total

    return total_time, loss.item()

       

def main():

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
        fn=run_overlap_ddp,
        args=(world_size, backend, cfg, True, warmup, nsteps, seed),
        nprocs=world_size,
        join=True,
    )




if __name__ == "__main__":
    main()

import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_systems.flash_attn.model_builder import build_model, ModelConfig
from cs336_basics.losses import cross_entropy
from cs336_basics.optim import AdamW
from timeit import default_timer as timer
from torch._utils import (
    _flatten_dense_tensors,
    _unflatten_dense_tensors,
)
# close mixed precision、torch.compile



def run_naive_ddp(rank, world_size, backend, cfg, use_flash, warmup, nsteps, seed=123):
    torch.manual_seed(seed) # for model param init
    setup(rank, world_size, backend)

    device = torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

    model = build_model(cfg, use_flash=use_flash).to(device)
    init_model_and_broadcast(model, rank, src=0)
    # if device.type == "cuda":
    #     model = torch.compile(model)

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
        total_time, comm_time, pack_time, allreduce_time, unpack_time, loss = train_step(
            model, optimizer, x, y, device, world_size
        )
        total_time_acc += total_time
        comm_time_acc += comm_time
        pack_time_acc += pack_time
        allreduce_time_acc += allreduce_time
        unpack_time_acc += unpack_time

        last_loss = loss

    # Worst-case (slowest rank) step/comm time is what determines wall-clock iteration time.
    total_tensor = torch.tensor(total_time_acc / nsteps, device=device)
    comm_tensor = torch.tensor(comm_time_acc / nsteps, device=device)
    pack_tensor = torch.tensor(pack_time_acc / nsteps, device=device)
    allreduce_tensor = torch.tensor(allreduce_time_acc / nsteps, device=device)
    unpack_tensor = torch.tensor(unpack_time_acc / nsteps, device=device)

    loss_tensor = torch.tensor(last_loss, device=device)

    dist.all_reduce(total_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(comm_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(pack_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(allreduce_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(unpack_tensor, op=dist.ReduceOp.MAX)

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
        print(f"Avg comm time: {comm_tensor.item():.4f}s")
        print(f"Pack time:      {pack_tensor:.6f}s")
        print(f"AllReduce time: {allreduce_tensor:.6f}s")
        print(f"Unpack time:    {unpack_tensor:.6f}s")
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

def init_model_and_broadcast(model, rank, src):
    if rank == src:
        pass
    with torch.no_grad():
        for p in model.parameters():
            dist.broadcast(p.data, src=src)
        for b in model.buffers():
            dist.broadcast(b.data, src=src)
            

def train_step(model, optimizer, x, y, device, world_size=None):
    start_total = timer()

    logits = model(x)
    loss = cross_entropy(logits, y)

    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()

    comm_time = 0.0
    pack_time = 0.0
    allreduce_time = 0.0
    unpack_time = 0.0

    if world_size is not None:

        # -------- PACK --------
        t0 = timer()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        flat = _flatten_dense_tensors(grads)
        if device.type == "cuda":
            torch.cuda.synchronize()
        pack_time = timer() - t0

        # -------- ALLREDUCE --------
        t1 = timer()
        dist.all_reduce(flat)
        if device.type == "cuda":
            torch.cuda.synchronize()

        flat /= world_size

        allreduce_time = timer() - t1

        # -------- UNPACK --------
        t2 = timer()
        for grad, synced in zip(grads, _unflatten_dense_tensors(flat, grads)):
            grad.copy_(synced)
        if device.type == "cuda":
            torch.cuda.synchronize()
        unpack_time = timer() - t2

        comm_time = pack_time + allreduce_time + unpack_time

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_time = timer() - start_total

    return total_time, comm_time, pack_time, allreduce_time, unpack_time, loss.item()

       

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
        fn=run_naive_ddp,
        args=(world_size, backend, cfg, True, warmup, nsteps, seed),
        nprocs=world_size,
        join=True,
    )




if __name__ == "__main__":
    main()

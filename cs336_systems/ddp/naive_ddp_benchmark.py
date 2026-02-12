import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_systems.flash_attn.model_builder import build_model, ModelConfig
from cs336_basics.losses import cross_entropy
from cs336_basics.optim import AdamW
from timeit import default_timer as timer
# use mixed precision、torch.compile



def run_naive_ddp(rank, world_size, backend, cfg, use_flash, warmup, nsteps, seed=123):
    torch.manual_seed(seed) # for model param init
    setup(rank, world_size, backend)

    device = torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

    model = build_model(cfg, use_flash=use_flash).to(device)
    init_model_and_broadcast(model, rank, src=0)
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
    last_loss = None

    for _ in range(nsteps):
        x, y = generate_and_scatter_data(rank, world_size, cfg, device, gen)
        total_time, comm_time, loss = train_step(
            model, optimizer, x, y, device, world_size
        )
        total_time_acc += total_time
        comm_time_acc += comm_time
        last_loss = loss

    # Worst-case (slowest rank) step/comm time is what determines wall-clock iteration time.
    total_tensor = torch.tensor(total_time_acc / nsteps, device=device)
    comm_tensor = torch.tensor(comm_time_acc / nsteps, device=device)
    loss_tensor = torch.tensor(last_loss, device=device)

    dist.all_reduce(total_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(comm_tensor, op=dist.ReduceOp.MAX)
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
        print(f"Avg comm time: {comm_tensor.item():.4f}s")
        print(f"Last loss: {loss_tensor.item():.6f}")
        print(f"Peak mem: {peak_tensor.item():.2f} GiB")
    dist.barrier()
    dist.destroy_process_group()

    return model 


def run_single(backend, cfg, use_flash, warmup, nsteps, seed=123):
    torch.manual_seed(seed) # for model param init

    device = torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(0)
        device = torch.device(f"cuda:{0}")

    model = build_model(cfg, use_flash=use_flash).to(device)
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
        x, y = generate_data_batch(cfg, device, gen)
        _ = train_step(model, optimizer, x, y, device)

    # ---- measure ----
    total_time_acc = 0.0
    last_loss = None

    for _ in range(nsteps):
        x, y = generate_data_batch(cfg, device, gen)
        total_time, _, loss = train_step(model, optimizer, x, y, device)
        total_time_acc += total_time
        last_loss = loss

    print(f"Avg step time: {total_time_acc / nsteps:.4f}s")
    print(f"Last loss: {last_loss:.6f}")

    peak_mem = 0.0
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024**3)  # GiB

    if device.type == "cuda":
        print(f"Peak mem: {peak_mem:.2f} GiB")

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

        x_chunks = list(global_x.chunk(world_size, dim=0))
        y_chunks = list(global_y.chunk(world_size, dim=0))
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

    # with torch.autocast(
    #     device_type=device.type,
    #     dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    #     enabled=(device.type == "cuda"),
    # ):
    #     logits = model(x)
    #     loss = cross_entropy(logits, y)
    logits = model(x)
    loss = cross_entropy(logits, y)    
    loss.backward()
    torch.cuda.synchronize() if device.type == "cuda" else None

    comm_time = 0.0

    if world_size is not None:
        start_comm = timer()
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad)
                p.grad /= world_size

        torch.cuda.synchronize() if device.type == "cuda" else None
        comm_time = timer() - start_comm

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize() if device.type == "cuda" else None
    total_time = timer() - start_total

    return total_time, comm_time, loss.item()

    
def compare_models(m1, m2, atol=1e-4, rtol=1e-4):
    m1 = (m1._orig_mod if hasattr(m1, "_orig_mod") else m1).to("cpu").float()
    m2 = (m2._orig_mod if hasattr(m2, "_orig_mod") else m2).to("cpu").float()

    sd1 = m1.state_dict()
    sd2 = m2.state_dict()

    if sd1.keys() != sd2.keys():
        print("State dict keys mismatch.")
        return False

    worst = 0.0
    worst_name = None
    mismatched = 0

    for k in sd1:
        t1, t2 = sd1[k], sd2[k]
        if not torch.is_floating_point(t1):
            continue
        diff = (t1 - t2).abs().max().item()
        if diff > worst:
            worst = diff
            worst_name = k
        if not torch.allclose(t1, t2, atol=atol, rtol=rtol):
            mismatched += 1

    if mismatched == 0:
        print(f"Models match. Worst max diff={worst:.6g} at {worst_name}.")
        return True

    print(f"Models NOT match. Mismatched {mismatched} tensors. "
          f"Worst max diff={worst:.6g} at {worst_name}.")
    return False
    

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

    print("Single GPU + flash:")
    model_use_flash = run_single(backend, cfg, use_flash=True, warmup=warmup, nsteps=nsteps, seed=seed)

    print("Single GPU + no flash:")
    model_no_flash = run_single(backend, cfg, use_flash=False, warmup=warmup, nsteps=nsteps, seed=seed)

    print("Comparing: single_flash vs single_no_flash")
    compare_models(model_use_flash, model_no_flash)

    print("Multi GPU + flash:")
    del model_no_flash
    torch.cuda.empty_cache()

    mp.spawn(
        fn=run_naive_ddp,
        args=(world_size, backend, cfg, True, warmup, nsteps, seed),
        nprocs=world_size,
        join=True,
    )

    ddp_model = build_model(cfg, use_flash=True)
    sd = torch.load("ddp.pt", map_location="cpu")
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    ddp_model.load_state_dict(sd)

    print("Comparing: single_flash vs ddp_flash")
    compare_models(model_use_flash, ddp_model)

    if os.path.exists("ddp.pt"):
        os.remove("ddp.pt")


if __name__ == "__main__":
    main()

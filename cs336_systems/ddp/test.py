import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_systems.flash_attn.model_builder import build_model, ModelConfig
from cs336_basics.losses import cross_entropy
from cs336_basics.optim import AdamW
from timeit import default_timer as timer

# =========================
# Debug helpers (strong checks)
# =========================

def _tensor_fingerprint(t: torch.Tensor) -> tuple[int, int, int]:
    """
    Strong but cheap-ish fingerprint:
      - sum of all elements (int)
      - first element (int)
      - last element (int)
    Assumes integer tensor (x/y). We keep it identical to what you already used.
    """
    t = t.detach()
    s = int(t.sum().item())
    f = int(t.reshape(-1)[0].item())
    l = int(t.reshape(-1)[-1].item())
    return s, f, l

def _model_fingerprint(model: torch.nn.Module) -> tuple[float, float]:
    """
    Strong-ish parameter fingerprint:
      - sum over all float params (float64 accumulation)
      - sum over all float params squared (float64 accumulation)
    This catches almost any init mismatch.
    """
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    with torch.no_grad():
        s1 = 0.0
        s2 = 0.0
        for p in m.parameters():
            if not torch.is_floating_point(p):
                continue
            x = p.detach().to(dtype=torch.float64, device="cpu")
            s1 += float(x.sum().item())
            s2 += float((x * x).sum().item())
    return s1, s2

def _assert_close(a: float, b: float, name: str, atol: float = 0.0, rtol: float = 0.0):
    ok = abs(a - b) <= (atol + rtol * abs(b))
    if not ok:
        raise AssertionError(f"{name} mismatch: {a} vs {b} (atol={atol}, rtol={rtol})")

# =========================
# Distributed setup
# =========================

def setup(rank: int, world_size: int, backend: str) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def init_model_and_broadcast(model, rank, src):
    with torch.no_grad():
        for p in model.parameters():
            dist.broadcast(p.data, src=src)
        for b in model.buffers():
            dist.broadcast(b.data, src=src)

# =========================
# Data generation (with strong validation)
# =========================

def generate_data_batch(cfg, device, generator):
    x = torch.randint(
        0, cfg.vocab_size,
        (cfg.batch_size, cfg.context_len + 1),
        device=device,
        dtype=torch.long,
        generator=generator,
    )
    y = x[:, 1:]
    x = x[:, :-1]
    return x, y

def generate_and_scatter_data(rank, world_size, cfg, device, generator, step: int):
    assert cfg.batch_size % world_size == 0

    if rank == 0:
        global_x, global_y = generate_data_batch(cfg, device, generator)

        # global fingerprints
        gx = _tensor_fingerprint(global_x)
        gy = _tensor_fingerprint(global_y)

        # pre-chunk sum check (this should always match exactly)
        x_chunks = list(global_x.chunk(world_size, dim=0))
        y_chunks = list(global_y.chunk(world_size, dim=0))
        cx_sum = sum(int(c.sum().item()) for c in x_chunks)
        cy_sum = sum(int(c.sum().item()) for c in y_chunks)

        print(f"[DDP step {step}] GLOBAL x(fp={gx}) y(fp={gy}) | chunk_sum x={cx_sum} y={cy_sum}")
        if cx_sum != gx[0] or cy_sum != gy[0]:
            raise AssertionError(f"[DDP step {step}] pre-scatter chunk sum mismatch")

        # ship fingerprints to other ranks for checking
        meta = torch.tensor([gx[0], gx[1], gx[2], gy[0], gy[1], gy[2]], device=device, dtype=torch.long)
    else:
        x_chunks = None
        y_chunks = None
        meta = torch.empty(6, device=device, dtype=torch.long)

    # broadcast global fingerprints to all ranks
    dist.broadcast(meta, src=0)
    gx0, gx1, gx2, gy0, gy1, gy2 = [int(v.item()) for v in meta]

    # scatter
    x = torch.empty(cfg.batch_size // world_size, cfg.context_len, dtype=torch.long, device=device)
    y = torch.empty(cfg.batch_size // world_size, cfg.context_len, dtype=torch.long, device=device)
    dist.scatter(x, x_chunks, src=0)
    dist.scatter(y, y_chunks, src=0)

    # local fingerprints
    lx = _tensor_fingerprint(x)
    ly = _tensor_fingerprint(y)
    print(f"[DDP step {step}] RANK{rank} x(fp={lx}) y(fp={ly})")

    # post-scatter global sum check: sum of local sums must equal global sum
    local_sums = torch.tensor([lx[0], ly[0]], device=device, dtype=torch.long)
    dist.all_reduce(local_sums, op=dist.ReduceOp.SUM)
    if rank == 0:
        sx, sy = [int(v.item()) for v in local_sums]
        if sx != gx0 or sy != gy0:
            raise AssertionError(
                f"[DDP step {step}] post-scatter sum mismatch: "
                f"global_x_sum={gx0} sum(local_x)={sx} | global_y_sum={gy0} sum(local_y)={sy}"
            )
        print(f"[DDP step {step}] POST-SCATTER SUM OK: x={sx} y={sy}")

    return x, y

# =========================
# Train step (no autocast, no compile)
# =========================

def train_step(model, optimizer, x, y, device, world_size=None):
    start_total = timer()

    logits = model(x)
    loss = cross_entropy(logits, y)

    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    comm_time = 0.0
    if world_size is not None:
        start_comm = timer()
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad)
                p.grad /= world_size
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        comm_time = timer() - start_comm

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_time = timer() - start_total
    return total_time, comm_time, float(loss.item())

# =========================
# Compare models (unchanged logic)
# =========================

def compare_models(m1, m2, atol=1e-6, rtol=1e-6):
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

# =========================
# Single / DDP runners (2 steps total, strong init+data checks)
# =========================

def run_single(backend, cfg, use_flash, seed=123, nsteps=2):
    torch.manual_seed(seed)

    device = torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")

    model = build_model(cfg, use_flash=use_flash).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr)

    # generator on the SAME device as x (important for cuda)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # init fingerprint
    p1, p2 = _model_fingerprint(model)
    print(f"[SINGLE(use_flash={use_flash})] INIT param_fp: sum={p1:.6f} sumsq={p2:.6f}")

    # run exactly nsteps, print per-step data fp + loss
    for step in range(nsteps):
        x, y = generate_data_batch(cfg, device, gen)
        fx = _tensor_fingerprint(x)
        fy = _tensor_fingerprint(y)
        print(f"[SINGLE(use_flash={use_flash}) step {step}] x(fp={fx}) y(fp={fy})")

        _, _, loss = train_step(model, optimizer, x, y, device)
        print(f"[SINGLE(use_flash={use_flash}) step {step}] loss={loss:.6f}")

    return model

def run_naive_ddp(rank, world_size, backend, cfg, use_flash, seed=123, nsteps=2):
    torch.manual_seed(seed)
    setup(rank, world_size, backend)

    device = torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

    model = build_model(cfg, use_flash=use_flash).to(device)
    init_model_and_broadcast(model, rank, src=0)
    optimizer = AdamW(model.parameters(), lr=cfg.lr)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # init fingerprint per-rank, then assert equal across ranks
    p1, p2 = _model_fingerprint(model)
    fp = torch.tensor([p1, p2], device=device, dtype=torch.float64)
    dist.all_reduce(fp, op=dist.ReduceOp.MAX)
    fp_max = fp.clone()
    fp = torch.tensor([p1, p2], device=device, dtype=torch.float64)
    dist.all_reduce(fp, op=dist.ReduceOp.MIN)
    fp_min = fp.clone()

    if rank == 0:
        print(f"[DDP] INIT param_fp MIN: sum={fp_min[0].item():.6f} sumsq={fp_min[1].item():.6f}")
        print(f"[DDP] INIT param_fp MAX: sum={fp_max[0].item():.6f} sumsq={fp_max[1].item():.6f}")
        _assert_close(fp_min[0].item(), fp_max[0].item(), "DDP init param_sum", atol=0.0, rtol=0.0)
        _assert_close(fp_min[1].item(), fp_max[1].item(), "DDP init param_sumsq", atol=0.0, rtol=0.0)
        print("[DDP] INIT param_fp EXACT MATCH across ranks")

    # run exactly nsteps, with strong per-step data validation
    for step in range(nsteps):
        x, y = generate_and_scatter_data(rank, world_size, cfg, device, gen, step=step)
        _, _, loss = train_step(model, optimizer, x, y, device, world_size)
        print(f"[DDP step {step}] RANK{rank} loss={loss:.6f}")

    dist.barrier()
    if rank == 0:
        torch.save(model.state_dict(), "ddp.pt")
    dist.barrier()
    dist.destroy_process_group()
    return model

# =========================
# Main
# =========================

def main():
    world_size = 2
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    cfg = ModelConfig()
    seed = 123
    nsteps = 2

    print("Single GPU + flash:")
    model_use_flash = run_single(backend, cfg, use_flash=True, seed=seed, nsteps=nsteps)

    print("Single GPU + no flash:")
    model_no_flash = run_single(backend, cfg, use_flash=False, seed=seed, nsteps=nsteps)

    print("Comparing: single_flash vs single_no_flash")
    compare_models(model_use_flash, model_no_flash)

    print("Multi GPU + flash:")
    del model_no_flash
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mp.spawn(
        fn=run_naive_ddp,
        args=(world_size, backend, cfg, True, seed, nsteps),
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
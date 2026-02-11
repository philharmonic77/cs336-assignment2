import torch 
import os
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim import SGD


def setup(rank: int, world_size: int, backend: str) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)



def run_naive_ddp(rank, world_size, backend, batch_size, d1, d2, d3, nsteps, seed=123):
    torch.manual_seed(seed) # for model param init
    setup(rank, world_size, backend)

    device = torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

    model = ToyModel(d1, d2, d3).to(device)
    init_model_and_broadcast(model, rank, src=0)

    optimizer = SGD(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    gen = torch.Generator()
    gen.manual_seed(seed)

    for _ in range(nsteps):

        x, y = generate_and_scatter_data(rank, world_size, batch_size, d1, d3, device, gen)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad)
                p.grad /= world_size

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if rank == 0:
        torch.save(model.state_dict(), "ddp.pt")

    dist.destroy_process_group()
    return model 


def run_single(backend, batch_size, d1, d2, d3, nsteps, seed=123):
    torch.manual_seed(seed) # for model param init

    device = torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(0)
        device = torch.device(f"cuda:{0}")

    model = ToyModel(d1, d2, d3).to(device)
    optimizer = SGD(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    gen = torch.Generator()
    gen.manual_seed(seed)

    for _ in range(nsteps):
        x, y = get_batch(batch_size, d1, d3, device, gen)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return model

def get_batch(batch_size, d1, d3, device, generator):
    x = torch.randn(batch_size, d1, dtype=torch.float32, device=device, generator=generator)
    y = torch.randint(0, d3, (batch_size,), dtype=torch.long, device=device, generator=generator)   
    return x, y  


def generate_and_scatter_data(rank, world_size, batch_size, d1, d3, device, generator):
    assert batch_size % world_size == 0

    if rank == 0:
        global_x, global_y = get_batch(batch_size, d1, d3, device, generator)

        x_chunks = list(global_x.chunk(world_size, dim=0))
        y_chunks = list(global_y.chunk(world_size, dim=0))
    else:
        x_chunks = None
        y_chunks = None

    x = torch.empty(batch_size // world_size, d1, device=device)
    y = torch.empty(batch_size // world_size, dtype=torch.long, device=device)
    dist.scatter(x, x_chunks, src=0)
    dist.scatter(y, y_chunks, src=0)

    return x, y

def init_model_and_broadcast(model, rank, src):
    if rank == src:
        pass
    with torch.no_grad():
        for p in model.parameters():
            dist.broadcast(p.data, src=src)
        for b in model.buffers():
            dist.broadcast(b.data, src=src)



class ToyModel(nn.Module):
    def __init__(self, d1, d2, d3):
        super().__init__()
        self.fc1 = nn.Linear(d1, d2, bias=False)        
        self.fc2 = nn.Linear(d2, d3, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def compare_models(m1, m2, atol=1e-6):
    for (n1, p1), (n2, p2) in zip(
        m1.named_parameters(), m2.named_parameters()
    ):
        assert n1 == n2
        if not torch.allclose(p1, p2, atol=atol):
            max_diff = (p1 - p2).abs().max().item()
            print(f"Mismatch in {n1}, max diff = {max_diff}")
            return False
    print("Models match!")
    return True
    

def main():

    world_size = 2
    if torch.cuda.is_available():
        backend = "nccl"
    else:
        backend = "gloo"

    batch_size = 128
    d1, d2, d3 = 50, 100, 10
    nsteps = 5
    seed = 123


    mp.spawn(
        fn=run_naive_ddp,
        args=(world_size, backend, batch_size, d1, d2, d3, nsteps, seed),
        nprocs=world_size,
        join=True,
    )

    single_model = run_single(backend, batch_size, d1, d2, d3, nsteps, seed)
    ddp_model = ToyModel(d1, d2, d3).to(single_model.fc1.weight.device)
    ddp_model.load_state_dict(torch.load("ddp.pt"))

    compare_models(single_model, ddp_model)

    if os.path.exists("ddp.pt"):
        os.remove("ddp.pt")

if __name__ == "__main__":
    main()

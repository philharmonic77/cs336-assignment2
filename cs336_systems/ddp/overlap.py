import torch 
import torch.distributed as dist





class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        """
        Given an instantiated PyTorch nn.Module to be parallelized, construct a DDP container that will handle gradient synchronization across ranks.
        """
        super().__init__()
        self.module = module 
        self.world_size = dist.get_world_size()
        self.handles = []

        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)
        for b in self.module.buffers():
            dist.broadcast(b.data, src=0)

        for p in module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(lambda p: self.hook(p))

    def hook(self, p):
        handle = dist.all_reduce(p.grad, async_op=True)
        self.handles.append(handle)


    def forward(self, *inputs, **kwargs): 
        """
        Calls the wrapped module’s forward() method with the provided positional and keyword arguments.
        """
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self): 
        """
        When called, wait for asynchronous communication calls to be queued on GPU.
        """
        for h in self.handles:
            h.wait()

        for p in self.module.parameters():
            if p.grad is not None:
                p.grad.div_(self.world_size)
        self.handles.clear()

        


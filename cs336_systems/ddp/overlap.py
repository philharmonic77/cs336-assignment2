import torch 
import torch.distributed as dist
from torch._utils import (
    _flatten_dense_tensors,
    _unflatten_dense_tensors,
)


class DDP_BUCKETED(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float): 
        """
        Given an instantiated PyTorch nn.Module to be parallelized, construct a DDP container that will handle gradient syn-chronization across ranks. Gradient synchronization should be bucketed, with each bucket holding at most bucket_size_mb of parameters.
        """
        super().__init__()
        self.module = module 
        self.world_size = dist.get_world_size()
        self.handles = []
        self.bucket_size_mb = bucket_size_mb

        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)
        for b in self.module.buffers():
            dist.broadcast(b.data, src=0)

        # build bucket
        buckets_list = self._build_bucket()
        self.buckets = []
        self.param_to_bucket = {}

        for bucket_id, bucket_params in enumerate(buckets_list):
            bucket = {
                "params": bucket_params,
                "ready": 0,
                "size": len(bucket_params),
                "handle": None,
            }
            self.buckets.append(bucket)

            for p in bucket_params:
                self.param_to_bucket[p] = bucket_id        

        # register hook
        for p in module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._make_hook()) 

    def _make_hook(self):
        def hook(param):
            bucket_id = self.param_to_bucket[param]
            bucket = self.buckets[bucket_id]

            bucket["ready"] += 1

            if bucket["ready"] == bucket["size"]:
                self._allreduce_bucket(bucket)

        return hook
    
    def _allreduce_bucket(self, bucket):
        grads = [p.grad for p in bucket["params"]]
        flat = _flatten_dense_tensors(grads)

        handle = dist.all_reduce(flat, async_op=True)

        bucket["handle"] = handle
        bucket["flat"] = flat
        bucket["grads"] = grads

    def _build_bucket(self):
        bucket_size_bytes = self.bucket_size_mb * 1024 * 1024
        buckets_list = []

        current_bucket = []
        current_bytes = 0
        for p in reversed(list(self.module.parameters())):

            if not p.requires_grad:
                continue

            p_bytes = p.element_size() * p.numel()

            if p_bytes > bucket_size_bytes:
                if current_bucket:
                    buckets_list.append(current_bucket)
                    current_bucket = []
                    current_bytes = 0
                buckets_list.append([p])
                continue

            if current_bytes + p_bytes > bucket_size_bytes:
                buckets_list.append(current_bucket)
                current_bucket = [p]
                current_bytes = p_bytes
            else:
                current_bucket.append(p)
                current_bytes += p_bytes

        if current_bucket:
            buckets_list.append(current_bucket)

        return buckets_list


    def forward(self, *inputs, **kwargs): 
        """
        Calls the wrapped module’s forward() method with the provided positional and keyword arguments.
        """
        if len(inputs) == 0 and len(kwargs) == 0:
            return self
        return self.module(*inputs, **kwargs) 
             
    def finish_gradient_synchronization(self): 
        """
        When called, wait for asynchronous communication calls to be queued on GPU.
        """
        for bucket in self.buckets:
            if bucket["handle"] is not None:
                bucket["handle"].wait()

            for grad, synced in zip(bucket["grads"], _unflatten_dense_tensors(bucket["flat"], bucket["grads"])):
                grad.copy_(synced)

            for p in bucket["params"]:

                if p.grad is not None:
                    p.grad.div_(self.world_size)

            bucket["handle"] = None
            bucket["ready"] = 0

    def start_train_batch(self):
        for bucket in self.buckets:
            bucket["ready"] = 0
            bucket["handle"] = None
            if "flat" in bucket:
                del bucket["flat"]
            if "grads" in bucket:
                del bucket["grads"]
        


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
                p.register_post_accumulate_grad_hook(lambda p: self._hook(p))

    def _hook(self, p):
        handle = dist.all_reduce(p.grad, async_op=True)
        self.handles.append(handle)


    def forward(self, *inputs, **kwargs): 
        """
        Calls the wrapped module’s forward() method with the provided positional and keyword arguments.
        """
        if len(inputs) == 0 and len(kwargs) == 0:
            return self
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

        


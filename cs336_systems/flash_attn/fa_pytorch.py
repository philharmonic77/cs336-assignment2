import torch
import math

class FlashAttentionPytorchFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # handle is_causal
        if is_causal:
            pass
        
        # flatten
        *batch_dims, Nq, d = Q.shape
        *batch_dims_k, Nk, dk = K.shape
        *batch_dims_v, Nv, dv = V.shape
        assert batch_dims == batch_dims_k == batch_dims_v
        assert Nk == Nv
        assert d == dk

        ctx.batch_dims = batch_dims

        B = math.prod(batch_dims) if batch_dims else 1
        Q = Q.reshape(B, Nq, d)
        K = K.reshape(B, Nk, d)
        V = V.reshape(B, Nk, dv)

        # init
        Bq, Bk = 64, 64
        scale = 1.0 / math.sqrt(d)
        O = torch.empty(B, Nq, dv, device=Q.device, dtype=Q.dtype)
        L = torch.empty(B, Nq, device=Q.device, dtype=Q.dtype)
        
        for i in range(0, Nq, Bq):
            qs, qe = i, min(i + Bq, Nq)
            Q_i = Q[:, qs: qe, :] # (B, bq, d)
            bq = qe - qs

            O_i = torch.zeros(B, bq, dv, device=Q.device, dtype=Q.dtype)
            l_i = torch.zeros(B, bq, device=Q.device, dtype=Q.dtype)
            m_i = torch.empty(B, bq, device=Q.device, dtype=Q.dtype).fill_(float('-inf'))

            for j in range(0, Nk, Bk):
                ks, ke = j, min(j + Bk, Nk)
                K_j = K[:, ks: ke, :] # (B, bk, d)
                V_j = V[:, ks: ke, :] # (B, bk, dv)

                # calc attn / online softmax
                S_i = torch.matmul(Q_i, K_j.transpose(-1, -2)) * scale # (B, bq, bk)
                m_i_new = torch.maximum(m_i, S_i.max(dim=-1).values) # (B, bq)
                P_i= torch.exp(S_i - m_i_new.unsqueeze(-1)) # (B, bq, bk)
                factor = torch.exp(m_i - m_i_new) # (B, bq)
                l_i_new = factor * l_i + P_i.sum(dim=-1) # (B, bq)
                O_i_new = factor.unsqueeze(-1) * O_i + torch.matmul(P_i, V_j) # (B, bq, dv)

                m_i = m_i_new
                l_i = l_i_new
                O_i = O_i_new 
            
            O[:, qs: qe, :] = O_i / l_i.unsqueeze(-1) # (B, bq, dv)
            L[:, qs: qe] = m_i + torch.log(l_i) # (B, bq)
      
        ctx.save_for_backward(Q, K, V, O, L)

        return O.reshape(*batch_dims, Nq, dv)

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError
    
def cdiv(a, b):
    return (a + b - 1) // b
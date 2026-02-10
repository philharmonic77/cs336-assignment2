from typing import Any
import torch
import triton # type: ignore
import triton.language as tl # type: ignore
from triton import cdiv # type: ignore
import math


@triton.jit
def flash_attention_fwd(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE:  tl.constexpr
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    # Initialize a buffer to accumulate
    O = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    m = tl.full((Q_TILE_SIZE,), -float("inf"), tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), tl.float32)

    Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (Q_TILE_SIZE, D)

    for _ in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)
        V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)

        S = tl.dot(Q, tl.trans(K)) * scale # (Q_TILE_SIZE, K_TILE_SIZE)
        m_new = tl.maximum(m, tl.max(S, axis=1)) # (Q_TILE_SIZE, ）

        P = tl.exp(S - m_new[:, None]) # (Q_TILE_SIZE, K_TILE_SIZE)

        factor = tl.exp(m - m_new) # (Q_TILE_SIZE, ）
        l_new = factor * l + tl.sum(P, axis=1) # (Q_TILE_SIZE, ）

        P = P.to(V_block_ptr.type.element_ty)
        O = tl.dot(P, V, acc=O * factor[:, None]) # (Q_TILE_SIZE, D)

        m = m_new
        l = l_new 

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0)) 
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0)) 

    O = (O / l[:, None]).to(O_block_ptr.type.element_ty)
    tl.store(O_block_ptr, O, boundary_check=(0, 1))
    tl.store(L_block_ptr, m + tl.log(l), boundary_check=(0, ))



class FlashAttentionTritonFunc(torch.autograd.Function):
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
        assert d == dk == dv

        ctx.batch_dims = batch_dims

        B = math.prod(batch_dims) if batch_dims else 1
        Q = Q.reshape(B, Nq, d)
        K = K.reshape(B, Nk, d)
        V = V.reshape(B, Nk, d)   

        # init
        ctx.Bq, ctx.Bk = 64, 64
        scale = 1.0 / math.sqrt(d)  
        O = torch.empty(B, Nq, d, device=Q.device, dtype=Q.dtype)
        L = torch.empty(B, Nq, device=Q.device, dtype=torch.float32)

        
        flash_attention_fwd[(cdiv(Nq, ctx.Bq), B)]( # type: ignore
                Q, K, V,
                O, L,
                Q.stride(0), Q.stride(1), Q.stride(2),
                K.stride(0), K.stride(1), K.stride(2),
                V.stride(0), V.stride(1), V.stride(2),
                O.stride(0), O.stride(1), O.stride(2),
                L.stride(0), L.stride(1),
                Nq, Nk,
                scale,
                D=d,
                Q_TILE_SIZE=ctx.Bq,
                K_TILE_SIZE=ctx.Bk,
        )

        ctx.save_for_backward(Q, K, V, O, L)

        return O.reshape(*batch_dims, Nq, d)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        raise NotImplementedError
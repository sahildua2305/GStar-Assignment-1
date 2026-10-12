# problem_8.py
import torch
import triton
import triton.language as tl
import math
from typing import Optional

import problem_5 as p5


class FlashAttention2Function(torch.autograd.Function):
    """
    Triton implementation of FlashAttention-2, supports causal attention and GQA.
    """
    @staticmethod
    def forward(ctx, q, k, v, is_causal=True, softmax_scale: Optional[float] = None):
        batch, n_heads, seq_len, head_dim = q.shape
        n_kv_heads = k.shape[1]

        assert is_causal, "This kernel only supports causal attention"
        assert n_heads % n_kv_heads == 0, "num_attention_heads must be divisible by num_kv_heads"

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)

        o = torch.empty_like(q)
        M = torch.empty((batch, n_heads, seq_len), device=q.device, dtype=torch.float32)

        BLOCK_M, BLOCK_N = 128, 64
        grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)
        
        _flash_attention_forward_gqa_kernel[grid](
            q, k, v, o, M,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            softmax_scale,
            seq_len,
            n_heads,
            n_kv_heads,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        
        ctx.save_for_backward(q, k, v, o, M)
        ctx.softmax_scale = softmax_scale
        ctx.num_heads = n_heads
        ctx.num_kv_heads = n_kv_heads
        ctx.is_causal = is_causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        batch, n_heads, seq_len, head_dim = q.shape
        n_kv_heads = ctx.num_kv_heads

        dq = torch.empty_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        num_warps, num_stages = 4, 3

        BLOCK_M, BLOCK_N = 128, 32
        preprocess_grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)
        D = torch.empty_like(M)

        # Compute delta = sum(dO * O) and store it in D for backward pass
        _attn_bwd_preprocess[preprocess_grid](
            O=o,
            dO=do,
            D=D,
            SEQ_LEN=seq_len,
            HEAD_DIM=head_dim,
            BLOCK_SIZE_Q=BLOCK_M,
        )

        loop1_grid = (triton.cdiv(seq_len, BLOCK_M), 1, batch * n_heads)

        # Causal attention stage
        stage = 3 if ctx.is_causal else 1

        _attn_bwd_dk_dv[loop1_grid](
            Q=q,
            K=k,
            V=v,
            softmax_scale=ctx.softmax_scale,
            dO=do,
            dQ=dq,
            dK=dk,
            dV=dv,
            D=D,
            M=M,
            stride_batch=q.stride(0),
            stride_head=q.stride(1),
            stride_seq=q.stride(2),
            stride_dim=q.stride(3),
            NUM_HEADS=n_heads,
            SEQ_LEN=seq_len,
            HEAD_DIM=head_dim,
            BLOCK_SIZE_Q=BLOCK_N,
            BLOCK_SIZE_K=BLOCK_M,
            STAGE=stage,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        # Fix Q and iterate through all the KV block
        _attn_bwd_dq[loop1_grid](
            Q=q,
            K=k,
            V=v,
            softmax_scale=ctx.softmax_scale,
            dO=do,
            dQ=dq,
            dK=dk,
            dV=dv,
            M=M,
            D=D,
            stride_batch=q.stride(0),
            stride_head=q.stride(1),
            stride_seq=q.stride(2),
            stride_dim=q.stride(3),
            NUM_HEADS=n_heads,
            SEQ_LEN=seq_len,
            BLOCK_Q=BLOCK_M,
            BLOCK_KV=BLOCK_N,
            HEAD_DIM=head_dim,
            STAGE=stage,
            num_warps=num_warps,
            num_stages=num_stages,
        )


        # [OPTIONAL BONUS] STUDENT IMPLEMENTATION REQUIRED
        # Implement the Triton backward kernel for GQA from scratch.
        # You should:
        #   1. Precompute delta = sum(dO * O)
        #   2. Recompute attention probabilities P = softmax(QK^T)
        #   3. Use delta + dO to accumulate gradients for dq, dk, dv
        #   4. Respect GQA mapping and causal mask
        
        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None

@triton.jit
def _flash_attention_forward_gqa_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr, M_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    # Kernel parameters
    softmax_scale,
    SEQ_LEN,
    N_Q_HEADS,
    N_KV_HEADS,
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel template for the forward pass of causal FlashAttention with GQA.
    """
    # 1. Identify the block of queries and the batch/head to be processed.
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # --- STUDENT IMPLEMENTATION REQUIRED HERE (Part 1) ---
    # Your goal is to map the current query head (q_head_idx) to its corresponding shared key/value head (kv_head_idx).
    # 1. Calculate how many query heads are in each group.
    # 2. Use integer division to find the correct kv_head_idx.
    
    num_q_heads_per_group = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // num_q_heads_per_group # Placeholder: Replace with your calculation
    # --- END OF STUDENT IMPLEMENTATION ---


    # 2. Initialize accumulators in SRAM.
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 3. Load the block of queries (Q_i).
    q_offsets = (q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M))
    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)
    
    qk_scale = softmax_scale * 1.44269504
    
    # --- Phase 1: Off-Diagonal Blocks ---
    for start_n in range(0, q_block_idx * BLOCK_M, BLOCK_N):
        # --- STUDENT IMPLEMENTATION REQUIRED HERE (Part 2) ---
        # 1. Modify the pointer arithmetic for K and V to use your `kv_head_idx`.
        # 2. Reuse your working implementation for the online softmax update
        #    from your solution to Problem 4.
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)

        # 2. Compute the attention scores (S_ij).
        s_ij = tl.dot(q_block, k_block)
        s_ij *= qk_scale

        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        # 3. Update the online softmax statistics (m_i, l_i) and the accumulator (acc).
        m_ij = tl.max(s_ij, axis=-1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp2(m_i - m_new)
        l_i = alpha * l_i
        acc = alpha[:, None] * acc.to(tl.float32)

        p_ij = tl.exp2(s_ij - m_new[:, None])

        l_i += tl.sum(p_ij, axis=-1)
        acc += tl.dot(p_ij, v_block.to(tl.float32))

        m_i = m_new
        # --- END OF STUDENT IMPLEMENTATION ---

    # --- Phase 2: Diagonal Blocks ---
    diag_start = q_block_idx * BLOCK_M
    for start_n in range(diag_start, (q_block_idx + 1) * BLOCK_M, BLOCK_N):
        # --- STUDENT IMPLEMENTATION REQUIRED HERE (Part 3) ---
        # 1. Modify the pointer arithmetic for K and V to use your `kv_head_idx`.
        # 2. Reuse your working implementation for the masked online softmax
        #    update from your solution to Problem 4.
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)

        s_ij = tl.dot(q_block, k_block)
        s_ij *= qk_scale

        causal_mask = q_offsets[:, None] >= k_offsets[None, :]
        s_ij = tl.where(causal_mask, s_ij, -1.0e6)

        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        m_ij = tl.max(s_ij, axis=-1)
        m_new = tl.maximum(m_ij, m_i)

        alpha = tl.exp2(m_i - m_new)
        l_i *= alpha
        acc = alpha[:, None] * acc.to(tl.float32)

        p_ij = tl.exp2(s_ij - m_new[:, None])

        acc += tl.dot(p_ij, v_block.to(tl.float32))
        l_i += tl.sum(p_ij, axis=-1)

        m_i = m_new
        # --- END OF STUDENT IMPLEMENTATION ---

    # 4. Normalize and write the final output block.
    l_i_safe = l_i[:, None] + 1e-6
    # acc = acc / l_i_safe
    acc = acc / l_i[:, None]

    M_i = (m_i + tl.log2(l_i))
    m_ptrs = M_ptr + batch_head_idx * SEQ_LEN + q_offsets
    tl.store(m_ptrs, M_i, mask=q_offsets < SEQ_LEN)
    
    o_ptrs = O_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
             
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)

@triton.jit
def _attn_bwd_preprocess(
    O,
    dO,
    D, # [batch_size, num_heads, seq_len]
    SEQ_LEN,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
):
    block_index_q = tl.program_id(0)
    index_batch_head = tl.program_id(1)
    seq_start = block_index_q * BLOCK_SIZE_Q
    seq_end = min(seq_start + BLOCK_SIZE_Q, SEQ_LEN)

    offsets_q = seq_start + tl.arange(0, BLOCK_SIZE_Q)
    offsets_dim = tl.arange(0, HEAD_DIM)

    O_block = tl.load( # O [batch_size, num_heads, seq_len, head_dim]
        O
        + index_batch_head * HEAD_DIM * SEQ_LEN
        + offsets_q[:, None] * HEAD_DIM
        + offsets_dim[None, :],
        mask=offsets_q[:, None] < SEQ_LEN,
        other=0.0,
    ).to(tl.float32)

    dO_block = tl.load( # O [batch_size, num_heads, seq_len, head_dim]
        dO
        + index_batch_head * HEAD_DIM * SEQ_LEN
        + offsets_q[:, None] * HEAD_DIM
        + offsets_dim[None, :],
        mask=offsets_q[:, None] < SEQ_LEN,
        other=0.0,
    ).to(tl.float32)

    Di_block = tl.sum(dO_block * O_block, axis=1) # [BLOCK_SIZE_Q]
    Di_block_ptrs = D + index_batch_head * SEQ_LEN + offsets_q
    tl.store(Di_block_ptrs, Di_block)

@triton.jit
def _attn_bwd_dk_dv(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    D, # [batch_size, num_heads, seq_len]
    M, # [batch_size, num_heads, seq_len]
    stride_batch: tl.constexpr,
    stride_head: tl.constexpr,
    stride_seq: tl.constexpr,
    stride_dim: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    STAGE: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(tl.int64)

    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    Q = Q + offset_batch_head
    K = K + offset_batch_head
    V = V + offset_batch_head
    dQ = dQ + offset_batch_head
    dK = dK + offset_batch_head
    dV = dV + offset_batch_head
    dO = dO + offset_batch_head

    M = M + offset_batch_head_seq
    D = D + offset_batch_head_seq

    offsets_dim = tl.arange(0, HEAD_DIM)
    
    index_block = tl.program_id(0)
    seq_start_kv = index_block * BLOCK_SIZE_K

    offsets_kv = seq_start_kv + tl.arange(0, BLOCK_SIZE_K)

    dV_block = tl.zeros([BLOCK_SIZE_K, HEAD_DIM], dtype=tl.float32)
    dK_block = tl.zeros([BLOCK_SIZE_K, HEAD_DIM], dtype=tl.float32)

    K_block = tl.load(
        K
        + offsets_kv[:, None] * stride_seq
        + offsets_dim[None, :] * stride_dim,
        mask=offsets_kv[:, None] < SEQ_LEN,
        other=0.0,
    ) # [BLLOCK_SIZE_K, HEAD_DIM]
    V_block = tl.load(
        V
        + offsets_kv[:, None] * stride_seq
        + offsets_dim[None, :] * stride_dim,
        mask=offsets_kv[:, None] < SEQ_LEN,
        other=0.0,
    ) # [BLLOCK_SIZE_K, HEAD_DIM]

    offsets_q = tl.arange(0, BLOCK_SIZE_Q)

    q_ptrs = (
        Q
        + offsets_q[:, None] * stride_seq
        + offsets_dim[None, :] * stride_dim
    )
    # qT_ptrs = (
    #     Q
    #     + offsets_q[None, :] * stride_seq
    #     + offsets_dim[:, None] * stride_dim
    # )
    qT_ptrs = tl.trans(q_ptrs)
    dO_ptrs = (
        dO
        + offsets_q[:, None] * stride_seq
        + offsets_dim[None, :] * stride_dim
    )

    curr_q = 0
    num_steps = SEQ_LEN // BLOCK_SIZE_Q

    for block_idx in range(num_steps):
        qT_block = tl.load(qT_ptrs)
        offsets_q = curr_q + tl.arange(0, BLOCK_SIZE_Q)
        m = tl.load(M + offsets_q)

        # QK_T_block = softmax_scale * tl.dot(K_block, qT_block)
        # P_T_block = tl.math.exp(QK_T_block - m[None, :])

        qk_scale = softmax_scale * 1.44269504
        QK_T_block = qk_scale * tl.dot(K_block, qT_block)
        P_T_block = tl.exp2(QK_T_block - m[None, :])

        if STAGE == 3:
            # Causal attention: mask out future positions
            causal_mask = offsets_q[None, :] >= offsets_kv[:, None]
            P_T_block = tl.where(causal_mask, P_T_block, 0.0)
    
        dO_block = tl.load(dO_ptrs)
        dV_block += tl.dot(P_T_block.to(dO_block.dtype), dO_block)

        Di = tl.load(D + offsets_q)

        dpT_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)
        dS_T_block = P_T_block * (dpT_block - Di[None, :])

        dK_block += tl.dot(dS_T_block.to(qT_block.dtype), tl.trans(qT_block))
        curr_q += BLOCK_SIZE_Q
        qT_ptrs += BLOCK_SIZE_Q * stride_seq
        dO_ptrs += BLOCK_SIZE_Q * stride_seq

    dK_block *= softmax_scale

    dK_block_ptrs = (
        dK
        + offsets_kv[:, None] * stride_seq
        + offsets_dim[None, :] * stride_dim
    )
    tl.store(dK_block_ptrs, dK_block)

    dV_block_ptrs = (
        dV
        + offsets_kv[:, None] * stride_seq
        + offsets_dim[None, :] * stride_dim
    )
    tl.store(dV_block_ptrs, dV_block)

@triton.jit
def _attn_bwd_dq(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(
        tl.int64
    )
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head

    M += offset_batch_head_seq
    D += offset_batch_head_seq

    # load scales
    offs_dim = tl.arange(0, HEAD_DIM)

    index_block_kv = tl.program_id(0)

    start_q = index_block_kv * BLOCK_Q
    offs_q = start_q + tl.arange(0, BLOCK_Q)

    Q_block = tl.load(Q + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim)
    dQ_block = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
    dO_block = tl.load(
        dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )

    M_block = tl.load(M + offs_q)
    M_block = M_block[:, None]

    offs_kv = tl.arange(0, BLOCK_KV)

    kT_ptrs = K + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    vT_ptrs = V + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim

    Di = tl.load(D + offs_q)

    curr_kv = 0
    num_steps = SEQ_LEN // BLOCK_KV
    for blk_idx in range(num_steps):
        K_T_block = tl.load(kT_ptrs)
        V_T_block = tl.load(vT_ptrs)
        # QK_block = softmax_scale * tl.dot(Q_block, K_T_block)
        # P_block = tl.math.exp(QK_block - M_block)
        qk_scale = softmax_scale * 1.44269504
        QK_block = qk_scale * tl.dot(Q_block, K_T_block)
        P_block = tl.exp2(QK_block - M_block)

        if STAGE == 3:
            offs_kv = curr_kv + tl.arange(0, BLOCK_KV)
            mask_block = offs_q[:, None] >= offs_kv[None, :]
            P_block = tl.where(mask_block, P_block, 0.0)

        dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)
        dS_block = P_block * (dP_block - Di[:, None])
        
        dQ_block += qk_scale * tl.dot(dS_block.to(K_T_block.dtype), tl.trans(K_T_block))

        curr_kv += BLOCK_KV
        kT_ptrs += BLOCK_KV * stride_seq
        vT_ptrs += BLOCK_KV * stride_seq

    dQ_block *= softmax_scale

    dQ_block_ptrs = dQ + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dQ_block_ptrs, dQ_block)


def flash_attention_gqa(q, k, v, is_causal=True, softmax_scale=None):
    return FlashAttention2Function.apply(q, k, v, is_causal, softmax_scale)
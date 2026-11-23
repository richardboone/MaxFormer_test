import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


def _get_arg(args, name, default):
    if args is None:
        return default
    val = getattr(args, name, None)
    return default if val is None else val


def _encode_surrogate(args, gama):
    mode = _get_arg(args, 'dS_du', 'Gamma')
    if mode == 'sigmoid':
        surrogate_mode = 0
    elif mode in ('Gamma', 'gamma'):
        surrogate_mode = 1
    elif mode == 'fast_sigmoid':
        surrogate_mode = 2
    elif mode == 'ste':
        surrogate_mode = 3
    else:
        raise ValueError(f"Unsupported dS_du {mode!r} in Triton kernel")
    surrogate_alpha = float(_get_arg(args, 'surrogate_alpha', 4.0))
    surrogate_gamma = float(_get_arg(args, 'surrogate_gamma', gama))
    return surrogate_mode, surrogate_alpha, surrogate_gamma


# ---------------------------------------------------------------------------
# Triton Fused Forward Kernel
# ---------------------------------------------------------------------------
@triton.jit
def _lif_forward_kernel(
    x_ptr,                  # [T, N] input tensor
    spikes_ptr,             # [T, N] output spikes
    mem_before_ptr,         # [T, N] saved membrane potential before spikes
    T: tl.constexpr,        # number of timesteps
    N,                      # total number of spatial/batch elements
    thresh,                 # threshold V_th
    decay,                  # decay constant (1 - 1/tau)
    input_scale,            # input scale (1/tau or decoupled)
    reset_mode: tl.constexpr, # 0 = hard reset, 1 = soft reset
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    mem = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for t in range(T):
        curr_offset = t * N + offsets
        x_val = tl.load(x_ptr + curr_offset, mask=mask, other=0.0).to(tl.float32)

        # Membrane potential before spike: mem * decay + x * input_scale
        mem_before = mem * decay + x_val * input_scale
        tl.store(mem_before_ptr + curr_offset, mem_before, mask=mask)

        # Spike generation: Heaviside(mem_before - thresh)
        spike = tl.where(mem_before > thresh, 1.0, 0.0)
        tl.store(spikes_ptr + curr_offset, spike, mask=mask)

        # Reset dynamics
        if reset_mode == 1:  # soft reset: mem = mem_before - thresh * spike
            mem = mem_before - thresh * spike
        else:  # hard reset: mem = mem_before * (1 - spike)
            mem = mem_before * (1.0 - spike)


# ---------------------------------------------------------------------------
# Triton Fused Backward Kernel (LIF Baseline & Legacy Conservative C-Grad)
# ---------------------------------------------------------------------------
@triton.jit
def _lif_conservative_backward_kernel(
    grad_output_ptr,        # [T, N] dL/dS
    mem_before_ptr,         # [T, N] u1
    spikes_ptr,             # [T, N] S
    grad_x_ptr,             # [T, N] dL/dx output
    T: tl.constexpr,        # timesteps
    N,                      # total elements
    thresh,
    decay,
    input_scale,
    gama,
    surrogate_mode: tl.constexpr, # 0 = sigmoid, 1 = gamma, 2 = fast_sigmoid, 3 = ste
    surrogate_alpha,
    reset_mode: tl.constexpr,     # 0 = hard, 1 = soft
    detach_reset: tl.constexpr,   # 0 = False, 1 = True
    cgrad_mode: tl.constexpr,     # 0 = standard LIF / TET, 1 = conservative_cgrad
    # conservative_cgrad parameters
    alpha_m,
    alpha_d,
    epsilon,
    intervention_threshold,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    grad_memb_last = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Iterate backwards in time: T-1 down to 0
    for step in range(T):
        t = T - 1 - step
        curr_offset = t * N + offsets

        dL_dS = tl.load(grad_output_ptr + curr_offset, mask=mask, other=0.0).to(tl.float32)
        u1 = tl.load(mem_before_ptr + curr_offset, mask=mask, other=0.0).to(tl.float32)
        spike = tl.load(spikes_ptr + curr_offset, mask=mask, other=0.0).to(tl.float32)

        # Temporal error propagation: dL/dU2 = grad_memb_last * decay
        dL_dU2 = grad_memb_last * decay

        # Surrogate gradient dS/dU1
        if surrogate_mode == 0:  # sigmoid
            sgax = (u1 - thresh) * surrogate_alpha
            sig = tl.sigmoid(sgax)
            dS_dU1 = (1.0 - sig) * sig * surrogate_alpha
        elif surrogate_mode == 2:  # fast_sigmoid
            denom = surrogate_alpha * tl.abs(u1 - thresh) + 1.0
            dS_dU1 = 1.0 / (denom * denom)
        elif surrogate_mode == 3:  # ste
            dS_dU1 = tl.where(tl.abs(u1 - thresh) < 0.8, 1.0, 0.05)
        else:  # gamma (triangular)
            dist = tl.abs(u1 - thresh)
            dS_dU1 = (1.0 / (gama * gama)) * tl.maximum(0.0, gama - dist)

        # Standard temporal Jacobian dU2/dU1
        if reset_mode == 1:  # soft reset
            if detach_reset == 1:
                dU2_dU1_standard = 1.0
            else:
                dU2_dU1_standard = 1.0 - thresh * dS_dU1
        else:  # hard reset
            if detach_reset == 1:
                dU2_dU1_standard = 1.0 - spike
            else:
                dU2_dU1_standard = (1.0 - spike) - (u1 * dS_dU1)

        # Base gradient (standard BPTT)
        base_function = dL_dS * dS_dU1 + dL_dU2 * dU2_dU1_standard

        if cgrad_mode == 1:  # legacy conservative_cgrad
            if reset_mode == 1:  # soft reset
                term_supra = (2.0 * thresh - u1) * dL_dU2 - dL_dS
            else:  # hard reset
                term_supra = (thresh * dL_dU2) - dL_dS
            term_sub = dL_dS - (u1 * dL_dU2)

            m = tl.where(u1 < thresh, term_sub, term_supra)
            m_grad = tl.where(u1 < thresh, m, -m)

            delta = u1 - thresh
            misalignment = -m_grad * base_function

            g_m = tl.sigmoid(alpha_m * (tl.abs(m) - 0.1))
            g_d = tl.sigmoid(alpha_d * (epsilon - tl.abs(delta)))
            g_misalign = tl.sigmoid(alpha_m * misalignment)

            intervention_signal = g_m * g_d * g_misalign
            do_intervene = tl.where(intervention_signal > intervention_threshold, 1.0, 0.0)

            sign_m_grad = tl.where(m_grad > 0.0, 1.0, tl.where(m_grad < 0.0, -1.0, 0.0))
            soft_correction = sign_m_grad * (tl.abs(base_function) * 0.5)

            blend_factor = do_intervene * tl.sigmoid(5.0 * (intervention_signal - intervention_threshold))
            dL_dU1 = (1.0 - 0.5 * blend_factor) * base_function + 0.5 * blend_factor * soft_correction
        else:
            dL_dU1 = base_function

        # Update running membrane gradient accumulator
        grad_memb_last = dL_dU1

        # Store gradient with respect to x[t]
        grad_x_val = grad_memb_last * input_scale
        tl.store(grad_x_ptr + curr_offset, grad_x_val, mask=mask)


# ---------------------------------------------------------------------------
# Triton Fused 1-Pass Backward Kernel (Improved C-Grad)
# ---------------------------------------------------------------------------
@triton.jit
def _lif_improved_cgrad_backward_kernel(
    grad_output_ptr,        # [T, N] dL/dS
    mem_before_ptr,         # [T, N] u1
    spikes_ptr,             # [T, N] S
    grad_x_ptr,             # [T, N] dL/dx output
    T: tl.constexpr,        # timesteps
    N,                      # total elements
    thresh,
    decay,
    input_scale,
    surrogate_mode: tl.constexpr, # 0 = sigmoid, 1 = gamma, 2 = fast_sigmoid, 3 = ste
    surrogate_alpha,
    surrogate_gamma,
    reset_mode: tl.constexpr,     # 0 = hard, 1 = soft
    detach_reset: tl.constexpr,   # 0 = False, 1 = True
    is_zero_eta: tl.constexpr,    # 1 = pure BPTT (eta == 0.0)
    norm_mode: tl.constexpr,      # 0 = none, 1 = local
    harmful_only: tl.constexpr,   # 0 = two-sided |m_eff|, 1 = harmful m_eff > 0
    use_sharp_gate: tl.constexpr, # 1 = thresholded sigmoid, 0 = smooth [0, 1] clamp
    epsilon,
    alpha_m,
    beta_d,
    alpha_mis,
    margin_m,
    h_thresh,
    gamma_sharp,
    corr_scale,
    eta,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    grad_memb_last = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for step in range(T):
        t = T - 1 - step
        curr_offset = t * N + offsets

        dL_dS = tl.load(grad_output_ptr + curr_offset, mask=mask, other=0.0).to(tl.float32)
        u1 = tl.load(mem_before_ptr + curr_offset, mask=mask, other=0.0).to(tl.float32)
        spike = tl.load(spikes_ptr + curr_offset, mask=mask, other=0.0).to(tl.float32)

        dL_dU2 = grad_memb_last * decay

        if surrogate_mode == 0:  # sigmoid
            sgax = (u1 - thresh) * surrogate_alpha
            sig = tl.sigmoid(sgax)
            dS_dU1 = (1.0 - sig) * sig * surrogate_alpha
        elif surrogate_mode == 2:  # fast_sigmoid
            denom = surrogate_alpha * tl.abs(u1 - thresh) + 1.0
            dS_dU1 = 1.0 / (denom * denom)
        elif surrogate_mode == 3:  # ste
            dS_dU1 = tl.where(tl.abs(u1 - thresh) < 0.8, 1.0, 0.05)
        else:  # gamma (triangular)
            dist = tl.abs(u1 - thresh)
            dS_dU1 = (1.0 / (surrogate_gamma * surrogate_gamma)) * tl.maximum(0.0, surrogate_gamma - dist)

        if reset_mode == 1:  # soft reset
            if detach_reset == 1:
                dU2_dU1_standard = 1.0
            else:
                dU2_dU1_standard = 1.0 - thresh * dS_dU1
        else:  # hard reset
            if detach_reset == 1:
                dU2_dU1_standard = 1.0 - spike
            else:
                dU2_dU1_standard = (1.0 - spike) - (u1 * dS_dU1)

        base_function = dL_dS * dS_dU1 + dL_dU2 * dU2_dU1_standard

        if is_zero_eta == 1:
            dL_dU1 = base_function
        else:
            if reset_mode == 1:
                term_supra = (2.0 * thresh - u1) * dL_dU2 - dL_dS
            else:
                term_supra = thresh * dL_dU2 - dL_dS
            term_sub = dL_dS - u1 * dL_dU2

            m = tl.where(u1 < thresh, term_sub, term_supra)
            m_grad = tl.where(u1 < thresh, m, -m)
            delta = u1 - thresh

            if norm_mode == 1:  # local
                err_scale = tl.abs(dL_dS) + tl.abs(thresh * dL_dU2) + 1e-12
                m_eff = m / err_scale
                misalignment = (-m_grad / err_scale) * (base_function / err_scale)
            else:  # none
                m_eff = m
                misalignment = -m_grad * base_function

            if harmful_only == 1:
                m_score = m_eff
            else:
                m_score = tl.abs(m_eff)

            g_m = tl.sigmoid(alpha_m * (m_score - margin_m))
            g_d = tl.sigmoid(beta_d * (epsilon - tl.abs(delta)))
            g_misalign = tl.sigmoid(alpha_mis * misalignment)

            intervention_signal = g_m * g_d * g_misalign
            if use_sharp_gate == 1:
                do_intervene = tl.where(intervention_signal > h_thresh, 1.0, 0.0)
                blend_factor = do_intervene * tl.sigmoid(gamma_sharp * (intervention_signal - h_thresh))
            else:
                blend_factor = tl.minimum(1.0, tl.maximum(0.0, intervention_signal))

            sign_m_grad = tl.where(m_grad > 0.0, 1.0, tl.where(m_grad < 0.0, -1.0, 0.0))
            soft_correction = sign_m_grad * (tl.abs(base_function) * corr_scale)
            w = eta * blend_factor
            dL_dU1 = (1.0 - w) * base_function + w * soft_correction

        grad_memb_last = dL_dU1
        grad_x_val = grad_memb_last * input_scale
        tl.store(grad_x_ptr + curr_offset, grad_x_val, mask=mask)


# ---------------------------------------------------------------------------
# Triton Invariant C-Grad: Fast GPU Reduction & Fused Step Kernel
# ---------------------------------------------------------------------------
@triton.jit
def _invariant_cgrad_reduce_kernel(
    dL_dS_ptr,              # [N] dL/dS at time t
    u1_ptr,                 # [N] u1 at time t
    spike_ptr,              # [N] S at time t
    grad_memb_last_ptr,     # [N] grad_memb_last from future time t+1
    sum_m2_out_ptr,         # [num_blocks] partial sums of m^2
    sum_pred2_out_ptr,      # [num_blocks] partial sums of dL_pred^2
    N,
    thresh,
    decay,
    gama,
    surrogate_mode: tl.constexpr,
    surrogate_alpha,
    reset_mode: tl.constexpr,
    detach_reset: tl.constexpr,
    consistency_sign: tl.constexpr, # 0 = product, 1 = ref, 2 = sign
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    dL_dS = tl.load(dL_dS_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    u1 = tl.load(u1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    spike = tl.load(spike_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grad_memb_last = tl.load(grad_memb_last_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    dL_dU2 = grad_memb_last * decay

    if surrogate_mode == 0:
        sgax = (u1 - thresh) * surrogate_alpha
        sig = tl.sigmoid(sgax)
        dS_dU1 = (1.0 - sig) * sig * surrogate_alpha
    elif surrogate_mode == 2:
        denom = surrogate_alpha * tl.abs(u1 - thresh) + 1.0
        dS_dU1 = 1.0 / (denom * denom)
    elif surrogate_mode == 3:
        dS_dU1 = tl.where(tl.abs(u1 - thresh) < 0.8, 1.0, 0.05)
    else:
        dist = tl.abs(u1 - thresh)
        dS_dU1 = (1.0 / (gama * gama)) * tl.maximum(0.0, gama - dist)

    if reset_mode == 1:
        if detach_reset == 1:
            dU2_dU1_standard = 1.0
        else:
            dU2_dU1_standard = 1.0 - thresh * dS_dU1
    else:
        if detach_reset == 1:
            dU2_dU1_standard = 1.0 - spike
        else:
            dU2_dU1_standard = (1.0 - spike) - (u1 * dS_dU1)

    base_function = dL_dS * dS_dU1 + dL_dU2 * dU2_dU1_standard

    if reset_mode == 1:
        term_supra = (2.0 * thresh - u1) * dL_dU2 - dL_dS
    else:
        term_supra = (thresh * dL_dU2) - dL_dS
    term_sub = dL_dS - (u1 * dL_dU2)

    m = tl.where(u1 < thresh, term_sub, term_supra)

    rho = thresh - u1
    if consistency_sign == 0:  # 'product'
        dL_pred = rho * base_function
    else:
        dL_pred = tl.where(u1 < thresh, base_function, -base_function) * thresh

    m2 = tl.where(mask, m * m, 0.0)
    pred2 = tl.where(mask, dL_pred * dL_pred, 0.0)

    sum_m2 = tl.sum(m2, axis=0)
    sum_pred2 = tl.sum(pred2, axis=0)

    tl.store(sum_m2_out_ptr + pid, sum_m2)
    tl.store(sum_pred2_out_ptr + pid, sum_pred2)


@triton.jit
def _invariant_compute_norm_kernel(
    partial_m2_ptr,
    partial_pred2_ptr,
    norm_out_ptr,
    num_blocks,
    N,
    BLOCK_SIZE: tl.constexpr
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_blocks

    m2_vals = tl.load(partial_m2_ptr + offsets, mask=mask, other=0.0)
    pred2_vals = tl.load(partial_pred2_ptr + offsets, mask=mask, other=0.0)

    sum_m2 = tl.sum(m2_vals, axis=0)
    sum_pred2 = tl.sum(pred2_vals, axis=0)

    mean_val = (sum_m2 + sum_pred2) / N
    norm_val = tl.sqrt(tl.maximum(1e-30, mean_val))

    tl.store(norm_out_ptr, norm_val)


@triton.jit
def _invariant_cgrad_step_kernel(
    dL_dS_ptr,              # [N] dL/dS at time t
    u1_ptr,                 # [N] u1 at time t
    spike_ptr,              # [N] S at time t
    grad_memb_last_ptr,     # [N] grad_memb_last from future time t+1 (IN/OUT)
    grad_x_ptr,             # [N] dL/dx output at time t
    norm_ptr,               # pointer to scalar RMS norm on GPU
    N,
    thresh,
    decay,
    input_scale,
    gama,
    surrogate_mode: tl.constexpr,
    surrogate_alpha,
    reset_mode: tl.constexpr,
    detach_reset: tl.constexpr,
    consistency_sign: tl.constexpr, # 0 = product, 1 = ref, 2 = sign
    # Invariant C-Grad hyperparameters
    alpha_m,
    alpha_d,
    alpha_cons,
    epsilon,
    kappa,
    intervention_threshold,
    eta,
    gamma,
    two_sided: tl.constexpr,
    correction_mode: tl.constexpr,  # 0 = blend, 1 = mask, 2 = flip
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    norm_val = tl.load(norm_ptr)

    dL_dS = tl.load(dL_dS_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    u1 = tl.load(u1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    spike = tl.load(spike_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grad_memb_last = tl.load(grad_memb_last_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    dL_dU2 = grad_memb_last * decay

    if surrogate_mode == 0:
        sgax = (u1 - thresh) * surrogate_alpha
        sig = tl.sigmoid(sgax)
        dS_dU1 = (1.0 - sig) * sig * surrogate_alpha
    elif surrogate_mode == 2:
        denom = surrogate_alpha * tl.abs(u1 - thresh) + 1.0
        dS_dU1 = 1.0 / (denom * denom)
    elif surrogate_mode == 3:
        dS_dU1 = tl.where(tl.abs(u1 - thresh) < 0.8, 1.0, 0.05)
    else:
        dist = tl.abs(u1 - thresh)
        dS_dU1 = (1.0 / (gama * gama)) * tl.maximum(0.0, gama - dist)

    if reset_mode == 1:
        if detach_reset == 1:
            dU2_dU1_standard = 1.0
        else:
            dU2_dU1_standard = 1.0 - thresh * dS_dU1
    else:
        if detach_reset == 1:
            dU2_dU1_standard = 1.0 - spike
        else:
            dU2_dU1_standard = (1.0 - spike) - (u1 * dS_dU1)

    base_function = dL_dS * dS_dU1 + dL_dU2 * dU2_dU1_standard

    if reset_mode == 1:
        term_supra = (2.0 * thresh - u1) * dL_dU2 - dL_dS
    else:
        term_supra = (thresh * dL_dU2) - dL_dS
    term_sub = dL_dS - (u1 * dL_dU2)

    m = tl.where(u1 < thresh, term_sub, term_supra)
    m_grad = tl.where(u1 < thresh, m, -m)

    rho = thresh - u1
    delta = u1 - thresh

    if consistency_sign == 0:
        dL_pred = rho * base_function
    else:
        dL_pred = tl.where(u1 < thresh, base_function, -base_function) * thresh

    inv_norm = 1.0 / norm_val
    m_hat = m * inv_norm
    pred_hat = dL_pred * inv_norm

    # Gate 1: Harmfulness
    if two_sided:
        g_m = tl.sigmoid(alpha_m * (tl.abs(m_hat) - kappa))
    else:
        g_m = tl.sigmoid(alpha_m * (m_hat - kappa))

    # Gate 2: Nearness to threshold
    g_d = tl.sigmoid(alpha_d * (epsilon - tl.abs(delta)))

    # Gate 3: Consistency
    if consistency_sign == 2:  # sign mode
        sign_m_hat = tl.where(m_hat > 0.0, 1.0, tl.where(m_hat < 0.0, -1.0, 0.0))
        sign_pred_hat = tl.where(pred_hat > 0.0, 1.0, tl.where(pred_hat < 0.0, -1.0, 0.0))
        g_cons = tl.sigmoid(alpha_cons * (-sign_m_hat * sign_pred_hat))
    else:
        g_cons = tl.sigmoid(alpha_cons * (-m_hat * pred_hat))

    # Combined intervention signal (Geometric Mean)
    prod_gates = tl.maximum(1e-12, g_m * g_d * g_cons)
    intervention_signal = tl.exp(0.3333333333333333 * tl.log(prod_gates))

    do_intervene = tl.where(intervention_signal > intervention_threshold, 1.0, 0.0)

    sign_m_grad = tl.where(m_grad > 0.0, 1.0, tl.where(m_grad < 0.0, -1.0, 0.0))
    soft_correction = sign_m_grad * tl.abs(base_function)

    blend_factor = do_intervene * tl.sigmoid(gamma * (intervention_signal - intervention_threshold))
    w = eta * blend_factor

    if correction_mode == 1:  # mask
        dL_dU1 = (1.0 - do_intervene) * base_function
    elif correction_mode == 2:  # flip
        dL_dU1 = tl.where(do_intervene > 0.0, soft_correction, base_function)
    else:  # blend
        dL_dU1 = (1.0 - w) * base_function + w * soft_correction

    # Store updated gradient accumulator in-place for next step
    tl.store(grad_memb_last_ptr + offsets, dL_dU1, mask=mask)

    # Store input gradient
    grad_x_val = dL_dU1 * input_scale
    tl.store(grad_x_ptr + offsets, grad_x_val, mask=mask)


# ---------------------------------------------------------------------------
# Python Interface & Dispatch Helpers
# ---------------------------------------------------------------------------
def triton_lif_forward(x, thresh=1.0, decay=0.5, input_scale=0.5, reset_mode='hard'):
    """
    Fused Triton LIF Forward Pass.
    Input: x of shape [T, B, *spatial] (contiguous or will be made contiguous)
    Returns: (spikes, mem_before_spikes)
    """
    orig_shape = x.shape
    T = orig_shape[0]
    N = x[0].numel()
    x_2d = x.contiguous().view(T, N)

    spikes = torch.empty_like(x_2d)
    mem_before_spikes = torch.empty_like(x_2d)

    reset_flag = 1 if reset_mode == 'soft' else 0
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    _lif_forward_kernel[grid](
        x_2d,
        spikes,
        mem_before_spikes,
        T=T,
        N=N,
        thresh=float(thresh),
        decay=float(decay),
        input_scale=float(input_scale),
        reset_mode=reset_flag,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return spikes.view(orig_shape), mem_before_spikes.view(orig_shape)


def triton_lif_backward_conservative(
    grad_output,
    mem_before_spikes,
    spikes,
    thresh=1.0,
    decay=0.5,
    input_scale=0.5,
    gama=1.0,
    args=None,
    detach_reset=False,
    reset_mode='hard'
):
    """
    Fused 1-pass backward for standard LIF / TET / legacy conservative_cgrad.
    """
    orig_shape = grad_output.shape
    T = orig_shape[0]
    N = grad_output[0].numel()

    grad_out_2d = grad_output.contiguous().view(T, N)
    mem_2d = mem_before_spikes.contiguous().view(T, N)
    spikes_2d = spikes.contiguous().view(T, N)
    grad_x_2d = torch.empty_like(grad_out_2d)

    surrogate_mode, surrogate_alpha, surrogate_gamma = _encode_surrogate(args, gama)
    reset_flag = 1 if reset_mode == 'soft' else 0
    detach_flag = 1 if detach_reset else 0

    mode = _get_arg(args, 'du_du', 'LIF')
    cgrad_flag = 1 if mode == 'conservative_cgrad' else 0

    alpha_m = float(_get_arg(args, 'snnbp_alpha', 2.0))
    alpha_d = float(_get_arg(args, 'snnbp_beta', 2.0))
    epsilon = float(_get_arg(args, 'snnbp_epsilon', 0.3))
    intervention_threshold = float(_get_arg(args, 'snnbp_intervention', 0.8))

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    _lif_conservative_backward_kernel[grid](
        grad_out_2d,
        mem_2d,
        spikes_2d,
        grad_x_2d,
        T=T,
        N=N,
        thresh=float(thresh),
        decay=float(decay),
        input_scale=float(input_scale),
        gama=float(surrogate_gamma),
        surrogate_mode=surrogate_mode,
        surrogate_alpha=surrogate_alpha,
        reset_mode=reset_flag,
        detach_reset=detach_flag,
        cgrad_mode=cgrad_flag,
        alpha_m=alpha_m,
        alpha_d=alpha_d,
        epsilon=epsilon,
        intervention_threshold=intervention_threshold,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return grad_x_2d.view(orig_shape)


def triton_lif_backward_improved(
    grad_output,
    mem_before_spikes,
    spikes,
    thresh=1.0,
    decay=0.5,
    input_scale=0.5,
    gama=1.0,
    args=None,
    detach_reset=False,
    reset_mode='hard'
):
    """
    Fused 1-pass backward for improved_cgrad (supports 'local' and 'none' norms).
    """
    orig_shape = grad_output.shape
    T = orig_shape[0]
    N = grad_output[0].numel()

    grad_out_2d = grad_output.contiguous().view(T, N)
    mem_2d = mem_before_spikes.contiguous().view(T, N)
    spikes_2d = spikes.contiguous().view(T, N)
    grad_x_2d = torch.empty_like(grad_out_2d)

    surrogate_mode, surrogate_alpha, surrogate_gamma = _encode_surrogate(args, gama)
    reset_flag = 1 if reset_mode == 'soft' else 0
    detach_flag = 1 if detach_reset else 0

    eta = float(_get_arg(args, 'snnbp_eta', 1.0))
    is_zero_eta = 1 if eta == 0.0 else 0

    norm_str = _get_arg(args, 'snnbp_norm', 'local')
    if norm_str == 'local':
        norm_mode = 1
    elif norm_str == 'none':
        norm_mode = 0
    else:
        raise ValueError(f"Triton improved_cgrad supports snnbp_norm in ('local', 'none'), got {norm_str!r}")

    harmful_only = 1 if bool(_get_arg(args, 'snnbp_harmful_only', True)) else 0
    epsilon = float(_get_arg(args, 'snnbp_epsilon', 0.35))
    alpha_m = float(_get_arg(args, 'snnbp_alpha', 3.0))
    beta_d = float(_get_arg(args, 'snnbp_beta', 8.0))
    alpha_mis = float(_get_arg(args, 'snnbp_alpha_mis', alpha_m))
    margin_m = float(_get_arg(args, 'snnbp_margin_m', 0.0))
    h_thresh = float(_get_arg(args, 'snnbp_intervention', 0.0))
    gamma_sharp = float(_get_arg(args, 'snnbp_gamma_sharp', 0.0))
    corr_scale = float(_get_arg(args, 'snnbp_corr_scale', 0.5))
    use_sharp_gate = 1 if gamma_sharp > 0.0 else 0

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    _lif_improved_cgrad_backward_kernel[grid](
        grad_out_2d,
        mem_2d,
        spikes_2d,
        grad_x_2d,
        T=T,
        N=N,
        thresh=float(thresh),
        decay=float(decay),
        input_scale=float(input_scale),
        surrogate_mode=surrogate_mode,
        surrogate_alpha=surrogate_alpha,
        surrogate_gamma=surrogate_gamma,
        reset_mode=reset_flag,
        detach_reset=detach_flag,
        is_zero_eta=is_zero_eta,
        norm_mode=norm_mode,
        harmful_only=harmful_only,
        use_sharp_gate=use_sharp_gate,
        epsilon=epsilon,
        alpha_m=alpha_m,
        beta_d=beta_d,
        alpha_mis=alpha_mis,
        margin_m=margin_m,
        h_thresh=h_thresh,
        gamma_sharp=gamma_sharp,
        corr_scale=corr_scale,
        eta=eta,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return grad_x_2d.view(orig_shape)


def triton_lif_backward_invariant(
    grad_output,
    mem_before_spikes,
    spikes,
    thresh=1.0,
    decay=0.5,
    input_scale=0.5,
    gama=1.0,
    args=None,
    detach_reset=False,
    reset_mode='hard'
):
    """
    Two-pass device-only backward for scale-invariant C-Grad (invariant_cgrad).
    """
    orig_shape = grad_output.shape
    T = orig_shape[0]
    N = grad_output[0].numel()

    grad_out_2d = grad_output.contiguous().view(T, N)
    mem_2d = mem_before_spikes.contiguous().view(T, N)
    spikes_2d = spikes.contiguous().view(T, N)
    grad_x_2d = torch.empty_like(grad_out_2d)

    surrogate_mode, surrogate_alpha, surrogate_gamma = _encode_surrogate(args, gama)
    reset_flag = 1 if reset_mode == 'soft' else 0
    detach_flag = 1 if detach_reset else 0

    consistency_str = _get_arg(args, 'snnbp_consistency', 'product')
    if consistency_str == 'product':
        consistency_sign = 0
    elif consistency_str == 'ref':
        consistency_sign = 1
    else:  # 'sign'
        consistency_sign = 2

    alpha_m = float(_get_arg(args, 'snnbp_alpha', 2.0))
    alpha_d = float(_get_arg(args, 'snnbp_beta', 2.0))
    alpha_cons = float(_get_arg(args, 'snnbp_alpha_cons', 2.0))
    epsilon = float(_get_arg(args, 'snnbp_epsilon', 0.3))
    kappa = float(_get_arg(args, 'snnbp_kappa', 1.0))
    intervention_threshold = float(_get_arg(args, 'snnbp_h', 0.5))
    eta = float(_get_arg(args, 'snnbp_eta', 0.5))
    gamma = float(_get_arg(args, 'snnbp_gamma', 5.0))
    two_sided = bool(_get_arg(args, 'snnbp_two_sided', True))

    corr_str = _get_arg(args, 'snnbp_correction', 'blend')
    if corr_str == 'mask':
        correction_mode = 1
    elif corr_str == 'flip':
        correction_mode = 2
    else:
        correction_mode = 0

    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    grid = (num_blocks,)

    # Running membrane gradient accumulator and reduction buffers in FP32
    grad_memb_last = torch.zeros(N, device=grad_output.device, dtype=torch.float32)
    partial_m2 = torch.empty(num_blocks, device=grad_output.device, dtype=torch.float32)
    partial_pred2 = torch.empty(num_blocks, device=grad_output.device, dtype=torch.float32)
    norm_val_tensor = torch.empty(1, device=grad_output.device, dtype=torch.float32)

    RED_BLOCK_SIZE = triton.next_power_of_2(max(1, num_blocks))

    for step in range(T):
        t = T - 1 - step
        dL_dS_t = grad_out_2d[t]
        u1_t = mem_2d[t]
        spike_t = spikes_2d[t]
        grad_x_t = grad_x_2d[t]

        _invariant_cgrad_reduce_kernel[grid](
            dL_dS_t,
            u1_t,
            spike_t,
            grad_memb_last,
            partial_m2,
            partial_pred2,
            N=N,
            thresh=float(thresh),
            decay=float(decay),
            gama=float(surrogate_gamma),
            surrogate_mode=surrogate_mode,
            surrogate_alpha=surrogate_alpha,
            reset_mode=reset_flag,
            detach_reset=detach_flag,
            consistency_sign=consistency_sign,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        _invariant_compute_norm_kernel[(1,)](
            partial_m2,
            partial_pred2,
            norm_val_tensor,
            num_blocks=num_blocks,
            N=N,
            BLOCK_SIZE=RED_BLOCK_SIZE,
        )

        _invariant_cgrad_step_kernel[grid](
            dL_dS_t,
            u1_t,
            spike_t,
            grad_memb_last,
            grad_x_t,
            norm_ptr=norm_val_tensor,
            N=N,
            thresh=float(thresh),
            decay=float(decay),
            input_scale=float(input_scale),
            gama=float(surrogate_gamma),
            surrogate_mode=surrogate_mode,
            surrogate_alpha=surrogate_alpha,
            reset_mode=reset_flag,
            detach_reset=detach_flag,
            consistency_sign=consistency_sign,
            alpha_m=alpha_m,
            alpha_d=alpha_d,
            alpha_cons=alpha_cons,
            epsilon=epsilon,
            kappa=kappa,
            intervention_threshold=intervention_threshold,
            eta=eta,
            gamma=gamma,
            two_sided=two_sided,
            correction_mode=correction_mode,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return grad_x_2d.view(orig_shape)


# ---------------------------------------------------------------------------
# Autograd Function Wrapper
# ---------------------------------------------------------------------------
class TritonTimeParallel_LIFSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, thresh, decay, input_scale, gama, args, detach_reset, reset_mode):
        spikes, mem_before = triton_lif_forward(
            x, thresh=thresh, decay=decay, input_scale=input_scale, reset_mode=reset_mode
        )
        ctx.save_for_backward(mem_before, spikes, x, torch.tensor([gama]))
        ctx.thresh = thresh
        ctx.decay = decay
        ctx.input_scale = input_scale
        ctx.args = args
        ctx.detach_reset = detach_reset
        ctx.reset_mode = reset_mode
        return spikes

    @staticmethod
    def backward(ctx, grad_output):
        mem_before, spikes, x, gama = ctx.saved_tensors
        mode = _get_arg(ctx.args, 'du_du', 'LIF')

        if mode == 'invariant_cgrad':
            grad_x = triton_lif_backward_invariant(
                grad_output=grad_output,
                mem_before_spikes=mem_before,
                spikes=spikes,
                thresh=ctx.thresh,
                decay=ctx.decay,
                input_scale=ctx.input_scale,
                gama=gama.item(),
                args=ctx.args,
                detach_reset=ctx.detach_reset,
                reset_mode=ctx.reset_mode,
            )
        elif mode == 'improved_cgrad':
            grad_x = triton_lif_backward_improved(
                grad_output=grad_output,
                mem_before_spikes=mem_before,
                spikes=spikes,
                thresh=ctx.thresh,
                decay=ctx.decay,
                input_scale=ctx.input_scale,
                gama=gama.item(),
                args=ctx.args,
                detach_reset=ctx.detach_reset,
                reset_mode=ctx.reset_mode,
            )
        else:
            grad_x = triton_lif_backward_conservative(
                grad_output=grad_output,
                mem_before_spikes=mem_before,
                spikes=spikes,
                thresh=ctx.thresh,
                decay=ctx.decay,
                input_scale=ctx.input_scale,
                gama=gama.item(),
                args=ctx.args,
                detach_reset=ctx.detach_reset,
                reset_mode=ctx.reset_mode,
            )

        return grad_x, None, None, None, None, None, None, None

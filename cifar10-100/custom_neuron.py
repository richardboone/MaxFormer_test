import copy
import math
import os
import random
import torch
import torch.nn as nn

try:
    from spikingjelly.clock_driven.neuron import MultiStepLIFNode as OriginalLIFNode
except ImportError:
    try:
        from spikingjelly.activation_based.neuron import LIFNode as OriginalLIFNode
    except ImportError:
        OriginalLIFNode = None

try:
    from triton_neuron import TritonTimeParallel_LIFSpike
    TRITON_AVAILABLE = True
except ImportError:
    try:
        from .triton_neuron import TritonTimeParallel_LIFSpike
        TRITON_AVAILABLE = True
    except ImportError:
        TritonTimeParallel_LIFSpike = None
        TRITON_AVAILABLE = False

# --- Global Args Handler ---
GLOBAL_ARGS = None
METRICS_COLLECTOR = None
DISABLE_CUPY = os.environ.get('DISABLE_CUPY', '0') == '1'
_INVARIANT_GATE_CHECKED = False


def set_global_args(args):
    global GLOBAL_ARGS
    GLOBAL_ARGS = args


def set_metrics_collector(collector):
    global METRICS_COLLECTOR
    METRICS_COLLECTOR = collector


def _get_arg(args, name, default):
    if args is None:
        return default
    val = getattr(args, name, None)
    return default if val is None else val


def _resolve_tau_and_scale(tau, input_scale, args):
    """Resolve (decay, eff_input_scale) from layer defaults and optional Namespace overrides."""
    eff_tau = float(tau)
    raw_tau = _get_arg(args, 'snnbp_tau', None)
    if raw_tau is not None:
        raw_tau = float(raw_tau)
        if 0.0 < raw_tau < 1.0:
            # Compatibility with repo convention where snnbp_tau < 1.0 is decay factor
            decay = raw_tau
            eff_tau = 1.0 / (1.0 - decay)
        else:
            eff_tau = raw_tau
            decay = 1.0 - (1.0 / eff_tau)
    else:
        decay = 1.0 - (1.0 / eff_tau)

    raw_scale = _get_arg(args, 'snnbp_input_scale', None)
    if raw_scale is not None:
        if raw_scale in ('1/tau', 'coupled'):
            eff_scale = 1.0 - decay
        else:
            eff_scale = float(raw_scale)
    elif input_scale in ('1/tau', 'coupled'):
        eff_scale = 1.0 - decay
    else:
        eff_scale = float(input_scale)

    return decay, eff_scale


# --- C-Grad intervention statistics -----------------------------------------
_CGRAD_STATS = None
_CGRAD_STATS_P = 0.05
_CGRAD_TOTAL = 0.0

_STAT_KEYS = (
    'fired',
    'effective_fired',
    'm_neg',
    'effective_m_neg',
    'signal_sum',
    'weight_sum',
    'near_thresh',
)


def enable_cgrad_stats(device, sample_prob=0.05):
    """Start collecting C-Grad intervention statistics on-device with zero per-step CPU sync."""
    global _CGRAD_STATS, _CGRAD_STATS_P, _CGRAD_TOTAL
    _CGRAD_STATS = {
        k: torch.zeros((), device=device, dtype=torch.float64) for k in _STAT_KEYS
    }
    _CGRAD_STATS_P = float(sample_prob)
    _CGRAD_TOTAL = 0.0


def disable_cgrad_stats():
    global _CGRAD_STATS, _CGRAD_TOTAL
    _CGRAD_STATS = None
    _CGRAD_TOTAL = 0.0


def pop_cgrad_stats():
    """Return accumulated statistics as plain floats and reset counters."""
    global _CGRAD_TOTAL
    if _CGRAD_STATS is None or _CGRAD_TOTAL == 0:
        return None
    total = _CGRAD_TOTAL
    fired = _CGRAD_STATS['fired'].item()
    eff_fired = _CGRAD_STATS['effective_fired'].item()
    out = {
        'cgrad/intervention_rate': fired / total,
        'cgrad/effective_rate': eff_fired / total,
        'cgrad/mean_signal': _CGRAD_STATS['signal_sum'].item() / total,
        'cgrad/mean_w': _CGRAD_STATS['weight_sum'].item() / total,
        'cgrad/near_threshold_rate': _CGRAD_STATS['near_thresh'].item() / total,
        'cgrad/frac_interventions_m_neg': (
            _CGRAD_STATS['m_neg'].item() / fired if fired > 0 else 0.0
        ),
        'cgrad/frac_effective_m_neg': (
            _CGRAD_STATS['effective_m_neg'].item() / eff_fired if eff_fired > 0 else 0.0
        ),
        'cgrad/neuron_timesteps_sampled': total,
    }
    for k in _STAT_KEYS:
        _CGRAD_STATS[k].zero_()
    _CGRAD_TOTAL = 0.0
    return out


def _accumulate_cgrad_stats(active, weight, signal, m, near_thresh, base_function):
    """Accumulate stats on-device without CPU sync or gradient side effects."""
    global _CGRAD_TOTAL
    s = _CGRAD_STATS
    active_f = active.to(torch.float32)
    nonzero_base = (base_function != 0).to(torch.float32)
    eff_active = active_f * nonzero_base
    m_neg_f = (m < 0).to(torch.float32)
    s['fired'] += active_f.sum(dtype=torch.float32)
    s['effective_fired'] += eff_active.sum(dtype=torch.float32)
    s['m_neg'] += (active_f * m_neg_f).sum(dtype=torch.float32)
    s['effective_m_neg'] += (eff_active * m_neg_f).sum(dtype=torch.float32)
    s['signal_sum'] += signal.to(torch.float32).sum(dtype=torch.float32)
    s['weight_sum'] += weight.to(torch.float32).sum(dtype=torch.float32)
    s['near_thresh'] += near_thresh.to(torch.float32).sum(dtype=torch.float32)
    _CGRAD_TOTAL += float(active.numel())


def _check_invariant_gate_reachable(alpha_d, epsilon, h):
    """Warn once if invariant_cgrad is configured so that it can never intervene."""
    global _INVARIANT_GATE_CHECKED
    if _INVARIANT_GATE_CHECKED:
        return
    _INVARIANT_GATE_CHECKED = True
    ceiling = (1.0 / (1.0 + math.exp(-alpha_d * epsilon))) ** (1.0 / 3.0)
    if h >= ceiling:
        import warnings
        warnings.warn(
            f"invariant_cgrad: snnbp_h={h:.3f} is at or above the attainable "
            f"ceiling of the intervention signal ({ceiling:.3f}, set by "
            f"sigmoid(alpha_d*epsilon)**(1/3) with alpha_d={alpha_d}, "
            f"epsilon={epsilon}). C-Grad will never intervene and training will "
            f"be identical to standard BPTT. Lower snnbp_h or raise "
            f"snnbp_beta/snnbp_epsilon.", RuntimeWarning)


def _surrogate_derivative(u, thresh, gama_val, args):
    mode = _get_arg(args, 'dS_du', 'Gamma')
    if mode == "sigmoid":
        alpha = float(_get_arg(args, 'surrogate_alpha', 4.0))
        sgax = (u - thresh) * alpha
        sig = torch.sigmoid(sgax)
        return (1.0 - sig) * sig * alpha
    if mode in ("Gamma", "gamma"):
        g = float(_get_arg(args, 'surrogate_gamma', gama_val))
        return (1.0 / (g * g)) * (g - (u - thresh).abs()).clamp(min=0.0)
    if mode == "fast_sigmoid":
        alpha = float(_get_arg(args, 'surrogate_alpha', 4.0))
        return 1.0 / (alpha * (u - thresh).abs() + 1.0) ** 2
    if mode == "ste":
        return torch.where(
            (u - thresh).abs() < 0.8,
            torch.ones_like(u),
            torch.full_like(u, 0.05),
        )
    # Default fallback (Gamma)
    g = float(_get_arg(args, 'surrogate_gamma', gama_val))
    return (1.0 / (g * g)) * (g - (u - thresh).abs()).clamp(min=0.0)


def _compute_standard_du2_du1(u1, spike, thresh, dS_dU1, detach_reset, reset_mode):
    if reset_mode == 'soft':
        if detach_reset:
            return torch.ones_like(u1)
        return 1.0 - thresh * dS_dU1
    if detach_reset:
        return 1.0 - spike
    return (1.0 - spike) - (u1 * dS_dU1)


def _compute_cgrad_margins(u1, thresh, dL_dS, dL_dU2, reset_mode):
    if reset_mode == 'soft':
        term_supra = (2.0 * thresh - u1) * dL_dU2 - dL_dS
    else:
        term_supra = thresh * dL_dU2 - dL_dS
    term_sub = dL_dS - u1 * dL_dU2
    m = torch.where(u1 < thresh, term_sub, term_supra)
    m_grad = torch.where(u1 < thresh, m, -m)
    return m, m_grad


# --- Custom Autograd Function ---
class TimeParallel_LIFSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, thresh, decay, input_scale, gama, args, detach_reset, reset_mode, collect_stats=None):
        # SpikingJelly Standard: [Time, Batch, *Spatial]
        T, batch_size, *spatial_dims = x.shape
        work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        mem = torch.zeros(batch_size, *spatial_dims, device=x.device, dtype=work_dtype)

        mem_before_spikes = []
        spikes = []

        for t in range(T):
            mem_before_spike = mem * decay + x[t].to(work_dtype) * input_scale
            mem_before_spikes.append(mem_before_spike)

            spike = (mem_before_spike > thresh).to(work_dtype)
            spikes.append(spike)

            if reset_mode == 'soft':
                mem = mem_before_spike - thresh * spike
            else:  # 'hard'
                mem = mem_before_spike * (1.0 - spike)

        mem_before_spikes = torch.stack(mem_before_spikes, dim=0)
        spikes_tensor = torch.stack(spikes, dim=0).to(x.dtype)

        ctx.save_for_backward(mem_before_spikes, spikes_tensor)
        ctx.thresh = float(thresh)
        ctx.decay = float(decay)
        ctx.input_scale = float(input_scale)
        ctx.gama = float(gama)
        ctx.args = args
        ctx.detach_reset = bool(detach_reset)
        ctx.reset_mode = reset_mode
        ctx.collect_stats = collect_stats

        return spikes_tensor

    @staticmethod
    def backward(ctx, grad_output):
        mem_before_spikes, spikes_tensor = ctx.saved_tensors
        thresh = ctx.thresh
        decay = ctx.decay
        input_scale = ctx.input_scale
        gama_val = ctx.gama
        args = ctx.args
        detach_reset = ctx.detach_reset
        reset_mode = ctx.reset_mode
        collect_stats = ctx.collect_stats

        T = mem_before_spikes.shape[0]
        work_dtype = torch.float64 if grad_output.dtype == torch.float64 else torch.float32
        grad_x = torch.zeros_like(grad_output)
        grad_memb_last = torch.zeros(
            mem_before_spikes.shape[1:], device=grad_output.device, dtype=work_dtype
        )

        mode = _get_arg(args, 'du_du', 'complex54')
        if collect_stats is True:
            do_collect = _CGRAD_STATS is not None
        elif collect_stats is False:
            do_collect = False
        else:
            do_collect = (_CGRAD_STATS is not None and random.random() < _CGRAD_STATS_P)

        for t in reversed(range(T)):
            u1 = mem_before_spikes[t].to(work_dtype)
            spike = spikes_tensor[t].to(work_dtype)
            dL_dS = grad_output[t].to(work_dtype)
            dL_dU2 = grad_memb_last * decay

            dS_dU1 = _surrogate_derivative(u1, thresh, gama_val, args)
            dU2_dU1_standard = _compute_standard_du2_du1(
                u1, spike, thresh, dS_dU1, detach_reset, reset_mode
            )
            base_function = dL_dS * dS_dU1 + dL_dU2 * dU2_dU1_standard

            if mode == "improved_cgrad":
                blend_eta = float(_get_arg(args, 'snnbp_eta', 1.0))
                epsilon = float(_get_arg(args, 'snnbp_epsilon', 0.35))
                if blend_eta == 0.0:
                    dL_dU1 = base_function
                    if do_collect:
                        delta = u1 - thresh
                        zeros = torch.zeros_like(base_function)
                        _accumulate_cgrad_stats(
                            zeros, zeros, zeros, zeros, delta.abs() <= epsilon, base_function
                        )
                else:
                    norm_mode = _get_arg(args, 'snnbp_norm', 'local')
                    harmful_only = bool(_get_arg(args, 'snnbp_harmful_only', True))
                    alpha_m = float(_get_arg(args, 'snnbp_alpha', 3.0))
                    beta_d = float(_get_arg(args, 'snnbp_beta', 8.0))
                    alpha_mis = float(_get_arg(args, 'snnbp_alpha_mis', alpha_m))
                    margin_m = float(_get_arg(args, 'snnbp_margin_m', 0.0))
                    h_thresh = float(_get_arg(args, 'snnbp_intervention', 0.0))
                    gamma_sharp = float(_get_arg(args, 'snnbp_gamma_sharp', 0.0))
                    corr_scale = float(_get_arg(args, 'snnbp_corr_scale', 0.5))

                    delta = u1 - thresh
                    m, m_grad = _compute_cgrad_margins(u1, thresh, dL_dS, dL_dU2, reset_mode)

                    if norm_mode == 'none':
                        m_eff = m
                        misalignment = -m_grad * base_function
                    elif norm_mode == 'local':
                        err_scale = dL_dS.abs() + (thresh * dL_dU2).abs() + 1e-12
                        m_eff = m / err_scale
                        misalignment = (-m_grad / err_scale) * (base_function / err_scale)
                    elif norm_mode == 'rms':
                        rms_m = torch.sqrt((m * m).mean() + 1e-20)
                        rms_g = torch.sqrt((base_function * base_function).mean() + 1e-20)
                        m_eff = m / rms_m
                        misalignment = (-m_grad / rms_m) * (base_function / rms_g)
                    else:
                        raise ValueError(
                            f"Unsupported snnbp_norm {norm_mode!r}; expected 'local', 'rms', or 'none'."
                        )

                    m_score = m_eff if harmful_only else m_eff.abs()
                    g_m = torch.sigmoid(alpha_m * (m_score - margin_m))
                    g_d = torch.sigmoid(beta_d * (epsilon - delta.abs()))
                    g_misalign = torch.sigmoid(alpha_mis * misalignment)

                    intervention_signal = g_m * g_d * g_misalign
                    if gamma_sharp > 0.0:
                        do_intervene = (intervention_signal > h_thresh).to(work_dtype)
                        blend_factor = do_intervene * torch.sigmoid(
                            gamma_sharp * (intervention_signal - h_thresh)
                        )
                    else:
                        blend_factor = torch.clamp(intervention_signal, 0.0, 1.0)

                    soft_correction = torch.sign(m_grad) * (base_function.abs() * corr_scale)
                    w = blend_eta * blend_factor
                    dL_dU1 = (1.0 - w) * base_function + w * soft_correction

                    if do_collect:
                        _accumulate_cgrad_stats(
                            w.detach() > 0.01,
                            w.detach(),
                            intervention_signal.detach(),
                            m.detach(),
                            delta.abs() <= epsilon,
                            base_function.detach(),
                        )

                    if METRICS_COLLECTOR is not None:
                        stats = {
                            't': t,
                            'mean_omega': float(intervention_signal.detach().mean().item()),
                            'active_frac': float((w.detach() > 0.01).to(work_dtype).mean().item()),
                            'mean_weight': float(w.detach().mean().item()),
                        }
                        if callable(METRICS_COLLECTOR):
                            METRICS_COLLECTOR(stats)
                        elif hasattr(METRICS_COLLECTOR, 'append'):
                            METRICS_COLLECTOR.append(stats)

            elif mode == "complex54":
                epsilon = float(_get_arg(args, 'snnbp_epsilon', 0.3468))
                alpha = float(_get_arg(args, 'snnbp_alpha', 1.1742))
                beta = float(_get_arg(args, 'snnbp_beta', 0.9245))
                p = float(_get_arg(args, 'snnbp_p', 9.5334))

                m, m_grad = _compute_cgrad_margins(u1, thresh, dL_dS, dL_dU2, reset_mode)
                g_m = torch.sigmoid(alpha * m)
                delta = u1 - thresh
                g_d = torch.sigmoid(beta * (epsilon - delta.abs()))
                g_dir = torch.clamp(-1.0 * torch.sign(m_grad) * torch.sign(base_function), 0.0, 1.0)
                f = torch.clamp(p * g_m * g_d * g_dir, 0.0, 1.0)
                compute_dist = torch.clamp(delta.abs(), min=epsilon)
                dL_dU1 = f * (m_grad / compute_dist) + (1.0 - f) * base_function

            elif mode == "smooth_cgrad":
                epsilon = float(_get_arg(args, 'snnbp_epsilon', 0.3468))
                alpha = float(_get_arg(args, 'snnbp_alpha', 1.1742))
                beta = float(_get_arg(args, 'snnbp_beta', 0.9245))
                p = float(_get_arg(args, 'snnbp_p', 9.5334))
                k_dir = float(_get_arg(args, 'snnbp_k_dir', 1.0))

                m, m_grad = _compute_cgrad_margins(u1, thresh, dL_dS, dL_dU2, reset_mode)
                g_m = torch.sigmoid(alpha * m)
                delta = u1 - thresh
                g_d = torch.sigmoid(beta * (epsilon - delta.abs()))
                g_dir = torch.sigmoid(-k_dir * m_grad * base_function)
                f = torch.clamp(p * g_m * g_d * g_dir, 0.0, 1.0)
                compute_dist = torch.clamp(delta.abs(), min=epsilon)
                dL_dU1 = f * (m_grad / compute_dist) + (1.0 - f) * base_function

            elif mode == "stable_cgrad":
                epsilon = float(_get_arg(args, 'snnbp_epsilon', 0.3))
                alpha = float(_get_arg(args, 'snnbp_alpha', 1.0))
                beta = float(_get_arg(args, 'snnbp_beta', 1.0))
                max_ratio = float(_get_arg(args, 'snnbp_max_ratio', 3.0))
                k_smooth = float(_get_arg(args, 'snnbp_k_dir', 2.0))

                delta = u1 - thresh
                m, m_grad = _compute_cgrad_margins(u1, thresh, dL_dS, dL_dU2, reset_mode)
                log_scale = torch.log1p(delta.abs() / epsilon)
                scaled_correction = m_grad * torch.tanh(log_scale)
                g_m = torch.sigmoid(alpha * m)
                g_d = torch.sigmoid(beta * (epsilon - delta.abs()))
                alignment = -m_grad * base_function
                g_dir = torch.sigmoid(k_smooth * alignment / (base_function.abs() + 1e-8))
                f = g_m * g_d * g_dir
                base_mag = base_function.abs() + 1e-8
                correction = scaled_correction * base_mag * max_ratio
                dL_dU1 = f * correction + (1.0 - f) * base_function

            elif mode == "adaptive_cgrad":
                epsilon = float(_get_arg(args, 'snnbp_epsilon', 0.3))
                alpha = float(_get_arg(args, 'snnbp_alpha', 1.0))
                beta = float(_get_arg(args, 'snnbp_beta', 1.0))
                decay_rate = float(_get_arg(args, 'snnbp_decay', 0.5))
                k_dir = float(_get_arg(args, 'snnbp_k_dir', 1.0))

                delta = u1 - thresh
                m, m_grad = _compute_cgrad_margins(u1, thresh, dL_dS, dL_dU2, reset_mode)
                base_mag = base_function.abs() + 1e-8
                adaptive_scale = 1.0 / (1.0 + decay_rate * base_mag)
                soft_dist = torch.sqrt(delta.abs().pow(2) + epsilon ** 2)
                raw_correction = m_grad / soft_dist
                max_correction = 2.0 * base_mag
                clamped_correction = torch.clamp(raw_correction, -max_correction, max_correction)
                g_m = torch.sigmoid(alpha * m)
                g_d = torch.sigmoid(beta * (epsilon - delta.abs()))
                g_dir = torch.sigmoid(-k_dir * m_grad * base_function / (base_mag ** 2 + 1e-6))
                f = torch.clamp(g_m * g_d * g_dir * adaptive_scale, 0.0, 1.0)
                dL_dU1 = f * clamped_correction + (1.0 - f) * base_function

            elif mode == "conservative_cgrad":
                epsilon = float(_get_arg(args, 'snnbp_epsilon', 0.3))
                alpha = float(_get_arg(args, 'snnbp_alpha', 2.0))
                beta = float(_get_arg(args, 'snnbp_beta', 2.0))
                intervention_threshold = float(_get_arg(args, 'snnbp_intervention', 0.8))

                delta = u1 - thresh
                m, m_grad = _compute_cgrad_margins(u1, thresh, dL_dS, dL_dU2, reset_mode)
                misalignment = -m_grad * base_function
                g_m = torch.sigmoid(alpha * (m.abs() - 0.1))
                g_d = torch.sigmoid(beta * (epsilon - delta.abs()))
                g_misalign = torch.sigmoid(alpha * misalignment)
                intervention_signal = g_m * g_d * g_misalign
                do_intervene = (intervention_signal > intervention_threshold).to(work_dtype)
                soft_correction = torch.sign(m_grad) * (base_function.abs() * 0.5)
                blend_factor = do_intervene * torch.sigmoid(5.0 * (intervention_signal - intervention_threshold))
                dL_dU1 = (1.0 - 0.5 * blend_factor) * base_function + 0.5 * blend_factor * soft_correction

            elif mode == "conservative_cgrad_2":
                epsilon = float(_get_arg(args, 'snnbp_epsilon', 0.3))
                alpha = float(_get_arg(args, 'snnbp_alpha', 2.0))
                beta = float(_get_arg(args, 'snnbp_beta', 2.0))
                intervention_threshold = float(_get_arg(args, 'snnbp_intervention', 0.8))
                eta = float(_get_arg(args, 'snnbp_eta', 0.5))
                p = float(_get_arg(args, 'snnbp_p', 0.5))

                delta = u1 - thresh
                m, m_grad = _compute_cgrad_margins(u1, thresh, dL_dS, dL_dU2, reset_mode)
                misalignment = -m_grad * base_function
                g_m = torch.sigmoid(alpha * (m.abs() - 0.1))
                g_d = torch.sigmoid(beta * (epsilon - delta.abs()))
                g_misalign = torch.sigmoid(p * misalignment)
                intervention_signal = g_m * g_d * g_misalign
                do_intervene = (intervention_signal > intervention_threshold).to(work_dtype)
                soft_correction = torch.sign(m_grad) * (base_function.abs() * eta)
                blend_factor = do_intervene * torch.sigmoid(5.0 * (intervention_signal - intervention_threshold))
                dL_dU1 = (1.0 - 0.5 * blend_factor) * base_function + 0.5 * blend_factor * soft_correction

            elif mode == "conservative_ablate":
                epsilon = float(_get_arg(args, 'snnbp_epsilon', 0.3))
                alpha = float(_get_arg(args, 'snnbp_alpha', 2.0))
                beta = float(_get_arg(args, 'snnbp_beta', 2.0))
                intervention_threshold = float(_get_arg(args, 'snnbp_intervention', 0.8))

                use_gm = bool(_get_arg(args, 'ablation_gm', True))
                use_gd = bool(_get_arg(args, 'ablation_gd', True))
                use_gmisalign = bool(_get_arg(args, 'ablation_gmisalign', True))
                use_intervention = bool(_get_arg(args, 'ablation_intervention', True))

                delta = u1 - thresh
                m, m_grad = _compute_cgrad_margins(u1, thresh, dL_dS, dL_dU2, reset_mode)
                misalignment = -m_grad * base_function
                g_m = torch.sigmoid(alpha * (m.abs() - 0.1)) if use_gm else torch.ones_like(m)
                g_d = torch.sigmoid(beta * (epsilon - delta.abs())) if use_gd else torch.ones_like(delta)
                g_misalign = torch.sigmoid(alpha * misalignment) if use_gmisalign else torch.ones_like(misalignment)
                intervention_signal = g_m * g_d * g_misalign
                do_intervene = (
                    (intervention_signal > intervention_threshold).to(work_dtype)
                    if use_intervention
                    else torch.ones_like(intervention_signal)
                )
                soft_correction = torch.sign(m_grad) * (base_function.abs() * 0.5)
                blend_factor = do_intervene * torch.sigmoid(5.0 * (intervention_signal - intervention_threshold))
                dL_dU1 = (1.0 - 0.5 * blend_factor) * base_function + 0.5 * blend_factor * soft_correction

            elif mode == "invariant_cgrad":
                epsilon = float(_get_arg(args, 'snnbp_epsilon', 0.3))
                alpha = float(_get_arg(args, 'snnbp_alpha', 2.0))
                beta = float(_get_arg(args, 'snnbp_beta', 2.0))
                alpha_c = float(_get_arg(args, 'snnbp_alpha_cons', 2.0))
                kappa = float(_get_arg(args, 'snnbp_kappa', 1.0))
                intervention_threshold = float(_get_arg(args, 'snnbp_h', 0.5))
                eta = float(_get_arg(args, 'snnbp_eta', 0.5))
                gamma = float(_get_arg(args, 'snnbp_gamma', 5.0))
                two_sided = bool(_get_arg(args, 'snnbp_two_sided', True))
                _check_invariant_gate_reachable(beta, epsilon, intervention_threshold)

                delta = u1 - thresh
                rho = thresh - u1
                m, m_grad = _compute_cgrad_margins(u1, thresh, dL_dS, dL_dU2, reset_mode)

                consistency = _get_arg(args, 'snnbp_consistency', 'product')
                if consistency == 'product':
                    dL_pred = rho * base_function
                else:
                    dL_pred = torch.where(u1 < thresh, base_function, -base_function) * thresh

                m32 = m.float()
                pred32 = dL_pred.float()
                norm = torch.sqrt(m32.pow(2).mean() + pred32.pow(2).mean()).clamp(min=1e-30)
                m_hat = (m32 / norm).to(work_dtype)
                pred_hat = (pred32 / norm).to(work_dtype)

                if two_sided:
                    g_m = torch.sigmoid(alpha * (m_hat.abs() - kappa))
                else:
                    g_m = torch.sigmoid(alpha * (m_hat - kappa))
                g_d = torch.sigmoid(beta * (epsilon - delta.abs()))
                if consistency == 'sign':
                    g_cons = torch.sigmoid(alpha_c * (-torch.sign(m_hat) * torch.sign(pred_hat)))
                else:
                    g_cons = torch.sigmoid(alpha_c * (-m_hat * pred_hat))

                intervention_signal = (g_m * g_d * g_cons).pow(1.0 / 3.0)
                do_intervene = (intervention_signal > intervention_threshold).to(work_dtype)

                soft_correction = torch.sign(m_grad) * base_function.abs()
                blend_factor = do_intervene * torch.sigmoid(
                    gamma * (intervention_signal - intervention_threshold)
                )
                w = eta * blend_factor

                correction = _get_arg(args, 'snnbp_correction', 'blend')
                if correction == 'mask':
                    dL_dU1 = (1.0 - do_intervene) * base_function
                    eff_w = do_intervene
                elif correction == 'flip':
                    dL_dU1 = torch.where(do_intervene.bool(), soft_correction, base_function)
                    eff_w = do_intervene
                else:
                    dL_dU1 = (1.0 - w) * base_function + w * soft_correction
                    eff_w = w

                if do_collect:
                    _accumulate_cgrad_stats(
                        do_intervene.detach(),
                        eff_w.detach(),
                        intervention_signal.detach(),
                        m.detach(),
                        delta.abs() <= epsilon,
                        base_function.detach(),
                    )

            elif mode == "":
                raise ValueError("Empty du_du mode is invalid")

            elif mode == "TET":
                dL_dU1 = base_function
            else:
                # Default LIF
                dL_dU1 = base_function

            grad_memb_last = dL_dU1
            grad_x[t] = (grad_memb_last * input_scale).to(grad_output.dtype)

        return grad_x, None, None, None, None, None, None, None, None


## --- Native Drop-in Replacement ---
class LIFSpikeLayer_Cons(nn.Module):
    def __init__(
        self,
        thresh=1.0,
        tau=2.0,
        gama=1.0,
        detach_reset=True,
        input_scale='coupled',
        reset_mode='hard',
        args=None,
        **kwargs,
    ):
        super(LIFSpikeLayer_Cons, self).__init__()
        if 'v_threshold' in kwargs:
            thresh = kwargs['v_threshold']
        self.thresh = float(thresh)
        self.tau = float(tau)
        self.gama = float(gama)
        self.detach_reset = bool(detach_reset)
        self.raw_input_scale = input_scale
        self.reset_mode = reset_mode
        self._explicit_args = args

        active_args = self._explicit_args if self._explicit_args is not None else GLOBAL_ARGS
        if active_args is not None and getattr(active_args, 'reset_mode', None) is not None:
            self.reset_mode = active_args.reset_mode
        if active_args is not None and getattr(active_args, 'detach_reset', None) is not None:
            self.detach_reset = bool(active_args.detach_reset)
        self.decay, self.input_scale = _resolve_tau_and_scale(self.tau, self.raw_input_scale, active_args)

    @property
    def args(self):
        return self._explicit_args if self._explicit_args is not None else GLOBAL_ARGS

    def forward(self, x):
        active_args = self.args
        decay, input_scale = _resolve_tau_and_scale(self.tau, self.raw_input_scale, active_args)
        reset_mode = (
            active_args.reset_mode
            if (active_args is not None and getattr(active_args, 'reset_mode', None) is not None)
            else self.reset_mode
        )
        detach_reset = (
            bool(active_args.detach_reset)
            if (active_args is not None and getattr(active_args, 'detach_reset', None) is not None)
            else self.detach_reset
        )

        # Keep cached attributes synchronized with active args
        self.decay = decay
        self.input_scale = input_scale

        mode = _get_arg(active_args, 'du_du', 'complex54')
        norm_mode = _get_arg(active_args, 'snnbp_norm', 'local')

        collect_stats = (
            _CGRAD_STATS is not None
            and self.training
            and mode in ('improved_cgrad', 'invariant_cgrad')
            and random.random() < _CGRAD_STATS_P
        )

        triton_supported_mode = mode in (
            'LIF',
            'TET',
            'conservative_cgrad',
            'invariant_cgrad',
            'improved_cgrad',
        )
        triton_supported_norm = not (mode == 'improved_cgrad' and norm_mode == 'rms')
        use_triton = (
            TRITON_AVAILABLE
            and x.is_cuda
            and x.dtype in (torch.float32, torch.float16, torch.bfloat16)
            and (os.environ.get('DISABLE_TRITON', '0') != '1')
            and triton_supported_mode
            and triton_supported_norm
            and not collect_stats
        )
        if use_triton:
            return TritonTimeParallel_LIFSpike.apply(
                x, self.thresh, decay, input_scale, self.gama, active_args, detach_reset, reset_mode
            )
        return TimeParallel_LIFSpike.apply(
            x, self.thresh, decay, input_scale, self.gama, active_args, detach_reset, reset_mode, collect_stats
        )


class MultiStepLIFNode(nn.Module):
    def __init__(
        self,
        thresh=1.0,
        tau=2.0,
        gama=1.0,
        detach_reset=True,
        input_scale='coupled',
        reset_mode='hard',
        args=None,
        **kwargs,
    ):
        super(MultiStepLIFNode, self).__init__()

        active_args = args if args is not None else GLOBAL_ARGS
        use_custom = True
        if active_args is not None and getattr(active_args, 'use_custom_neuron', None) is not None:
            use_custom = active_args.use_custom_neuron

        real_detach_reset = detach_reset
        if active_args is not None and getattr(active_args, 'detach_reset', None) is not None:
            real_detach_reset = bool(active_args.detach_reset)

        real_thresh = thresh
        if 'v_threshold' in kwargs:
            real_thresh = kwargs['v_threshold']

        if use_custom:
            self.impl = LIFSpikeLayer_Cons(
                thresh=real_thresh,
                tau=tau,
                gama=gama,
                detach_reset=real_detach_reset,
                input_scale=input_scale,
                reset_mode=reset_mode,
                args=args,
                **kwargs,
            )
        else:
            if OriginalLIFNode is None:
                raise ImportError("spikingjelly is required when use_custom_neuron=False.")
            oj_kwargs = kwargs.copy()
            if 'v_threshold' not in oj_kwargs:
                oj_kwargs['v_threshold'] = real_thresh
            if DISABLE_CUPY or (active_args is not None and getattr(active_args, 'disable_cupy', False)):
                oj_kwargs['backend'] = 'torch'
            self.impl = OriginalLIFNode(
                tau=tau,
                detach_reset=real_detach_reset,
                **oj_kwargs,
            )

    def forward(self, x):
        return self.impl(x)

    def reset(self):
        if hasattr(self.impl, 'reset'):
            self.impl.reset()

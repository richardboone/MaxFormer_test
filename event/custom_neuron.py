import torch
import torch.nn as nn
import math
import random
from spikingjelly.clock_driven.neuron import MultiStepLIFNode as OriginalLIFNode

# --- Global Args Handler ---
GLOBAL_ARGS = None
_INVARIANT_GATE_CHECKED = False

def set_global_args(args):
    global GLOBAL_ARGS
    GLOBAL_ARGS = args


# --- C-Grad intervention statistics -----------------------------------------
# Off by default. When enabled, invariant_cgrad accumulates how often it actually
# intervenes. This matters for hyperparameter sweeps: a configuration whose gate
# never opens trains as plain BPTT, scores like the baseline, and is otherwise
# indistinguishable from a genuinely good configuration -- so without this metric
# a sweep cannot tell "good hyperparameters" from "did nothing".
#
# All accumulation stays on-device so the backward pass never blocks on a host
# sync; the single sync happens in pop_cgrad_stats() once per epoch. Collection is
# sampled (see enable_cgrad_stats) to keep the cost off the critical path.
_CGRAD_STATS = None
_CGRAD_STATS_P = 0.125      # sampling probability per backward call
_CGRAD_TOTAL = 0.0          # kept on the host: numel needs no kernel

_STAT_KEYS = ('fired', 'm_neg', 'signal_sum', 'near_thresh')


def enable_cgrad_stats(device, sample_prob=0.125):
    """Start collecting intervention statistics.

    Statistics are gathered on a random `sample_prob` fraction of backward calls.
    A training run touches hundreds of millions of neuron-timesteps, so an eighth
    of them estimates the intervention rate to far better precision than needed,
    at roughly an eighth of the cost. Sampling is random rather than strided
    because a fixed stride can alias against the number of neuron layers per step
    and silently always sample the same layers.
    """
    global _CGRAD_STATS, _CGRAD_STATS_P, _CGRAD_TOTAL
    _CGRAD_STATS = {k: torch.zeros((), device=device, dtype=torch.float64)
                    for k in _STAT_KEYS}
    _CGRAD_STATS_P = sample_prob
    _CGRAD_TOTAL = 0.0


def disable_cgrad_stats():
    global _CGRAD_STATS, _CGRAD_TOTAL
    _CGRAD_STATS = None
    _CGRAD_TOTAL = 0.0


def pop_cgrad_stats():
    """Return accumulated statistics as plain floats and reset the counters.

    Returns None when collection is disabled or nothing was sampled. Performs one
    device->host sync, once per epoch.
    """
    global _CGRAD_TOTAL
    if _CGRAD_STATS is None or _CGRAD_TOTAL == 0:
        return None
    total = _CGRAD_TOTAL
    fired = _CGRAD_STATS['fired'].item()
    out = {
        'cgrad/intervention_rate': fired / total,
        'cgrad/mean_signal': _CGRAD_STATS['signal_sum'].item() / total,
        'cgrad/near_threshold_rate': _CGRAD_STATS['near_thresh'].item() / total,
        # Fraction of interventions that act on a crossing predicted to REDUCE
        # loss. Sec 6.1 says these should not occur; in practice they are about
        # half of all interventions, and suppressing them collapses training.
        'cgrad/frac_interventions_m_neg':
            (_CGRAD_STATS['m_neg'].item() / fired) if fired > 0 else 0.0,
        'cgrad/neuron_timesteps_sampled': total,
    }
    for k in _STAT_KEYS:
        _CGRAD_STATS[k].zero_()
    _CGRAD_TOTAL = 0.0
    return out


def _accumulate_cgrad_stats(do_intervene, signal, m, near_thresh):
    """Accumulate on-device; no host sync, no effect on the returned gradient.

    Reductions run in fp32 rather than fp64: every accumulated quantity is either
    a count or a sum of values in [0,1], so fp32 is exact for counts up to 2**24
    and the fp64 scalar accumulators still carry the running totals.
    """
    global _CGRAD_TOTAL
    s = _CGRAD_STATS
    s['fired'] += do_intervene.sum(dtype=torch.float32)
    s['m_neg'] += (do_intervene * (m < 0)).sum(dtype=torch.float32)
    s['signal_sum'] += signal.sum(dtype=torch.float32)
    s['near_thresh'] += near_thresh.sum(dtype=torch.float32)
    _CGRAD_TOTAL += float(do_intervene.numel())


def _check_invariant_gate_reachable(alpha_d, epsilon, h):
    """Warn once if invariant_cgrad is configured so that it can never intervene.

    The intervention signal is the geometric mean of three gates, and g_d cannot
    exceed sigmoid(alpha_d*epsilon), so the signal is capped at
    sigmoid(alpha_d*epsilon)**(1/3). An h at or above that cap makes the mode a
    silent no-op -- it trains, it converges, and it is simply standard BPTT.
    conservative_cgrad has this failure mode with the benchmark defaults and gives
    no indication of it, so it is worth surfacing here.
    """
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

# --- Custom Autograd Function ---
class TimeParallel_LIFSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, thresh, decay, input_scale, gama, args, detach_reset, reset_mode):
        # SpikingJelly Standard: [Time, Batch, *Spatial]
        T, batch_size, *spatial_dims = x.shape
        device = x.device
        
        # Initialize membrane (Batch, Spatial)
        mem = torch.zeros(batch_size, *spatial_dims, device=device)
        
        # Storage lists
        mem_before_spikes = []
        spikes = []
        mems_after_spikes = []
        
        for t in range(T):
            # Update membrane: mem * decay + input[t] * input_scale
            # This matches SpikingJelly: v += (x - v) / tau  =>  v = v(1-1/tau) + x(1/tau)
            mem_before_spike = mem * decay + x[t] * input_scale
            mem_before_spikes.append(mem_before_spike.clone())
            
            # Spike generation
            spike = (mem_before_spike > thresh).float()
            spikes.append(spike)
            
            # Reset
            if reset_mode == 'soft':
                mem = mem_before_spike - thresh * spike
            else:  # 'hard'
                mem = mem_before_spike * (1 - spike)
            # mems_after_spikes.append(mem)
        
        # Stack along dimension 0 (Time)
        mem_before_spikes = torch.stack(mem_before_spikes, dim=0)
        spikes_tensor = torch.stack(spikes, dim=0)
        # mems_after_spikes = torch.stack(mems_after_spikes, dim=0)
        
        # Save context
        ctx.save_for_backward(mem_before_spikes, spikes_tensor, x, torch.tensor([gama]))
        ctx.thresh = thresh
        ctx.decay = decay
        ctx.input_scale = input_scale
        ctx.args = args
        ctx.detach_reset = detach_reset
        ctx.reset_mode = reset_mode
        
        return spikes_tensor

    @staticmethod
    def backward(ctx, grad_output):
        mem_before_spikes, spikes_tensor, x, gama = ctx.saved_tensors
        thresh = ctx.thresh
        decay = ctx.decay
        input_scale = ctx.input_scale
        args = ctx.args
        detach_reset = ctx.detach_reset
        reset_mode = ctx.reset_mode
        
        # Output Gradients
        grad_x = torch.zeros_like(grad_output)
        
        # Gradient accumulator for the next timestep (starts at 0 for last step)
        grad_memb_last = torch.zeros_like(mem_before_spikes[0]) 
        
        # Helper for Surrogate Gradient (dS/dU)
        def get_dS_dU1(u, thresh, gama, args):
            mode = getattr(args, 'dS_du', 'Gamma')
            if mode == "sigmoid":
                # SpikingJelly default alpha is often 4.0
                alpha = getattr(args, 'surrogate_alpha', 4.0) 
                sgax = (u - thresh) * alpha
                return (1.0 - torch.sigmoid(sgax)) * torch.sigmoid(sgax) * alpha
                
            elif mode == "Gamma":
                # Standard triangular surrogate
                return (1 / gama.item()**2) * (gama.item() - (u - thresh).abs()).clamp(min=0)
            
            # Default fallback
            return (1 / gama.item()**2) * (gama.item() - (u - thresh).abs()).clamp(min=0)

        # Backward Time Loop (Iterate T from end to start)
        for t in reversed(range(x.shape[0])):
            u1 = mem_before_spikes[t]
            dL_dS = grad_output[t]        # Gradient from loss w.r.t Spike[t]
            dL_dU2 = grad_memb_last * decay # Propagate via decay factor
            
            dS_dU1 = get_dS_dU1(u1, thresh, gama, args)
            
            # --- CUSTOM GRADIENT LOGIC ---
            mode = getattr(args, 'du_du', 'complex54')
            
            if reset_mode == 'soft':
                if detach_reset:
                    dU2_dU1_standard = torch.ones_like(u1)
                else:
                    # Full gradient: 1 - V_th * dS/dU
                    dU2_dU1_standard = 1 - thresh * dS_dU1
            else:  # 'hard'
                if detach_reset:
                    # Reset gradient is just (1 - S)
                    dU2_dU1_standard = (1 - spikes_tensor[t])
                else:
                    # Full gradient: (1 - S) - U * dS/dU
                    dU2_dU1_standard = (1 - spikes_tensor[t]) - (u1 * dS_dU1)

            if mode == "complex54":
                epsilon = getattr(args, 'snnbp_epsilon', 0.3468)
                alpha = getattr(args, 'snnbp_alpha', 1.1742)
                beta = getattr(args, 'snnbp_beta', 0.9245)
                p = getattr(args, 'snnbp_p', 9.5334)

                if reset_mode == 'soft':
                    term_supra_threshold = (2 * thresh - u1) * dL_dU2 - dL_dS
                else:
                    term_supra_threshold = (thresh * dL_dU2) - dL_dS
                term_sub_threshold = dL_dS - (u1 * dL_dU2)
                
                m = torch.where(u1 < thresh, term_sub_threshold, term_supra_threshold)
                m_grad = torch.where(u1 < thresh, m, -m)

                # Base Function (Standard BPTT)
                base_function = dL_dS * dS_dU1 + dL_dU2 * dU2_dU1_standard

                # Gates
                g_m = torch.sigmoid(alpha * m)
                delta = u1 - thresh
                g_d = torch.sigmoid(beta * (epsilon - delta.abs()))
                
                # Directionality
                g_dir = torch.clamp(-1 * torch.sign(m_grad) * torch.sign(base_function), 0, 1)
                
                # Fusion
                f = torch.clamp(p * g_m * g_d * g_dir, 0, 1)
                
                # Final Gradient
                compute_dist = torch.max(delta.abs(), epsilon * torch.ones_like(delta))
                dL_dU1 = f * (m_grad / (compute_dist)) + (1 - f) * base_function
            
            elif mode == "smooth_cgrad":
                epsilon = getattr(args, 'snnbp_epsilon', 0.3468)
                alpha = getattr(args, 'snnbp_alpha', 1.1742)
                beta = getattr(args, 'snnbp_beta', 0.9245)
                p = getattr(args, 'snnbp_p', 9.5334)
                k_dir = getattr(args, 'snnbp_k_dir', 1.0)

                if reset_mode == 'soft':
                    term_supra_threshold = (2 * thresh - u1) * dL_dU2 - dL_dS
                else:
                    term_supra_threshold = (thresh * dL_dU2) - dL_dS
                term_sub_threshold = dL_dS - (u1 * dL_dU2)
                
                m = torch.where(u1 < thresh, term_sub_threshold, term_supra_threshold)
                m_grad = torch.where(u1 < thresh, m, -m)

                # Base Function (Standard BPTT)
                base_function = dL_dS * dS_dU1 + dL_dU2 * dU2_dU1_standard

                # Gates
                g_m = torch.sigmoid(alpha * m)
                delta = u1 - thresh
                g_d = torch.sigmoid(beta * (epsilon - delta.abs()))
                
                # Directionality (Smoothed)
                g_dir = torch.sigmoid(-k_dir * m_grad * base_function)
                
                # Fusion
                f = torch.clamp(p * g_m * g_d * g_dir, 0, 1)
                
                # Final Gradient
                compute_dist = torch.max(delta.abs(), epsilon * torch.ones_like(delta))
                dL_dU1 = f * (m_grad / (compute_dist)) + (1 - f) * base_function

            elif mode == "stable_cgrad":
                """
                Stabilized Custom Gradient
                --------------------------
                Key stability improvements:
                1. Logarithmic scaling instead of division to prevent gradient explosion
                2. Soft magnitude clamping using tanh
                3. Smoother blending via softplus gates
                4. Gradient magnitude bounded relative to base_function
                """
                epsilon = getattr(args, 'snnbp_epsilon', 0.3)
                alpha = getattr(args, 'snnbp_alpha', 1.0)
                beta = getattr(args, 'snnbp_beta', 1.0)
                max_ratio = getattr(args, 'snnbp_max_ratio', 3.0)  # Max multiplier vs base gradient
                
                delta = u1 - thresh
                
                # Compute correction terms
                if reset_mode == 'soft':
                    term_supra_threshold = (2 * thresh - u1) * dL_dU2 - dL_dS
                else:
                    term_supra_threshold = (thresh * dL_dU2) - dL_dS
                term_sub_threshold = dL_dS - (u1 * dL_dU2)
                
                m = torch.where(u1 < thresh, term_sub_threshold, term_supra_threshold)
                m_grad = torch.where(u1 < thresh, m, -m)
                
                # Base Function (Standard BPTT)
                base_function = dL_dS * dS_dU1 + dL_dU2 * dU2_dU1_standard
                
                # Stability improvement 1: Use log-based scaling instead of division
                # log(1 + |delta|/epsilon) provides smooth, bounded scaling
                log_scale = torch.log1p(delta.abs() / epsilon)
                scaled_correction = m_grad * torch.tanh(log_scale)  # tanh bounds to [-1, 1]
                
                # Stability improvement 2: Soft gates using softplus (smoother than sigmoid)
                g_m = torch.sigmoid(alpha * m)  # Magnitude gate
                g_d = torch.sigmoid(beta * (epsilon - delta.abs()))  # Distance gate
                
                # Stability improvement 3: Smooth directionality (avoid hard sign)
                k_smooth = getattr(args, 'snnbp_k_dir', 2.0)
                alignment = -m_grad * base_function  # Negative when opposing
                g_dir = torch.sigmoid(k_smooth * alignment / (base_function.abs() + 1e-8))
                
                # Combined blend factor (no p multiplier, inherently bounded 0-1)
                f = g_m * g_d * g_dir
                
                # Stability improvement 4: Bound correction magnitude relative to base
                base_mag = base_function.abs() + 1e-8
                correction = scaled_correction * base_mag * max_ratio
                
                # Final blended gradient
                dL_dU1 = f * correction + (1 - f) * base_function

            elif mode == "adaptive_cgrad":
                """
                Adaptive Custom Gradient
                ------------------------
                Key stability improvements:
                1. Normalizes correction by local gradient statistics
                2. Uses running estimates to adapt correction strength
                3. EMA-style smoothing for magnitude bounds
                4. Correction strength decays when base gradient is large
                """
                epsilon = getattr(args, 'snnbp_epsilon', 0.3)
                alpha = getattr(args, 'snnbp_alpha', 1.0)
                beta = getattr(args, 'snnbp_beta', 1.0)
                decay_rate = getattr(args, 'snnbp_decay', 0.5)
                
                delta = u1 - thresh
                
                # Compute correction terms
                if reset_mode == 'soft':
                    term_supra_threshold = (2 * thresh - u1) * dL_dU2 - dL_dS
                else:
                    term_supra_threshold = (thresh * dL_dU2) - dL_dS
                term_sub_threshold = dL_dS - (u1 * dL_dU2)
                
                m = torch.where(u1 < thresh, term_sub_threshold, term_supra_threshold)
                m_grad = torch.where(u1 < thresh, m, -m)
                
                # Base Function (Standard BPTT)
                base_function = dL_dS * dS_dU1 + dL_dU2 * dU2_dU1_standard
                base_mag = base_function.abs() + 1e-8
                
                # Adaptive: Use base gradient magnitude to normalize correction
                # When base gradient is large, we trust it more (less correction)
                # When base gradient is small, correction has more influence
                adaptive_scale = 1.0 / (1.0 + decay_rate * base_mag)
                
                # Bounded correction: divide by soft distance, clamp magnitude
                soft_dist = torch.sqrt(delta.abs().pow(2) + epsilon**2)
                raw_correction = m_grad / soft_dist
                
                # Clamp correction to be within reasonable bounds of base gradient
                max_correction = 2.0 * base_mag
                clamped_correction = torch.clamp(raw_correction, -max_correction, max_correction)
                
                # Gates
                g_m = torch.sigmoid(alpha * m)
                g_d = torch.sigmoid(beta * (epsilon - delta.abs()))
                
                # Smooth directionality  
                k_dir = getattr(args, 'snnbp_k_dir', 1.0)
                g_dir = torch.sigmoid(-k_dir * m_grad * base_function / (base_mag**2 + 1e-6))
                
                # Blend factor with adaptive scaling
                f = torch.clamp(g_m * g_d * g_dir * adaptive_scale, 0, 1)
                
                # Final gradient
                dL_dU1 = f * clamped_correction + (1 - f) * base_function

            elif mode == "conservative_cgrad":
                """
                Conservative Custom Gradient
                ----------------------------
                Key stability improvements:
                1. Only intervenes in high-confidence problematic cases
                2. Uses higher thresholds for activation
                3. Interpolates toward base gradient rather than custom correction
                4. Minimal disruption to standard training dynamics
                """
                epsilon = getattr(args, 'snnbp_epsilon', 0.3)
                alpha = getattr(args, 'snnbp_alpha', 2.0)  # Higher = more selective
                beta = getattr(args, 'snnbp_beta', 2.0)
                intervention_threshold = getattr(args, 'snnbp_intervention', 0.8)
                eta = getattr(args, 'snnbp_eta', 0.5)        # Correction magnitude constant
                p = getattr(args, 'snnbp_p', 0.5)
                blend_scale = getattr(args, 'snnbp_blend', 0.5)  # Blend scale factor
                
                delta = u1 - thresh
                
                # Compute correction terms
                if reset_mode == 'soft':
                    term_supra_threshold = (2 * thresh - u1) * dL_dU2 - dL_dS
                else:
                    term_supra_threshold = (thresh * dL_dU2) - dL_dS
                term_sub_threshold = dL_dS - (u1 * dL_dU2)
                
                m = torch.where(u1 < thresh, term_sub_threshold, term_supra_threshold)
                m_grad = torch.where(u1 < thresh, m, -m)
                
                # Base Function (Standard BPTT)
                base_function = dL_dS * dS_dU1 + dL_dU2 * dU2_dU1_standard
                
                # Conservative approach: only care about clear misalignment
                # Detect when base gradient would push toward undesirable threshold crossing
                misalignment = -m_grad * base_function  # Positive when base opposes desired direction
                
                # High-threshold gates (only activate in clear cases)
                g_m = torch.sigmoid(alpha * (m.abs() - 0.1))  # Only for significant m
                g_d = torch.sigmoid(beta * (epsilon - delta.abs()))  # Near threshold
                g_misalign = torch.sigmoid(alpha * misalignment)  # Clear misalignment
                
                # Combined intervention signal
                intervention_signal = g_m * g_d * g_misalign
                
                # Only intervene past threshold
                do_intervene = (intervention_signal > intervention_threshold).float()
                
                # Soft intervention: interpolate toward corrected direction
                # Instead of replacing gradient, dampen the problematic component
                correction_direction = torch.sign(m_grad)
                correction_magnitude = base_function.abs() * 0.5  # Match base scale
                soft_correction = correction_direction * correction_magnitude
                
                # Smooth blending (gradual damping, not hard switch)
                blend_factor = do_intervene * torch.sigmoid(5 * (intervention_signal - intervention_threshold))
                
                # Final gradient: mostly base, with soft correction in extreme cases
                dL_dU1 = (1 - 0.5 * blend_factor) * base_function + 0.5 * blend_factor * soft_correction

            elif mode == "conservative_cgrad_2":
                """
                Conservative Custom Gradient
                ----------------------------
                Key stability improvements:
                1. Only intervenes in high-confidence problematic cases
                2. Uses higher thresholds for activation
                3. Interpolates toward base gradient rather than custom correction
                4. Minimal disruption to standard training dynamics
                """
                epsilon = getattr(args, 'snnbp_epsilon', 0.3)
                alpha = getattr(args, 'snnbp_alpha', 2.0)  # Higher = more selective
                beta = getattr(args, 'snnbp_beta', 2.0)
                intervention_threshold = getattr(args, 'snnbp_intervention', 0.8)
                eta = getattr(args, 'snnbp_eta', 0.5)        # Correction magnitude constant
                p = getattr(args, 'snnbp_p', 0.5)
                
                delta = u1 - thresh
                
                # Compute correction terms
                if reset_mode == 'soft':
                    term_supra_threshold = (2 * thresh - u1) * dL_dU2 - dL_dS
                else:
                    term_supra_threshold = (thresh * dL_dU2) - dL_dS
                term_sub_threshold = dL_dS - (u1 * dL_dU2)
                
                m = torch.where(u1 < thresh, term_sub_threshold, term_supra_threshold)
                m_grad = torch.where(u1 < thresh, m, -m)
                
                # Base Function (Standard BPTT)
                base_function = dL_dS * dS_dU1 + dL_dU2 * dU2_dU1_standard
                
                # Conservative approach: only care about clear misalignment
                # Detect when base gradient would push toward undesirable threshold crossing
                misalignment = -m_grad * base_function  # Positive when base opposes desired direction
                
                # High-threshold gates (only activate in clear cases)
                g_m = torch.sigmoid(alpha * (m.abs() - 0.1))  # Only for significant m
                g_d = torch.sigmoid(beta * (epsilon - delta.abs()))  # Near threshold
                g_misalign = torch.sigmoid(p * misalignment)  # Clear misalignment
                
                # Combined intervention signal
                intervention_signal = g_m * g_d * g_misalign
                
                # Only intervene past threshold
                do_intervene = (intervention_signal > intervention_threshold).float()
                
                # Soft intervention: interpolate toward corrected direction
                # Instead of replacing gradient, dampen the problematic component
                correction_direction = torch.sign(m_grad)
                correction_magnitude = base_function.abs() * eta  # Match base scale
                soft_correction = correction_direction * correction_magnitude
                
                # Smooth blending (gradual damping, not hard switch)
                blend_factor = do_intervene * torch.sigmoid(5 * (intervention_signal - intervention_threshold))
                
                dL_dU1 = (1 - 0.5 * blend_factor) * base_function + 0.5 * blend_factor * soft_correction

            elif mode == "conservative_ablate":
                """
                Conservative Custom Gradient — Ablation Variant
                ------------------------------------------------
                Identical to conservative_cgrad but reads ablation toggles
                from args to selectively disable individual gates:
                  - ablation_gm:           enable/disable g_m (magnitude gate)
                  - ablation_gd:           enable/disable g_d (proximity gate)
                  - ablation_gmisalign:    enable/disable g_misalign (alignment gate)
                  - ablation_intervention: enable/disable hard intervention threshold
                When a gate is OFF it is replaced with 1.0 (always-on),
                removing its filtering effect.
                """
                epsilon = getattr(args, 'snnbp_epsilon', 0.3)
                alpha = getattr(args, 'snnbp_alpha', 2.0)
                beta = getattr(args, 'snnbp_beta', 2.0)
                intervention_threshold = getattr(args, 'snnbp_intervention', 0.8)

                # Ablation toggles (default True = enabled = full method)
                use_gm = getattr(args, 'ablation_gm', True)
                use_gd = getattr(args, 'ablation_gd', True)
                use_gmisalign = getattr(args, 'ablation_gmisalign', True)
                use_intervention = getattr(args, 'ablation_intervention', True)

                delta = u1 - thresh

                # Compute correction terms
                if reset_mode == 'soft':
                    term_supra_threshold = (2 * thresh - u1) * dL_dU2 - dL_dS
                else:
                    term_supra_threshold = (thresh * dL_dU2) - dL_dS
                term_sub_threshold = dL_dS - (u1 * dL_dU2)

                m = torch.where(u1 < thresh, term_sub_threshold, term_supra_threshold)
                m_grad = torch.where(u1 < thresh, m, -m)

                # Base Function (Standard BPTT)
                base_function = dL_dS * dS_dU1 + dL_dU2 * dU2_dU1_standard

                # Conservative approach: only care about clear misalignment
                misalignment = -m_grad * base_function

                # Gates (conditionally ablated)
                g_m = torch.sigmoid(alpha * (m.abs() - 0.1)) if use_gm else torch.ones_like(m)
                g_d = torch.sigmoid(beta * (epsilon - delta.abs())) if use_gd else torch.ones_like(delta)
                g_misalign = torch.sigmoid(alpha * misalignment) if use_gmisalign else torch.ones_like(misalignment)

                # Combined intervention signal
                intervention_signal = g_m * g_d * g_misalign

                # Only intervene past threshold (or always intervene if ablated)
                if use_intervention:
                    do_intervene = (intervention_signal > intervention_threshold).float()
                else:
                    do_intervene = torch.ones_like(intervention_signal)

                # Soft intervention: interpolate toward corrected direction
                correction_direction = torch.sign(m_grad)
                correction_magnitude = base_function.abs() * 0.5
                soft_correction = correction_direction * correction_magnitude

                # Smooth blending
                blend_factor = do_intervene * torch.sigmoid(5 * (intervention_signal - intervention_threshold))

                # Final gradient
                dL_dU1 = (1 - 0.5 * blend_factor) * base_function + 0.5 * blend_factor * soft_correction

            elif mode == "invariant_cgrad":
                """
                Scale-Invariant C-Grad
                ----------------------
                Same three-gate structure as conservative_cgrad, but the gates are
                dimensionless, so the rule is equivariant under rescaling of the loss:
                replacing L by c*L leaves the produced gradient direction unchanged and
                scales its magnitude by exactly c.

                Motivation. dL_dS and dL_dU2 -- and therefore m[t] and base_function --
                carry units of loss, and loss has no canonical scale. It is set by the
                AMP GradScaler (65536 by default, and halved/doubled dynamically during
                training), by the batch size under mean reduction, by the TET lambda, and
                by depth: the first patch-embed layer and the last transformer block see
                gradients orders of magnitude apart inside a single backward pass.
                conservative_cgrad compares |m| against the constant 0.1 and feeds
                m*base_function (quadratic in the loss scale) into a sigmoid, so its gates
                open and close for reasons unrelated to the neuron. Consistency itself is
                a sign test and is scale-free; this mode makes the gates scale-free too.

                Construction. Following Sec. 4 of the paper, write the two competing
                estimates of the loss change induced by a minimal threshold crossing:

                    dL_pred = rho[t] * base_function      (Eq. 6, surrogate-predicted)
                    dL_jump = m[t]                        (Eq. 7/13, jump-induced)

                Both carry units of loss, so their ratio is dimensionless and their
                product normalised by a common loss-scale statistic is too. The update is
                inconsistent (Eq. 10) exactly when dL_pred * dL_jump < 0. Normalising both
                by the RMS loss-scale of the current tensor removes the scale and, because
                the statistic is computed per layer and per timestep, also equalises the
                gate across depth.

                Hyperparameters. alpha_m, alpha_d, h and eta keep their conservative_cgrad
                meanings; kappa is the harmfulness margin (now relative to the typical
                harmfulness in this layer rather than an absolute loss), and alpha_c
                replaces the reuse of alpha_m for the consistency gate. eta is applied
                here, unlike in conservative_cgrad where it is read but never used.
                snnbp_blend is deliberately not read: Table 3 lists b as a hyperparameter,
                but in Algorithm 1 b is the computed smooth factor sigma(gamma*(w-h)) and
                a separate constant multiplier would be redundant with eta.
                """
                epsilon = getattr(args, 'snnbp_epsilon', 0.3)
                alpha = getattr(args, 'snnbp_alpha', 2.0)          # alpha_m, harmfulness
                beta = getattr(args, 'snnbp_beta', 2.0)            # alpha_d, proximity
                alpha_c = getattr(args, 'snnbp_alpha_cons', 2.0)   # consistency gate
                kappa = getattr(args, 'snnbp_kappa', 1.0)          # relative margin on |m|
                # Deliberately NOT snnbp_intervention: that parameter defaults to 0.8 for
                # conservative_cgrad, and 0.8 drives this mode to a 0.00% intervention
                # rate. Sharing it would silently turn the method into a no-op.
                intervention_threshold = getattr(args, 'snnbp_h', 0.5)
                eta = getattr(args, 'snnbp_eta', 0.5)              # correction strength
                gamma = getattr(args, 'snnbp_gamma', 5.0)          # blend sharpness
                two_sided = getattr(args, 'snnbp_two_sided', True)
                _check_invariant_gate_reachable(beta, epsilon, intervention_threshold)

                delta = u1 - thresh
                rho = thresh - u1

                # Jump-induced loss change m[t] (Eq. 13 hard reset / Eq. 22 soft reset)
                if reset_mode == 'soft':
                    term_supra_threshold = (2 * thresh - u1) * dL_dU2 - dL_dS
                else:
                    term_supra_threshold = (thresh * dL_dU2) - dL_dS
                term_sub_threshold = dL_dS - (u1 * dL_dU2)

                m = torch.where(u1 < thresh, term_sub_threshold, term_supra_threshold)
                m_grad = torch.where(u1 < thresh, m, -m)

                # Base Function (Standard BPTT)
                base_function = dL_dS * dS_dU1 + dL_dU2 * dU2_dU1_standard

                # Surrogate-predicted loss change under the minimal crossing (Eq. 6)
                dL_pred = rho * base_function

                # Common loss-scale statistic for this layer at this timestep. Both terms
                # carry units of loss, so norm scales linearly with the loss scale and
                # dividing by it cancels the scale exactly. Reduced over every element
                # (batch and features) of the current tensor, which also equalises the
                # gate across depth.
                #
                # Computed in fp32 regardless of the incoming dtype: under AMP the loss
                # scale pushes |m| into the thousands, and m**2 would overflow fp16 to inf.
                # The normalised quantities are O(1), so casting them back is always safe.
                m32 = m.float()
                pred32 = dL_pred.float()
                norm = torch.sqrt(m32.pow(2).mean() + pred32.pow(2).mean()).clamp(min=1e-30)
                m_hat = (m32 / norm).to(m.dtype)
                pred_hat = (pred32 / norm).to(m.dtype)

                # Gate 1: harmfulness, now relative to the typical harmfulness in this
                # layer. kappa is dimensionless, unlike the hardcoded 0.1 it replaces.
                if two_sided:
                    g_m = torch.sigmoid(alpha * (m_hat.abs() - kappa))
                else:
                    # Paper Sec. 6.1 as literally written: act only on harmful crossings.
                    g_m = torch.sigmoid(alpha * (m_hat - kappa))

                # Gate 2: nearness to threshold. Unchanged -- delta is a membrane
                # potential measured in units of V_th, which is a genuine physical scale.
                g_d = torch.sigmoid(beta * (epsilon - delta.abs()))

                # Gate 3: consistency (Eq. 10). The product is negative exactly when the
                # surrogate-predicted and jump-induced loss changes disagree in sign, so
                # this gate opens precisely on inconsistent updates.
                g_cons = torch.sigmoid(alpha_c * (-m_hat * pred_hat))

                # Combined intervention signal, as the geometric mean of the three gates
                # rather than their raw product.
                #
                # A product of three sigmoids is badly miscalibrated as a soft AND: each
                # factor is bounded well below 1 (g_d in particular can never exceed
                # sigmoid(alpha_d*epsilon), which is 0.62 at the paper's defaults), so the
                # product has a ceiling near 0.3 and any h above that makes the rule dead
                # -- silently, with no error and no gradient change. That is exactly the
                # failure conservative_cgrad exhibits: with the benchmark defaults
                # (alpha_d=2.0, epsilon=0.3, h=0.8) its ceiling is 0.646 < h and it can
                # never fire at all. The geometric mean keeps the same "all three gates
                # must agree" semantics while staying in [0,1] and remaining reachable,
                # so h reads as "the typical gate value required to intervene".
                intervention_signal = (g_m * g_d * g_cons).pow(1.0 / 3.0)

                # Only intervene past threshold
                do_intervene = (intervention_signal > intervention_threshold).float()

                # random.random() rather than torch.rand(): a host-side draw costs
                # nothing, whereas sampling on device would force a sync.
                if _CGRAD_STATS is not None and random.random() < _CGRAD_STATS_P:
                    _accumulate_cgrad_stats(do_intervene, intervention_signal, m,
                                            delta.abs() <= epsilon)

                # Corrective direction (Algorithm 1 line 11): sign(m*rho) repels u1 from
                # the threshold when m > 0 and attracts it when m < 0. Magnitude is tied
                # to |base_function| so the correction is equivariant by construction.
                soft_correction = torch.sign(m_grad) * base_function.abs()

                # Smooth blending (Algorithm 1 lines 6-9), then the eta-weighted
                # interpolation of line 12: g_cgrad = (1 - eta*b)*g_base + eta*b*g_corr
                blend_factor = do_intervene * torch.sigmoid(
                    gamma * (intervention_signal - intervention_threshold))
                w = eta * blend_factor

                dL_dU1 = (1 - w) * base_function + w * soft_correction

            elif mode == "":
                print("not ready")
                exit()

                
            elif mode == "TET":
                if reset_mode == 'soft':
                    dU2_dU1 = 1 - thresh * dS_dU1
                else:
                    dU2_dU1 = (1 - spikes_tensor[t]) - (u1 * dS_dU1)
                dL_dU1 = dL_dS * dS_dU1 + dL_dU2 * dU2_dU1
            else:
                # Default LIF
                if reset_mode == 'soft':
                    dU2_dU1 = 1 - thresh * dS_dU1
                else:
                    dU2_dU1 = (1 - spikes_tensor[t]) - (u1 * dS_dU1)
                dL_dU1 = dL_dS * dS_dU1 + dL_dU2 * dU2_dU1

            # Update for next iteration
            grad_memb_last = dL_dU1
            
            # Gradient w.r.t Input X[t]
            # Since mem[t] = mem[t-1]*decay + x[t]*input_scale
            # dL/dx[t] = dL/dMem[t] * input_scale
            grad_x[t] = grad_memb_last * input_scale

        return grad_x, None, None, None, None, None, None, None

## --- Native Drop-in Replacement ---
class LIFSpikeLayer_Cons(nn.Module):
    def __init__(self, thresh=1.0, tau=2.0, gama=1.0, detach_reset=True, args=None, **kwargs):
        super(LIFSpikeLayer_Cons, self).__init__()
        self.thresh = thresh
        self.detach_reset = detach_reset # <--- Capture this arg
        
        self.args = args if args is not None else GLOBAL_ARGS
        self.reset_mode = getattr(self.args, 'reset_mode', 'hard') if self.args else 'hard'
        
        if self.args and hasattr(self.args, 'snnbp_tau') and self.args.snnbp_tau is not None:
            self.decay = self.args.snnbp_tau
            self.input_scale = 1.0 - self.decay
        else:
            self.decay = 1.0 - (1.0 / tau)
            self.input_scale = 1.0 / tau
            
        self.gama = gama

    def forward(self, x):
        # Pass detach_reset to the function
        return TimeParallel_LIFSpike.apply(x, self.thresh, self.decay, self.input_scale, self.gama, self.args, self.detach_reset, self.reset_mode)
        
class MultiStepLIFNode(nn.Module):
    def __init__(self, thresh=1.0, tau=2.0, gama=1.0, detach_reset=True, args=None, **kwargs):
        super(MultiStepLIFNode, self).__init__()
        
        use_custom = True
        if GLOBAL_ARGS is not None and hasattr(GLOBAL_ARGS, 'use_custom_neuron'):
             use_custom = GLOBAL_ARGS.use_custom_neuron
        
        # Override detach_reset if set globally
        real_detach_reset = detach_reset
        if GLOBAL_ARGS is not None and hasattr(GLOBAL_ARGS, 'detach_reset') and GLOBAL_ARGS.detach_reset is not None:
            real_detach_reset = GLOBAL_ARGS.detach_reset
        
        # Check for v_threshold in kwargs (aliasing thresh)
        real_thresh = thresh
        if 'v_threshold' in kwargs:
            real_thresh = kwargs['v_threshold']
        
        self.impl = None
        if use_custom:
            self.impl = LIFSpikeLayer_Cons(thresh=real_thresh, tau=tau, gama=gama, detach_reset=real_detach_reset, args=args, **kwargs)
        else:
            oj_kwargs = kwargs.copy()
            # Ensure v_threshold is set for Original
            if 'v_threshold' not in oj_kwargs:
                oj_kwargs['v_threshold'] = real_thresh
            
            # Handle potential mismatching args if necessary (e.g. backend)
            # SpikingJelly usually behaves well with extra kwargs or has backend.
            
            self.impl = OriginalLIFNode(
                tau=tau, 
                detach_reset=real_detach_reset, 
                **oj_kwargs
            )
            
    def forward(self, x):
        return self.impl(x)

    def reset(self):
        if hasattr(self.impl, 'reset'):
            self.impl.reset()
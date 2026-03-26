import torch
import torch.nn as nn
import math
from spikingjelly.clock_driven.neuron import MultiStepLIFNode as OriginalLIFNode

# --- Global Args Handler ---
GLOBAL_ARGS = None
METRICS_COLLECTOR = None

def set_global_args(args):
    global GLOBAL_ARGS
    GLOBAL_ARGS = args

def set_metrics_collector(collector):
    global METRICS_COLLECTOR
    METRICS_COLLECTOR = collector

# --- Custom Autograd Function ---
class TimeParallel_LIFSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, thresh, decay, input_scale, gama, args, detach_reset):
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
            
            # Soft Reset
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
        
        return spikes_tensor

    @staticmethod
    def backward(ctx, grad_output):
        mem_before_spikes, spikes_tensor, x, gama = ctx.saved_tensors
        thresh = ctx.thresh
        decay = ctx.decay
        input_scale = ctx.input_scale
        args = ctx.args
        detach_reset = ctx.detach_reset
        
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
                
                delta = u1 - thresh
                
                # Compute correction terms
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


            else:
                # Default LIF
                dU2_dU1 = (1 - spikes_tensor[t]) - (u1 * dS_dU1)
                dL_dU1 = dL_dS * dS_dU1 + dL_dU2 * dU2_dU1

            # Update for next iteration
            grad_memb_last = dL_dU1
            
            # Gradient w.r.t Input X[t]
            # Since mem[t] = mem[t-1]*decay + x[t]*input_scale
            # dL/dx[t] = dL/dMem[t] * input_scale
            grad_x[t] = grad_memb_last * input_scale

        return grad_x, None, None, None, None, None, None

## --- Native Drop-in Replacement ---
class LIFSpikeLayer_Cons(nn.Module):
    def __init__(self, thresh=1.0, tau=2.0, gama=1.0, detach_reset=True, args=None, **kwargs):
        super(LIFSpikeLayer_Cons, self).__init__()
        self.thresh = thresh
        self.detach_reset = detach_reset # <--- Capture this arg
        
        self.args = args if args is not None else GLOBAL_ARGS
        
        if self.args and hasattr(self.args, 'snnbp_tau') and self.args.snnbp_tau is not None:
            self.decay = self.args.snnbp_tau
            self.input_scale = 1.0 - self.decay
        else:
            self.decay = 1.0 - (1.0 / tau)
            self.input_scale = 1.0 / tau
            
        self.gama = gama

    def forward(self, x):
        # Pass detach_reset to the function
        return TimeParallel_LIFSpike.apply(x, self.thresh, self.decay, self.input_scale, self.gama, self.args, self.detach_reset)
        
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
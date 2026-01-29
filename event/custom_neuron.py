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
        T, batch_size, *spatial_dims = x.shape
        device = x.device
        
        mem = torch.zeros(batch_size, *spatial_dims, device=device)
        mem_before_spikes = []
        spikes = []
        
        for t in range(T):
            mem_before_spike = mem * decay + x[t] * input_scale
            mem_before_spikes.append(mem_before_spike.clone())
            
            spike = (mem_before_spike > thresh).float()
            spikes.append(spike)
            
            mem = mem_before_spike * (1 - spike)
        
        mem_before_spikes = torch.stack(mem_before_spikes, dim=0)
        spikes_tensor = torch.stack(spikes, dim=0)
        
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
        
        grad_x = torch.zeros_like(grad_output)
        grad_memb_last = torch.zeros_like(mem_before_spikes[0]) 
        
        # Helper for Surrogate Gradient (dS/dU)
        def get_dS_dU1(u, thresh, gama, args):
            mode = getattr(args, 'dS_du', 'Gamma')
            if mode == "sigmoid":
                alpha = getattr(args, 'surrogate_alpha', 4.0) 
                sgax = (u - thresh) * alpha
                return (1.0 - torch.sigmoid(sgax)) * torch.sigmoid(sgax) * alpha
            # Default to Gamma (Triangular)
            return (1 / gama.item()**2) * (gama.item() - (u - thresh).abs()).clamp(min=0)

        # Backward Time Loop
        for t in reversed(range(x.shape[0])):
            u1 = mem_before_spikes[t]
            dL_dS = grad_output[t]
            dL_dU2 = grad_memb_last * decay
            
            dS_dU1 = get_dS_dU1(u1, thresh, gama, args)
            
            # --- C-GRAD LOGIC ---
            # Check for 'conservative_cgrad' or default to standard
            mode = getattr(args, 'du_du', 'standard')
            
            if detach_reset:
                dU2_dU1_standard = (1 - spikes_tensor[t])
            else:
                dU2_dU1_standard = (1 - spikes_tensor[t]) - (u1 * dS_dU1)

            if mode == "conservative_cgrad":
                """
                Conservative Custom Gradient (C-Grad)
                Intervenes only in high-confidence problematic cases and interpolates 
                toward base gradient to maintain training stability.
                """
                epsilon = getattr(args, 'snnbp_epsilon', 0.3)
                alpha = getattr(args, 'snnbp_alpha', 2.0)
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
                
                # Detect misalignment
                misalignment = -m_grad * base_function 
                
                # Gates
                g_m = torch.sigmoid(alpha * (m.abs() - 0.1))
                g_d = torch.sigmoid(beta * (epsilon - delta.abs()))
                g_misalign = torch.sigmoid(alpha * misalignment)
                
                # Intervention Logic
                intervention_signal = g_m * g_d * g_misalign
                do_intervene = (intervention_signal > intervention_threshold).float()
                
                correction_direction = torch.sign(m_grad)
                correction_magnitude = base_function.abs() * 0.5 
                soft_correction = correction_direction * correction_magnitude
                
                blend_factor = do_intervene * torch.sigmoid(5 * (intervention_signal - intervention_threshold))
                
                dL_dU1 = (1 - 0.5 * blend_factor) * base_function + 0.5 * blend_factor * soft_correction

            else:
                # Default LIF Backward (Standard BPTT)
                dL_dU1 = dL_dS * dS_dU1 + dL_dU2 * dU2_dU1_standard

            grad_memb_last = dL_dU1
            grad_x[t] = grad_memb_last * input_scale

        return grad_x, None, None, None, None, None, None

## --- Native Drop-in Replacement ---
class LIFSpikeLayer_Cons(nn.Module):
    def __init__(self, thresh=1.0, tau=2.0, gama=1.0, detach_reset=True, args=None, **kwargs):
        super(LIFSpikeLayer_Cons, self).__init__()
        self.thresh = thresh
        self.detach_reset = detach_reset
        self.args = args if args is not None else GLOBAL_ARGS
        
        if self.args and hasattr(self.args, 'snnbp_tau') and self.args.snnbp_tau is not None:
            self.decay = self.args.snnbp_tau
            self.input_scale = 1.0 - self.decay
        else:
            self.decay = 1.0 - (1.0 / tau)
            self.input_scale = 1.0 / tau
            
        self.gama = gama

    def forward(self, x):
        return TimeParallel_LIFSpike.apply(x, self.thresh, self.decay, self.input_scale, self.gama, self.args, self.detach_reset)
        
class MultiStepLIFNode(nn.Module):
    def __init__(self, thresh=1.0, tau=2.0, gama=1.0, detach_reset=True, args=None, **kwargs):
        super(MultiStepLIFNode, self).__init__()
        
        use_custom = True
        if GLOBAL_ARGS is not None and hasattr(GLOBAL_ARGS, 'use_custom_neuron'):
             use_custom = GLOBAL_ARGS.use_custom_neuron
        
        real_detach_reset = detach_reset
        if GLOBAL_ARGS is not None and hasattr(GLOBAL_ARGS, 'detach_reset') and GLOBAL_ARGS.detach_reset is not None:
            real_detach_reset = GLOBAL_ARGS.detach_reset
        
        real_thresh = thresh
        if 'v_threshold' in kwargs:
            real_thresh = kwargs['v_threshold']
        
        self.impl = None
        if use_custom:
            self.impl = LIFSpikeLayer_Cons(thresh=real_thresh, tau=tau, gama=gama, detach_reset=real_detach_reset, args=args, **kwargs)
        else:
            oj_kwargs = kwargs.copy()
            if 'v_threshold' not in oj_kwargs:
                oj_kwargs['v_threshold'] = real_thresh
            
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
import torch
import torch.nn as nn
import math

# --- Global Args Handler ---
GLOBAL_ARGS = None

def set_global_args(args):
    global GLOBAL_ARGS
    GLOBAL_ARGS = args

# --- Custom Autograd Function ---
class TimeParallel_LIFSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, thresh, decay, input_scale, gama, args):
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
            mems_after_spikes.append(mem)
        
        # Stack along dimension 0 (Time)
        mem_before_spikes = torch.stack(mem_before_spikes, dim=0)
        spikes_tensor = torch.stack(spikes, dim=0)
        mems_after_spikes = torch.stack(mems_after_spikes, dim=0)
        
        # Save context
        ctx.save_for_backward(mem_before_spikes, spikes_tensor, mems_after_spikes, x, torch.tensor([gama]))
        ctx.thresh = thresh
        ctx.decay = decay
        ctx.input_scale = input_scale
        ctx.args = args
        
        return spikes_tensor

    @staticmethod
    def backward(ctx, grad_output):
        mem_before_spikes, spikes_tensor, mems_after_spikes, x, gama = ctx.saved_tensors
        thresh = ctx.thresh
        decay = ctx.decay
        input_scale = ctx.input_scale
        args = ctx.args
        
        # Output Gradients
        grad_x = torch.zeros_like(grad_output)
        
        # Gradient accumulator for the next timestep (starts at 0 for last step)
        grad_memb_last = torch.zeros_like(mem_before_spikes[0]) 
        
        # Helper for Surrogate Gradient (dS/dU)
        def get_dS_dU1(u, thresh, gama, args):
            mode = getattr(args, 'dS_du', 'Gamma')
            if mode == "Gamma":
                return (1 / gama.item()**2) * (gama.item() - (u - thresh).abs()).clamp(min=0)
            elif mode == "sigmoid":
                s = torch.sigmoid(u - thresh)
                return s * (1 - s)
            return (1 / gama.item()**2) * (gama.item() - (u - thresh).abs()).clamp(min=0)

        # Backward Time Loop (Iterate T from end to start)
        for t in reversed(range(x.shape[0])):
            dL_dS = grad_output[t]        # Gradient from loss w.r.t Spike[t]
            dL_dU2 = grad_memb_last * decay # Propagate via decay factor
            
            dS_dU1 = get_dS_dU1(mem_before_spikes[t], thresh, gama, args)
            
            # --- CUSTOM GRADIENT LOGIC ---
            mode = getattr(args, 'du_du', 'complex54')
            
            if mode == "complex54":
                epsilon = getattr(args, 'snnbp_epsilon', 0.1)
                alpha = getattr(args, 'snnbp_alpha', 1.0)
                beta = getattr(args, 'snnbp_beta', 1.0)
                p = getattr(args, 'snnbp_p', 1.0)

                u1 = mem_before_spikes[t]
                
                term_supra_threshold = (thresh * dL_dU2) - dL_dS
                term_sub_threshold = dL_dS - (u1 * dL_dU2)
                
                m = torch.where(u1 < thresh, term_sub_threshold, term_supra_threshold)
                m_grad = torch.where(u1 < thresh, m, -m)

                # Base Function (Standard BPTT)
                dU2_dU1_standard = (1 - spikes_tensor[t]) - (u1 * dS_dU1)
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
                epsilon = getattr(args, 'snnbp_epsilon', 0.1)
                alpha = getattr(args, 'snnbp_alpha', 1.0)
                beta = getattr(args, 'snnbp_beta', 1.0)
                p = getattr(args, 'snnbp_p', 4.0)

                u1 = mem_before_spikes[t]
                
                term_supra_threshold = (thresh * dL_dU2) - dL_dS
                term_sub_threshold = dL_dS - (u1 * dL_dU2)
                
                m = torch.where(u1 < thresh, term_sub_threshold, term_supra_threshold)
                m_grad = torch.where(u1 < thresh, m, -m)

                # Base Function (Standard BPTT)
                dU2_dU1_standard = (1 - spikes_tensor[t]) - (u1 * dS_dU1)
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

                
            elif mode == "TET":
                dU2_dU1 = (1 - spikes_tensor[t]) - (mem_before_spikes[t] * dS_dU1)
                dL_dU1 = dL_dS * dS_dU1 + dL_dU2 * dU2_dU1
            else:
                # Default LIF
                dU2_dU1 = (1 - spikes_tensor[t]) - (mem_before_spikes[t] * dS_dU1)
                dL_dU1 = dL_dS * dS_dU1 + dL_dU2 * dU2_dU1

            # Update for next iteration
            grad_memb_last = dL_dU1
            
            # Gradient w.r.t Input X[t]
            # Since mem[t] = mem[t-1]*decay + x[t]*input_scale
            # dL/dx[t] = dL/dMem[t] * input_scale
            grad_x[t] = grad_memb_last * input_scale

        return grad_x, None, None, None, None, None

# --- Native Drop-in Replacement ---
class LIFSpikeLayer_Cons(nn.Module):
    def __init__(self, thresh=1.0, tau=2.0, gama=1.0, args=None, **kwargs):
        super(LIFSpikeLayer_Cons, self).__init__()
        self.thresh = thresh
        
        # 1. Grab Args
        self.args = args if args is not None else GLOBAL_ARGS
        
        # 2. Determine Decay Factor
        # If user overrides via args (e.g., --snnbp-tau 0.5), use that directly as decay
        # Otherwise, calculate decay from tau (time constant)
        if self.args and hasattr(self.args, 'snnbp_tau') and self.args.snnbp_tau is not None:
            print("valid snnbp tau value", self.args.snnbp_tau)
            # Assume user provided explicit decay factor
            self.decay = self.args.snnbp_tau
            self.input_scale = 1.0 - self.decay # Input scale is 1 - decay
        else:
            # Standard SpikingJelly behavior: decay = 1 - 1/tau
            # Input is scaled by 1/tau
            self.decay = 1.0 - (1.0 / tau)
            self.input_scale = 1.0 / tau
        
        print(self.args.du_du)
        self.gama = gama

    def forward(self, x):
        # x is [Time, Batch, ...]
        return TimeParallel_LIFSpike.apply(x, self.thresh, self.decay, self.input_scale, self.gama, self.args)
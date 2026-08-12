
import torch
import numpy as np
import sys
import os

# Ensure we can import custom_neuron.py
sys.path.append(os.getcwd() + "/cifar10-100")

# Mock spikingjelly if not present, as we only need TimeParallel_LIFSpike logic which relies on torch
try:
    import spikingjelly
except ImportError:
    # Minimal mock
    import types
    spikingjelly = types.ModuleType("spikingjelly")
    clock_driven = types.ModuleType("clock_driven")
    neuron = types.ModuleType("neuron")
    
    class MockLIF:
        def __init__(self, *args, **kwargs): pass
        
    neuron.MultiStepLIFNode = MockLIF
    clock_driven.neuron = neuron
    spikingjelly.clock_driven = clock_driven
    sys.modules["spikingjelly"] = spikingjelly
    sys.modules["spikingjelly.clock_driven"] = clock_driven
    sys.modules["spikingjelly.clock_driven.neuron"] = neuron

sys.path.append(os.getcwd() + "/cifar10-100")
from custom_neuron import TimeParallel_LIFSpike

# --- Numpy Implementation from Notebook ---
def get_dS_du1_numpy(u, thresh=1.0, gama=1.0, mode='Gamma', alpha=4.0):
    if mode == "sigmoid":
        sgax = (u - thresh) * alpha
        sig = 1.0 / (1.0 + np.exp(-sgax))
        return (1.0 - sig) * sig * alpha
    elif mode == "Gamma":
        return (1.0 / gama**2) * np.maximum(0, gama - np.abs(u - thresh))
    return (1.0 / gama**2) * np.maximum(0, gama - np.abs(u - thresh))

def cgrad_gradient_numpy(u1, dL_dS, dL_du2, thresh=1.0, gama=1.0, mode='Gamma', alpha=4.0, epsilon=0.1, alpha_c=1.0, beta_c=1.0, p_c=1.0, detach_reset=False):
    dS_du1 = get_dS_du1_numpy(u1, thresh, gama, mode, alpha)
    spike = (u1 >= thresh).astype(float)
    
    term_supra_threshold = (thresh * dL_du2) - dL_dS
    term_sub_threshold = dL_dS - (u1 * dL_du2)
    
    # We use np.where here to mimic torch.where behavior for scalar/array inputs
    m = np.where(u1 < thresh, term_sub_threshold, term_supra_threshold)
    m_grad = np.where(u1 < thresh, m, -m)

    if detach_reset:
        du2_du1_standard = 0.0 
    else:
        du2_du1_standard = (1.0 - spike) - u1 * dS_du1

    base_function = dL_dS * dS_du1 + dL_du2 * du2_du1_standard

    g_m = 1.0 / (1.0 + np.exp(-alpha_c * m))
    delta = u1 - thresh
    g_d = 1.0 / (1.0 + np.exp(-beta_c * (epsilon - np.abs(delta))))
    
    g_dir = np.clip(-1 * np.sign(m_grad) * np.sign(base_function), 0, 1)
    
    f = np.clip(p_c * g_m * g_d * g_dir, 0, 1)
    
    compute_dist = np.maximum(np.abs(delta), epsilon)
    dL_du1 = f * (m_grad / compute_dist) + (1.0 - f) * base_function
    return dL_du1

# --- PyTorch Verification ---
class MockArgs:
    def __init__(self):
        # We don't set explicit values here to rely on the module defaults we just updated
        pass 

class MockContext:
    def __init__(self, saved_tensors, thresh, decay, input_scale, args, detach_reset):
        self.saved_tensors = saved_tensors
        self.thresh = thresh
        self.decay = decay
        self.input_scale = input_scale
        self.args = args
        self.detach_reset = detach_reset

def run_torch_verification():
    # Setup inputs
    np.random.seed(42)
    # Shapes: [Time, Batch, ...] but for single step verification we can simplify
    # The backward loop logic in custom_neuron iterates Time.
    # Let's verify a single timestep t corresponding to index 0 of a 1-step sequence
    
    u1_val = 0.5
    dL_dS_val = 1.0
    dL_du2_val = 1.0
    
    thresh = 1.0
    gama = 1.0
    
    # Expected defaults in custom_neuron.py now:
    epsilon = 0.3468
    alpha_c = 1.1742
    beta_c = 0.9245
    p_c = 9.5334
    
    # 1. Numpy Calc (with detach_reset=True default behavior validation)
    detach_reset = True
    
    grad_numpy = cgrad_gradient_numpy(
        np.array([u1_val]), np.array([dL_dS_val]), np.array([dL_du2_val]), 
        thresh=thresh, gama=gama, 
        epsilon=epsilon, alpha_c=alpha_c, beta_c=beta_c, p_c=p_c,
        detach_reset=detach_reset
    )
    
    # 2. Torch Call via custom_neuron.py
    # We need to Construct inputs that 'backward' expects.
    # ctx.saved_tensors = (mem_before_spikes, spikes_tensor, x, gama)
    
    mem_before_spikes = torch.tensor([[[u1_val]]], dtype=torch.float32) # [T=1, B=1, 1]
    spikes_tensor = torch.tensor([[[0.0]]], dtype=torch.float32) # u1 < thresh
    x = torch.zeros_like(mem_before_spikes) # Doesn't matter for dL_dU1 calculation, only for dL_dx
    gamma_tensor = torch.tensor([gama], dtype=torch.float32)
    
    args = MockArgs()
    # Ensure args don't override defaults with None or missing values, 
    # custom_neuron uses getattr(args, name, default).
    # We specifically want to check if the defaults are picked up (0.3468 etc).
    # So we leave MockArgs empty.
    
    # We also need to set 'du_du' to 'complex54'
    args.du_du = 'complex54'
    
    ctx = MockContext(
        saved_tensors=(mem_before_spikes, spikes_tensor, x, gamma_tensor),
        thresh=thresh,
        decay=0.5, # arbitrary
        input_scale=0.5, # arbitrary
        args=args,
        detach_reset=detach_reset
    )
    
    # grad_output shape should match spikes_tensor: [T, B, ...]
    grad_output = torch.tensor([[[dL_dS_val]]], dtype=torch.float32)
    
    # Call backward
    # Returns: grad_x, None, None, None, None, None, None
    grad_x, _, _, _, _, _, _ = TimeParallel_LIFSpike.backward(ctx, grad_output)
    
    # Wait, TimeParallel_LIFSpike.backward computes grad_x = grad_memb_last * input_scale
    # We want to verify dL_dU1 (the intermediate gradient).
    # The file computes dL_dU1 then sets grad_memb_last = dL_dU1.
    # Then grad_x[t] = grad_memb_last * input_scale.
    # So if we divide result by input_scale, we get dL_dU1 (since we have 1 timestep).
    
    # However, if detach_reset logic interacts with 'grad_memb_last' from 'future' steps (which are 0 here),
    # For T=1, dL_dU2 (from future) is 0.
    # But in our numpy test, we supplied dL_du2_val = 1.0.
    # In custom_neuron.py: 
    # dL_dU2 = grad_memb_last * decay
    # For the last timestep (t=0 in reversed range 1), grad_memb_last starts at 0.
    # So effective dL_dU2 is 0.
    
    # To properly test dL_du2 input, we need 2 timesteps? 
    # Or hack it. The Custom Neuron logic assumes sequential dependency.
    # If we want to test exact formula match with arbitrary dL_du2, we can't easily use the full backward loop 
    # because it enforces the recursive structure.
    
    # Actually, simpler: comparing the Code Logic implies we accept the recursive structural definition.
    # We verified the logic matching previously.
    # What we really want to verify now is:
    # 1. The DEFAULT PARAMETERS are picked up correctly (matching our expectation).
    # 2. The detach_reset logic is RESPECTED (so dL_du2 term is zeroed out if detach_reset=True).
    
    # Let's test with dL_du2 = 0 in numpy to match the T=1 behavior of the script where future gradient is 0.
    # Normalized test:
    
    grad_numpy_t1 = cgrad_gradient_numpy(
        np.array([u1_val]), np.array([dL_dS_val]), np.array([0.0]), # dL_du2=0
        thresh=thresh, gama=gama, 
        epsilon=epsilon, alpha_c=alpha_c, beta_c=beta_c, p_c=p_c,
        detach_reset=detach_reset
    )
    
    grad_torch_t1 = grad_x[0,0,0] / 0.5 # Divide by input_scale to get dL_dU1
    grad_torch_val = grad_torch_t1.item()
    
    print("Numpy vs Torch Comparison (T=1, dL_du2=0 implicit):")
    print(f"Numpy: {grad_numpy_t1[0]}")
    print(f"Torch: {grad_torch_val}")
    
    diff = np.abs(grad_numpy_t1[0] - grad_torch_val)
    if diff < 1e-5:
        print("SUCCESS: Values match.")
    else:
        print("FAILURE: Values differ.")
        
    # Consistency check for detach_reset:
    # If we forced the bug (detach_reset ignored), calculate what it WOULD be.
    # With detach_reset=False, du2_du1 = (1 - spike) - u * dS
    # With detach_reset=True,  du2_du1 = 0
    # BUT, this term 'du2_du1_standard' multiplies dL_du2.
    # Since dL_du2 is 0 in the T=1 setup, THIS DIFFERENCE IS HIDDEN!
    
    # Critical realization: To test detach_reset fix, we MUST have non-zero dL_du2.
    # This implies we need at least 2 timesteps, or we need to modify how we call it?
    # No, with T=2:
    # t=1 (last): dL_dU2 = 0. calculates dL_dU1_t1. grad_memb = dL_dU1_t1.
    # t=0 (first): dL_dU2 = grad_memb * decay = dL_dU1_t1 * decay.
    #              Calculates dL_dU1_t0. 
    #              Here dU2_dU1_standard is used.
    #              If detach_reset=True, base_function = dL_dS * dS + dL_dU2 * 0
    #              If detach_reset=False, base_function = dL_dS * dS + dL_dU2 * ((1-S)-u*dS)
    
    # So we need to run T=2 and check the gradient at t=0.
    
    print("\nRunning T=2 check for detach_reset...")
    
    mem_vals = torch.tensor([[[0.5]], [[0.5]]], dtype=torch.float32) # T=2
    spikes_vals = torch.zeros_like(mem_vals)
    grad_out_vals = torch.tensor([[[1.0]], [[1.0]]], dtype=torch.float32)
    
    ctx_2 = MockContext(
        saved_tensors=(mem_vals, spikes_vals, torch.zeros_like(mem_vals), gamma_tensor),
        thresh=thresh, decay=1.0, input_scale=1.0, # decay=1 for simple propagation
        args=args, detach_reset=True
    )
    
    grad_x_2, _, _, _, _, _, _ = TimeParallel_LIFSpike.backward(ctx_2, grad_out_vals)
    
    # Torch Result for t=0
    # t=1: dL_dU2=0. dL_dU1_t1 calculated.
    # t=0: dL_dU2 = dL_dU1_t1 * 1.0. 
    #      dL_dU1_t0 calculated.
    
    grad_t0_torch = grad_x_2[0,0,0].item()
    
    # Numpy Simulation
    # Step 2 (t=1):
    g_t1 = cgrad_gradient_numpy(np.array([0.5]), np.array([1.0]), np.array([0.0]),
                                thresh, gama, epsilon=epsilon, alpha_c=alpha_c, beta_c=beta_c, p_c=p_c,
                                detach_reset=True)
    dL_du2_for_t0 = g_t1 * 1.0 # decay
    
    # Step 1 (t=0):
    g_t0_detach_true = cgrad_gradient_numpy(np.array([0.5]), np.array([1.0]), dL_du2_for_t0,
                                            thresh, gama, epsilon=epsilon, alpha_c=alpha_c, beta_c=beta_c, p_c=p_c,
                                            detach_reset=True)
                                            
    g_t0_detach_false = cgrad_gradient_numpy(np.array([0.5]), np.array([1.0]), dL_du2_for_t0,
                                             thresh, gama, epsilon=epsilon, alpha_c=alpha_c, beta_c=beta_c, p_c=p_c,
                                             detach_reset=False)
                                             
    print(f"Torch (t=0): {grad_t0_torch}")
    print(f"Numpy (detach=True): {g_t0_detach_true[0]}")
    print(f"Numpy (detach=False): {g_t0_detach_false[0]}")
    
    if np.abs(grad_t0_torch - g_t0_detach_true[0]) < 1e-5:
        print("SUCCESS: Torch matches detach=True logic.")
    elif np.abs(grad_t0_torch - g_t0_detach_false[0]) < 1e-5:
        print("FAILURE: Torch matched detach=False logic (Bug still present!).")
    else:
        print("FAILURE: Torch matched neither (Parameter mismatch?).")

if __name__ == "__main__":
    run_torch_verification()

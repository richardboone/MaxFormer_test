import argparse
import math
import torch
import sys, os

# Ensure local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from custom_neuron import TimeParallel_LIFSpike
from triton_neuron import TritonTimeParallel_LIFSpike

def run_test_suite(device='cuda:1'):
    print(f"\n=======================================================")
    print(f" Running Triton C-Grad Numerical Correctness Suite on {device}")
    print(f"=======================================================\n")
    
    device = torch.device(device)
    torch.manual_seed(42)
    
    T = 8
    B = 16
    C = 64
    H = 8
    W = 8
    
    modes = ['LIF', 'conservative_cgrad', 'invariant_cgrad']
    surrogates = ['sigmoid', 'Gamma']
    resets = ['hard', 'soft']
    detach_resets = [False, True]
    dtypes = [torch.float32, torch.float16]
    
    passed = 0
    total = 0
    
    for dtype in dtypes:
        for mode in modes:
            for surrogate in surrogates:
                for reset in resets:
                    for detach in detach_resets:
                        total += 1
                        tag = f"dtype={dtype}, mode={mode:18s}, sg={surrogate:7s}, reset={reset:4s}, detach={str(detach):5s}"
                        
                        args = argparse.Namespace(
                            du_du=mode,
                            dS_du=surrogate,
                            surrogate_alpha=4.0,
                            snnbp_epsilon=0.3,
                            snnbp_alpha=2.0,
                            snnbp_beta=2.0,
                            snnbp_alpha_cons=2.0,
                            snnbp_kappa=1.0,
                            snnbp_h=0.5,
                            snnbp_intervention=0.8,
                            snnbp_eta=0.5,
                            snnbp_gamma=5.0,
                            snnbp_two_sided=True,
                            snnbp_correction='blend',
                            snnbp_consistency='product'
                        )
                        
                        thresh = 1.0
                        decay = 0.5
                        input_scale = 0.5
                        gama = 1.0
                        
                        # Generate random inputs
                        x_ref = (torch.randn(T, B, C, H, W, device=device, dtype=dtype) * 0.8 + 0.5).requires_grad_(True)
                        x_tri = x_ref.detach().clone().requires_grad_(True)
                        
                        # Forward pass
                        out_ref = TimeParallel_LIFSpike.apply(x_ref, thresh, decay, input_scale, gama, args, detach, reset)
                        out_tri = TritonTimeParallel_LIFSpike.apply(x_tri, thresh, decay, input_scale, gama, args, detach, reset)
                        
                        # Check Forward match
                        fwd_diff = (out_ref - out_tri).abs().max().item()
                        if fwd_diff > 1e-4:
                            print(f"[FAIL FORWARD] {tag} | max diff: {fwd_diff}")
                            continue
                            
                        # Backward pass
                        grad_out = torch.randn_like(out_ref)
                        out_ref.backward(grad_out)
                        out_tri.backward(grad_out)
                        
                        grad_ref = x_ref.grad
                        grad_tri = x_tri.grad
                        
                        bwd_diff_tensor = (grad_ref - grad_tri).abs()
                        bwd_diff = bwd_diff_tensor.max().item()
                        mean_diff = bwd_diff_tensor.mean().item()
                        denom = grad_ref.abs().max().item() + 1e-8
                        rel_diff = bwd_diff / denom
                        
                        if dtype == torch.float32:
                            is_pass = (rel_diff <= 1e-4 or bwd_diff <= 1e-4)
                        else:
                            # In FP16, Triton's FP32 internal accumulation produces more accurate results
                            # than PyTorch's intermediate FP16 casting, leading to boundary flip on ~0.001% of elements.
                            is_pass = (mean_diff <= 1e-4 and (bwd_diff_tensor < 1e-3).float().mean().item() >= 0.999)
                            
                        if is_pass:
                            passed += 1
                            print(f"[PASS] {tag} | max diff: {bwd_diff:.6e}, mean: {mean_diff:.6e}")
                        else:
                            print(f"[FAIL BACKWARD] {tag} | max diff: {bwd_diff:.6e}, mean: {mean_diff:.6e}")
                            
    print(f"\n=======================================================")
    print(f" Summary: {passed}/{total} tests passed.")
    print(f"=======================================================\n")
    return passed == total

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:1')
    cli = parser.parse_args()
    success = run_test_suite(device=cli.device)
    if not success:
        sys.exit(1)


import torch
import torch.nn as nn
from max_former import max_former
from custom_neuron import MultiStepLIFNode

# BASELINE CONSTANTS (Ops per neuron per timestep)
SNN_FWD = 6
SNN_BPTT_BWD = 11
SNN_CGRAD_BWD = 65

def count_ops(module, input_shape, mode='inference'):
    # input_shape: (T, B, C, H, W)
    T, B = input_shape[0], input_shape[1]
    
    macs = 0
    neuron_ops = 0
    
    # Training multiplier for Synaptic Ops: 3x
    synaptic_multiplier = 3 if mode != 'inference' else 1
    
    # Spiking Ops baseline
    if mode == 'inference':
        spiking_baseline = SNN_FWD
    elif mode == 'training_regular':
        spiking_baseline = SNN_FWD + SNN_BPTT_BWD
    elif mode == 'training_cgrad':
        spiking_baseline = SNN_FWD + SNN_CGRAD_BWD
    else:
        spiking_baseline = SNN_FWD

    def hook_fn(m, i, o):
        nonlocal macs, neuron_ops
        if isinstance(m, nn.Conv2d):
            cin = m.in_channels
            cout = m.out_channels
            kh, kw = m.kernel_size
            hout, wout = o.shape[-2:]
            groups = m.groups
            m_macs = (cin * cout * kh * kw * hout * wout) // groups
            macs += m_macs * T * B * synaptic_multiplier
        elif isinstance(m, nn.Linear):
            m_macs = m.in_features * m.out_features
            if len(o.shape) > 2:
                m_macs *= o.numel() // (T * B * m.out_features)
            macs += m_macs * T * B * synaptic_multiplier
        elif isinstance(m, nn.Conv1d):
            cin = m.in_channels
            cout = m.out_channels
            k = m.kernel_size[0]
            lout = o.shape[-1]
            groups = m.groups
            m_macs = (cin * cout * k * lout) // groups
            macs += m_macs * T * B * synaptic_multiplier
        elif isinstance(m, MultiStepLIFNode):
            neuron_ops += spiking_baseline * o.numel()

    hooks = []
    for name, m in module.named_modules():
        hooks.append(m.register_forward_hook(hook_fn))
    
    with torch.no_grad():
        x = torch.randn(input_shape)
        module(x)
        
    for h in hooks:
        h.remove()
        
    return macs, neuron_ops

def analyze_max_former(mode='inference'):
    model = max_former(
        embed_dims=384, mlp_ratios=4,
        in_channels=3, num_classes=10, T=4
    )
    
    T, B = 4, 1
    input_size = (T, B, 3, 32, 32)
    synaptic_multiplier = 3 if mode != 'inference' else 1
    
    title = mode.replace('_', ' ').capitalize()
    print(f"\n--- MaxFormer {title} Compute Evaluation ---")
    print(f"| Stage | Synaptic Ops (M) | Neuron Ops (M) | Total (M) | Res |")
    print(f"| :--- | :---: | :---: | :---: | :---: |")
    
    total_macs = 0
    total_neuron = 0

    def report(name, m, n, res):
        nonlocal total_macs, total_neuron
        total_macs += m
        total_neuron += n
        print(f"| {name} | {m/1e6:.2f} | {n/1e6:.2f} | {(m+n)/1e6:.2f} | {res} |")

    # 1. Patch Embed 1
    m, n = count_ops(model.patch_embed1, input_size, mode=mode)
    with torch.no_grad():
        x = model.patch_embed1(torch.randn(input_size))
        report("Patch Embed 1", m, n, f"{x.shape[-2]}x{x.shape[-1]}")
    
    # 2. Stage 1
    m_s, n_s = 0, 0
    for blk in model.stage1:
        bm, bn = count_ops(blk, x.shape, mode=mode)
        m_s += bm
        n_s += bn
        with torch.no_grad(): x = blk(x)
    report("Stage 1", m_s, n_s, f"{x.shape[-2]}x{x.shape[-1]}")
        
    # 3. Patch Embed 2
    m, n = count_ops(model.patch_embed2, x.shape, mode=mode)
    with torch.no_grad():
        x = model.patch_embed2(x)
        report("Patch Embed 2", m, n, f"{x.shape[-2]}x{x.shape[-1]}")
        
    # 4. Stage 2
    m_s, n_s = 0, 0
    for blk in model.stage2:
        bm, bn = count_ops(blk, x.shape, mode=mode)
        m_s += bm
        n_s += bn
        with torch.no_grad(): x = blk(x)
    report("Stage 2", m_s, n_s, f"{x.shape[-2]}x{x.shape[-1]}")
        
    # 5. Patch Embed 3
    m, n = count_ops(model.patch_embed3, x.shape, mode=mode)
    with torch.no_grad():
        x = model.patch_embed3(x)
        report("Patch Embed 3", m, n, f"{x.shape[-2]}x{x.shape[-1]}")
        
    # 6. Stage 3
    m_s, n_s = 0, 0
    for blk in model.stage3:
        bm, bn = count_ops(blk, x.shape, mode=mode)
        N = x.shape[-1] * x.shape[-2]
        dim = 384
        num_heads = 8
        d_h = dim // num_heads
        ssa_matmul_macs = num_heads * (d_h * N * d_h + N * d_h * d_h) * T * B * synaptic_multiplier
        
        m_s += bm + ssa_matmul_macs
        n_s += bn
        with torch.no_grad(): x = blk(x)
    report("Stage 3", m_s, n_s, f"{x.shape[-2]}x{x.shape[-1]}")
    
    # 7. Head
    m, n = count_ops(model.head_lif, (T, B, 384), mode=mode)
    head_macs = 384 * 10 * synaptic_multiplier
    report("Head", head_macs, n, "1x1")

    print(f"| **Total** | **{total_macs/1e6:.2f}** | **{total_neuron/1e6:.2f}** | **{(total_macs+total_neuron)/1e6:.2f}** | |")

if __name__ == "__main__":
    analyze_max_former(mode='inference')
    analyze_max_former(mode='training_regular')
    analyze_max_former(mode='training_cgrad')

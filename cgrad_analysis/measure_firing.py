"""Measure how often C-Grad's intervention gate actually fires inside a real
Max-Former training step, using the exact configs the paper reports."""
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import argparse, os, sys, math
import torch, torch.nn as nn

REPO = _os.path.join(_REPO, "cifar10-100")
sys.path.insert(0, REPO)
os.chdir(REPO)

import custom_neuron
from custom_neuron import TimeParallel_LIFSpike

STATS = {}


def patched_backward(ctx, grad_output):
    """Re-derives the conservative_cgrad gate and records statistics, then
    delegates to the original backward so training math is untouched."""
    mem_before_spikes, spikes_tensor, x, gama = ctx.saved_tensors
    thresh, decay, args = ctx.thresh, ctx.decay, ctx.args
    eps = getattr(args, 'snnbp_epsilon', 0.3)
    al = getattr(args, 'snnbp_alpha', 2.0)
    be = getattr(args, 'snnbp_beta', 2.0)
    h = getattr(args, 'snnbp_intervention', 0.8)

    gml = torch.zeros_like(mem_before_spikes[0])
    for t in reversed(range(x.shape[0])):
        u1, s = mem_before_spikes[t], spikes_tensor[t]
        dL_dS, dL_dU2 = grad_output[t], gml * decay
        dS = (1 / gama.item() ** 2) * (gama.item() - (u1 - thresh).abs()).clamp(min=0)
        dU2 = (1 - s) if ctx.detach_reset else (1 - s) - u1 * dS
        m = torch.where(u1 < thresh, dL_dS - u1 * dL_dU2, thresh * dL_dU2 - dL_dS)
        mg = torch.where(u1 < thresh, m, -m)
        bf = dL_dS * dS + dL_dU2 * dU2
        d = u1 - thresh
        g_m = torch.sigmoid(al * (m.abs() - 0.1))
        g_d = torch.sigmoid(be * (eps - d.abs()))
        g_mis = torch.sigmoid(al * (-mg * bf))
        sigl = g_m * g_d * g_mis
        do = sigl > h
        STATS['n'] = STATS.get('n', 0) + do.numel()
        STATS['fire'] = STATS.get('fire', 0) + do.sum().item()
        STATS['near'] = STATS.get('near', 0) + (d.abs() <= eps).sum().item()
        STATS['m_abs_max'] = max(STATS.get('m_abs_max', 0.0), m.abs().max().item())
        STATS['m_abs_p9999'] = max(STATS.get('m_abs_p9999', 0.0),
                                   torch.quantile(m.abs().flatten().float()[:1_000_000], 0.9999).item())
        STATS['sig_max'] = max(STATS.get('sig_max', 0.0), sigl.max().item())
        blend = do.float() * torch.sigmoid(5 * (sigl - h))
        gml = (1 - 0.5 * blend) * bf + 0.5 * blend * torch.sign(mg) * bf.abs()
    return ORIG(ctx, grad_output)


ORIG = TimeParallel_LIFSpike.backward
TimeParallel_LIFSpike.backward = staticmethod(patched_backward)

import yaml
from timm.models import create_model
import max_former  # noqa: registers max_former


def run(cfg_path, tag, steps=3, bs=16, T=4):
    STATS.clear()
    cfg = yaml.safe_load(open(cfg_path))
    cfg = {k.replace('-', '_'): v for k, v in cfg.items()}
    cfg.setdefault('dS_du', 'Gamma')
    args = argparse.Namespace(**cfg)
    args.use_custom_neuron = True
    custom_neuron.set_global_args(args)

    model = create_model('max_former', num_classes=cfg.get('num_classes', 10),
                         T=T, in_channels=3, embed_dims=384, mlp_ratios=4, depths=4,
                         pretrained=False).cuda()
    x = torch.randn(bs, 3, 32, 32).cuda()
    y = torch.randint(0, cfg.get('num_classes', 10), (bs,)).cuda()
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(steps):
        opt.zero_grad()
        crit(model(x), y).backward()
        opt.step()

    n, f = STATS['n'], STATS['fire']
    print(f"\n### {tag}   ({os.path.basename(cfg_path)})")
    print(f"  du_du={cfg.get('du_du')}  alpha_m={cfg.get('snnbp_alpha')}  "
          f"alpha_d={cfg.get('snnbp_beta')}  eps={cfg.get('snnbp_epsilon')}  "
          f"h={cfg.get('snnbp_intervention')}")
    print(f"  neuron-timesteps observed : {n:,}")
    print(f"  within eps of threshold   : {STATS['near']:,} "
          f"({100*STATS['near']/n:.2f}%)")
    print(f"  INTERVENTIONS FIRED       : {f:,} ({100*f/n:.6f}%)")
    print(f"  max |m| seen              : {STATS['m_abs_max']:.3e}")
    print(f"  max intervention_signal   : {STATS['sig_max']:.4f}   (needs > h={cfg.get('snnbp_intervention')})")


if __name__ == '__main__':
    torch.manual_seed(0)
    run(f"{REPO}/cifar10_best_cons.yaml", "CIFAR-10 / CIFAR-100 / ImageNet C-Grad config")
    run(_os.path.join(_REPO, "event/cifar10dvs_cgrad.yaml"),
        "CIFAR10-DVS C-Grad config (the 84.2% run)")
    run(_os.path.join(_REPO, "event/cifar10dvs_hp_ablation.yaml"),
        "Paper Table 3 default hyperparameters")

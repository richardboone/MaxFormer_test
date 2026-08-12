"""Same measurement as measure_firing.py, but through the real AMP +
GradScaler path that every config in the repo actually uses (amp: True)."""
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import argparse, os, sys
import torch, torch.nn as nn

REPO = _os.path.join(_REPO, "cifar10-100")
sys.path.insert(0, REPO)
os.chdir(REPO)

import custom_neuron
from custom_neuron import TimeParallel_LIFSpike

STATS = {}


def patched_backward(ctx, grad_output):
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
        near = d.abs() <= eps
        STATS['n'] = STATS.get('n', 0) + do.numel()
        STATS['fire'] = STATS.get('fire', 0) + do.sum().item()
        STATS['near'] = STATS.get('near', 0) + near.sum().item()
        STATS['m_neg_fire'] = STATS.get('m_neg_fire', 0) + (do & (m < 0)).sum().item()
        STATS['gm_sat'] = STATS.get('gm_sat', 0) + (g_m > 0.99).sum().item()
        STATS['gmis_bin'] = STATS.get('gmis_bin', 0) + ((g_mis > 0.99) | (g_mis < 0.01)).sum().item()
        STATS['m_max'] = max(STATS.get('m_max', 0.0), m.abs().max().float().item())
        STATS['sig_max'] = max(STATS.get('sig_max', 0.0), sigl.max().float().item())
        STATS['nan'] = STATS.get('nan', 0) + (~torch.isfinite(sigl)).sum().item()
        blend = do.float() * torch.sigmoid(5 * (sigl - h))
        gml = (1 - 0.5 * blend) * bf + 0.5 * blend * torch.sign(mg) * bf.abs()
    return ORIG(ctx, grad_output)


ORIG = TimeParallel_LIFSpike.backward
TimeParallel_LIFSpike.backward = staticmethod(patched_backward)

import yaml
from timm.models import create_model
import max_former  # noqa


def run(cfg_path, tag, steps=4, bs=16, T=4):
    STATS.clear()
    cfg = {k.replace('-', '_'): v for k, v in yaml.safe_load(open(cfg_path)).items()}
    cfg.setdefault('dS_du', 'Gamma')
    args = argparse.Namespace(**cfg)
    args.use_custom_neuron = True
    custom_neuron.set_global_args(args)

    model = create_model('max_former', num_classes=cfg.get('num_classes', 10), T=T,
                         in_channels=3, embed_dims=384, mlp_ratios=4, depths=4).cuda()
    x = torch.randn(bs, 3, 32, 32).cuda()
    y = torch.randint(0, cfg.get('num_classes', 10), (bs,)).cuda()
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()
    for _ in range(steps):
        opt.zero_grad()
        with torch.cuda.amp.autocast():
            loss = crit(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()

    n = STATS['n']
    print(f"\n### {tag}   ({os.path.basename(cfg_path)})")
    print(f"  a_m={cfg.get('snnbp_alpha'):.4g} a_d={cfg.get('snnbp_beta'):.4g} "
          f"eps={cfg.get('snnbp_epsilon'):.4g} h={cfg.get('snnbp_intervention'):.4g}"
          f"   GradScaler scale = {scaler.get_scale():.0f}")
    print(f"  neuron-timesteps            : {n:,}")
    print(f"  |delta| <= eps              : {100*STATS['near']/n:8.3f}%")
    print(f"  INTERVENTIONS FIRED         : {100*STATS['fire']/n:8.3f}%  ({STATS['fire']:,})")
    print(f"    ... of which m < 0        : {STATS['m_neg_fire']:,}   "
          f"(paper Sec 6.1: 'we only intervene when m[t] > 0')")
    print(f"  g_m saturated (>0.99)       : {100*STATS['gm_sat']/n:8.3f}%   "
          f"(harmfulness gate carries no information when saturated)")
    print(f"  g_mis binary (>0.99 or <.01): {100*STATS['gmis_bin']/n:8.3f}%   "
          f"(alignment gate degenerates to a hard sign test)")
    print(f"  max |m|                     : {STATS['m_max']:.4g}   (unscaled it would be "
          f"{STATS['m_max']/scaler.get_scale():.3g})")
    print(f"  max intervention_signal     : {STATS['sig_max']:.4f}  vs h={cfg.get('snnbp_intervention'):.4g}")
    print(f"  non-finite gate values      : {STATS['nan']:,}")


if __name__ == '__main__':
    torch.manual_seed(0)
    run(f"{REPO}/cifar10_best_cons.yaml", "CIFAR-10/100 + ImageNet C-Grad config")
    run(_os.path.join(_REPO, "event/cifar10dvs_cgrad.yaml"),
        "CIFAR10-DVS C-Grad config (84.2% run)")
    run(_os.path.join(_REPO, "event/cifar10dvs_hp_ablation.yaml"),
        "Paper Table 3 defaults")

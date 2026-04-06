import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--snnbp-intervention', type=float, default=0.8)

cfg = {'snnbp-intervention': 0.562}
parser.set_defaults(**cfg)

args = parser.parse_args([])
print(f"snnbp_intervention: {getattr(args, 'snnbp_intervention', 'MISSING')}")
print(f"snnbp-intervention: {getattr(args, 'snnbp-intervention', 'MISSING')}")

cfg2 = {'snnbp_intervention': 0.562}
parser2 = argparse.ArgumentParser()
parser2.add_argument('--snnbp-intervention', type=float, default=0.8)
parser2.set_defaults(**cfg2)
args2 = parser2.parse_args([])
print(f"args2 snnbp_intervention: {getattr(args2, 'snnbp_intervention', 'MISSING')}")


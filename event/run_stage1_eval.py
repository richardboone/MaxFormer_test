#!/usr/bin/env python3
"""Shared-GPU-polite, disconnect-resilient Stage 1 runner for CIFAR10-DVS MaxFormer."""

import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time


ARMS = [
    {
        "name": "A0_bptt_sig",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=sigmoid",
            "--snnbp-tau=2.0",
            "--snnbp-input-scale=coupled",
            "--detach-reset=True",
            "--snnbp-eta=0.0",
        ],
    },
    {
        "name": "A1_impr_harmful_sig",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=sigmoid",
            "--snnbp-tau=2.0",
            "--snnbp-input-scale=coupled",
            "--detach-reset=True",
            "--snnbp-eta=1.0",
            "--snnbp-harmful-only=True",
        ],
    },
    {
        "name": "A2_impr_twosided_sig",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=sigmoid",
            "--snnbp-tau=2.0",
            "--snnbp-input-scale=coupled",
            "--detach-reset=True",
            "--snnbp-eta=1.0",
            "--snnbp-harmful-only=False",
        ],
    },
    {
        "name": "B0_bptt_gamma",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=Gamma",
            "--snnbp-tau=2.0",
            "--snnbp-input-scale=coupled",
            "--detach-reset=True",
            "--snnbp-eta=0.0",
        ],
    },
    {
        "name": "B1_impr_harmful_gamma",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=Gamma",
            "--snnbp-tau=2.0",
            "--snnbp-input-scale=coupled",
            "--detach-reset=True",
            "--snnbp-eta=1.0",
            "--snnbp-harmful-only=True",
        ],
    },
    {
        "name": "B2_prior_inv_cgrad",
        "config": "./cifar10dvs_invariant.yaml",
        "flags": [
            "--du-du=invariant_cgrad",
            "--dS-du=Gamma",
            "--snnbp-tau=0.5",
            "--snnbp-input-scale=coupled",
            "--detach-reset=True",
            "--snnbp-consistency=sign",
            "--snnbp-correction=flip",
            "--snnbp-two-sided=True",
            "--snnbp-alpha-cons=6.0",
            "--snnbp-alpha=0.51",
            "--snnbp-beta=2.91",
            "--snnbp-kappa=0.728",
            "--snnbp-epsilon=0.470",
            "--snnbp-h=0.24",
        ],
    },
    {
        "name": "B3_impr_twosided_gamma",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=Gamma",
            "--snnbp-tau=2.0",
            "--snnbp-input-scale=coupled",
            "--detach-reset=True",
            "--snnbp-eta=1.0",
            "--snnbp-harmful-only=False",
        ],
    },
    {
        "name": "C0_bptt_undetached",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=sigmoid",
            "--snnbp-tau=2.0",
            "--snnbp-input-scale=coupled",
            "--detach-reset=False",
            "--snnbp-eta=0.0",
        ],
    },
    {
        "name": "C1_impr_undetached",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=sigmoid",
            "--snnbp-tau=2.0",
            "--snnbp-input-scale=coupled",
            "--detach-reset=False",
            "--snnbp-eta=1.0",
            "--snnbp-harmful-only=True",
        ],
    },
    {
        "name": "C2_impr_twosided_undetached",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=sigmoid",
            "--snnbp-tau=2.0",
            "--snnbp-input-scale=coupled",
            "--detach-reset=False",
            "--snnbp-eta=1.0",
            "--snnbp-harmful-only=False",
        ],
    },
    {
        "name": "D0_bptt_tau20_scale1",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=sigmoid",
            "--snnbp-tau=20.0",
            "--snnbp-input-scale=1.0",
            "--detach-reset=False",
            "--snnbp-eta=0.0",
        ],
    },
    {
        "name": "D1_impr_tau20_scale1",
        "config": "./cifar10dvs_improved_cgrad.yaml",
            "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=sigmoid",
            "--snnbp-tau=20.0",
            "--snnbp-input-scale=1.0",
            "--detach-reset=False",
            "--snnbp-eta=1.0",
            "--snnbp-harmful-only=True",
        ],
    },
    {
        "name": "D2_impr_twosided_tau20_scale1",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=sigmoid",
            "--snnbp-tau=20.0",
            "--snnbp-input-scale=1.0",
            "--detach-reset=False",
            "--snnbp-eta=1.0",
            "--snnbp-harmful-only=False",
        ],
    },
]


def query_gpu_stats():
    for attempt in range(3):
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=15,
            )
            stats = {}
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 3:
                    idx, mem_mb, util = int(parts[0]), int(parts[1]), int(parts[2])
                    stats[idx] = (mem_mb, util)
            if stats:
                return stats
        except (subprocess.SubprocessError, OSError, ValueError):
            pass
        if attempt < 2:
            time.sleep(2.0)
    return {}


def pick_free_gpu(active_jobs, candidate_gpus):
    stats = query_gpu_stats()
    if not stats:
        return None, 0, 0
    busy_ours = {job["gpu"] for job in active_jobs}
    external_busy = 0
    for idx, (mem_mb, util) in stats.items():
        if idx in busy_ours:
            continue
        if mem_mb >= 1000 or util >= 15:
            external_busy += 1

    if external_busy <= 3:
        max_slots = 4
    elif external_busy == 4:
        max_slots = 3
    else:
        max_slots = 2

    max_slots = min(max_slots, len(candidate_gpus))
    if len(active_jobs) >= max_slots:
        return None, max_slots, external_busy

    for g in candidate_gpus:
        if g in busy_ours or g not in stats:
            continue
        mem_mb, util = stats[g]
        if mem_mb < 1000 and util < 10:
            time.sleep(2.0)
            stats2 = query_gpu_stats()
            m2, u2 = stats2.get(g, (99999, 100))
            if m2 < 1000 and u2 < 10:
                return g, max_slots, external_busy
    return None, max_slots, external_busy


def write_json_atomic(path, payload):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def log_msg(master_log_path, msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(master_log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    try:
        print(line, flush=True)
    except OSError:
        pass


def load_summary_status(summary_path):
    if not os.path.exists(summary_path):
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def both_prior_seeds_collapsed(arm_name, seed, seeds, summaries_dir):
    prior_seeds = [s for s in seeds if s < seed]
    if len(prior_seeds) < 2:
        return False
    collapsed_cnt = 0
    for ps in prior_seeds:
        sp = os.path.join(summaries_dir, f"s1_{arm_name}_seed{ps}.json")
        s = load_summary_status(sp)
        if not s:
            return False
        if s.get("status") == "nan_collapsed":
            collapsed_cnt += 1
        elif s.get("status") == "completed" and s.get("val_mean_last_k", 100.0) < 35.0:
            collapsed_cnt += 1
        else:
            return False
    return collapsed_cnt >= 2


def main():
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)

    parser = argparse.ArgumentParser(description="Stage 1 CIFAR10-DVS Evaluation Scheduler")
    parser.add_argument("--data-path", default="/data/rboone/datasets/wg_dvst")
    parser.add_argument("--gpus", default="0,1,2,3,4")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cgrad-stats-prob", type=float, default=0.02)
    parser.add_argument("--out-dir", default="./logs/stage1_eval")
    parser.add_argument("--arms", default="", help="Optional comma-separated subset of arm names")
    parser.add_argument("--stall-timeout", type=int, default=900, help="Seconds of log inactivity before killing stalled run")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--dry-run", action="store_true")
    cli = parser.parse_args()

    event_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(event_dir)

    candidate_gpus = [int(x) for x in cli.gpus.split(",") if x.strip()]
    seeds = [int(x) for x in cli.seeds.split(",") if x.strip()]
    arm_filter = {x.strip() for x in cli.arms.split(",") if x.strip()} if cli.arms else None

    out_dir = os.path.abspath(cli.out_dir)
    run_logs_dir = os.path.join(out_dir, "run_logs")
    summaries_dir = os.path.join(out_dir, "summaries")
    os.makedirs(run_logs_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)
    master_log = os.path.join(out_dir, "master.log")
    status_path = os.path.join(out_dir, "status.json")

    selected_arms = [a for a in ARMS if (arm_filter is None or a["name"] in arm_filter)]
    tasks = []
    skipped = 0
    for seed in seeds:
        for arm in selected_arms:
            exp_name = f"s1_{arm['name']}_seed{seed}"
            summary_path = os.path.join(summaries_dir, f"{exp_name}.json")
            existing = load_summary_status(summary_path)
            if existing and existing.get("status") in ("completed", "nan_collapsed"):
                skipped += 1
                continue
            tasks.append({
                "arm": arm["name"],
                "seed": seed,
                "exp_name": exp_name,
                "config": arm["config"],
                "flags": arm["flags"],
                "summary_path": summary_path,
                "log_path": os.path.join(run_logs_dir, f"{exp_name}.log"),
                "retries": 0,
            })

    log_msg(
        master_log,
        f"Stage 1 scheduler starting: {len(tasks)} pending tasks ({skipped} already finished), "
        f"candidate_gpus={candidate_gpus}, seeds={seeds}, epochs={cli.epochs}",
    )

    if cli.dry_run:
        for idx, t in enumerate(tasks, 1):
            print(f"{idx:2d}. {t['exp_name']} | config={t['config']} | {' '.join(t['flags'])}")
        return

    active_jobs = []
    completed_list = []
    failed_list = []

    while tasks or active_jobs:
        # Poll finished or stalled jobs
        still_active = []
        now = time.time()
        for job in active_jobs:
            t = job["task"]
            ret = job["proc"].poll()
            if ret is None and cli.stall_timeout > 0:
                last_activity = job["launch_time"]
                if os.path.exists(t["log_path"]):
                    last_activity = max(last_activity, os.path.getmtime(t["log_path"]))
                idle = now - last_activity
                if idle > cli.stall_timeout:
                    log_msg(
                        master_log,
                        f"STALL {t['exp_name']} on GPU {job['gpu']} (idle {idle:.0f}s > {cli.stall_timeout}s) — killing process group",
                    )
                    try:
                        os.killpg(job["proc"].pid, signal.SIGTERM)
                        job["proc"].wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(job["proc"].pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        try:
                            job["proc"].wait(timeout=10)
                        except Exception:
                            pass
                    except ProcessLookupError:
                        pass
                    ret = -99
            if ret is None:
                still_active.append(job)
                continue
            job["log_fp"].close()
            summary = load_summary_status(t["summary_path"])
            st = summary.get("status") if summary else ("completed" if ret == 0 else "failed")
            if ret == 0 and st == "completed":
                acc_last = summary.get("val_mean_last_k", 0.0) if summary else 0.0
                acc_max = summary.get("max_val_acc1", 0.0) if summary else 0.0
                log_msg(
                    master_log,
                    f"DONE  {t['exp_name']} on GPU {job['gpu']} | mean_last10={acc_last:.2f} max={acc_max:.2f}",
                )
                completed_list.append(t["exp_name"])
            elif st == "nan_collapsed":
                log_msg(master_log, f"NAN   {t['exp_name']} on GPU {job['gpu']} (recorded as nan_collapsed)")
                completed_list.append(t["exp_name"])
            else:
                if t["retries"] < 2:
                    t["retries"] += 1
                    log_msg(master_log, f"RETRY {t['exp_name']} after exit={ret} on GPU {job['gpu']} (attempt {t['retries'] + 1})")
                    tasks.insert(0, t)
                else:
                    log_msg(master_log, f"FAIL  {t['exp_name']} permanently after retries (exit={ret})")
                    failed_list.append(t["exp_name"])
        active_jobs = still_active

        # Launch new jobs if polite capacity allows
        while tasks:
            next_task = tasks[0]
            if both_prior_seeds_collapsed(next_task["arm"], next_task["seed"], seeds, summaries_dir):
                tasks.pop(0)
                skipped += 1
                log_msg(master_log, f"SKIP  {next_task['exp_name']} (prior 2 seeds collapsed)")
                continue
            gpu_id, max_slots, ext_busy = pick_free_gpu(active_jobs, candidate_gpus)
            if gpu_id is None:
                break
            t = tasks.pop(0)
            cmd = [
                sys.executable,
                "train.py",
                "-c",
                t["config"],
                f"--data-path={cli.data_path}",
                "--dataset=cifar10dvs",
                "--model=max_former",
                "--T=16",
                f"--batch-size={cli.batch_size}",
                f"--epochs={cli.epochs}",
                f"--workers={cli.workers}",
                "--lr=0.0045",
                "--weight-decay=0.0079",
                "--amp=True",
                f"--seed={t['seed']}",
                f"--experiment={t['exp_name']}",
                f"--summary-json={t['summary_path']}",
                "--use-custom-neuron=True",
                "--reset-mode=hard",
                "--log-cgrad-stats=True",
                f"--cgrad-stats-prob={cli.cgrad_stats_prob}",
                "--metric-window=10",
                "--early-stop-patience=-1",
                "--device=cuda:0",
            ]
            if not cli.no_wandb:
                cmd.append("--log-wandb")
            cmd.extend(t["flags"])

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["PYTHONUNBUFFERED"] = "1"

            log_fp = open(t["log_path"], "a", encoding="utf-8")
            log_fp.write(f"\n=== Launching on GPU {gpu_id} at {datetime.datetime.now().isoformat()} ===\n")
            log_fp.write("CMD: " + " ".join(cmd) + "\n")
            log_fp.flush()

            proc = subprocess.Popen(
                cmd,
                cwd=event_dir,
                env=env,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            active_jobs.append({"task": t, "proc": proc, "gpu": gpu_id, "log_fp": log_fp, "launch_time": time.time()})
            log_msg(
                master_log,
                f"START {t['exp_name']} on GPU {gpu_id} (active={len(active_jobs)}/{max_slots}, ext_busy={ext_busy})",
            )

        write_json_atomic(
            status_path,
            {
                "updated_at": datetime.datetime.now().isoformat(),
                "active": [{"exp": j["task"]["exp_name"], "gpu": j["gpu"], "pid": j["proc"].pid} for j in active_jobs],
                "pending_count": len(tasks),
                "completed_count": len(completed_list) + skipped,
                "failed": failed_list,
            },
        )
        if tasks or active_jobs:
            time.sleep(15.0)

    log_msg(
        master_log,
        f"Stage 1 scheduler finished! completed={len(completed_list) + skipped}, failed={len(failed_list)} {failed_list}",
    )
    if failed_list:
        sys.exit(1)


if __name__ == "__main__":
    main()

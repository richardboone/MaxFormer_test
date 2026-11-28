#!/usr/bin/env python3
"""Set 2 CIFAR10-DVS Improved C-Grad HP tuning scheduler (T=16, paper lr/wd anchor)."""

import argparse
import datetime
import os
import signal
import subprocess
import sys
import time

EVENT_DIR = os.path.dirname(os.path.abspath(__file__))
if EVENT_DIR not in sys.path:
    sys.path.insert(0, EVENT_DIR)

from run_tune_phase1 import (  # noqa: E402
    both_prior_seeds_collapsed,
    load_summary_status,
    log_msg,
    pick_free_gpu,
    write_json_atomic,
)


SET2_BASE_FLAGS = [
    "--lr=0.005",
    "--weight-decay=0.06",
    "--amp=True",
    "--use-custom-neuron=True",
    "--du-du=improved_cgrad",
    "--dS-du=sigmoid",
    "--snnbp-tau=2.0",
    "--snnbp-input-scale=coupled",
    "--detach-reset=True",
    "--reset-mode=hard",
    "--surrogate-alpha=4.0",
    "--snnbp-norm=local",
    "--snnbp-misalign-mode=product",
    "--snnbp-harmful-only=False",
    "--snnbp-alpha-mis=25.0",
    "--snnbp-alpha=6.0",
    "--snnbp-margin-m=0.20",
    "--snnbp-epsilon=0.35",
    "--snnbp-beta=8.0",
    "--snnbp-intervention=0.0",
    "--snnbp-gamma-sharp=0.0",
    "--snnbp-corr-scale=0.5",
    "--snnbp-eta=0.35",
]


SET2_ARMS = [
    {
        "name": "S2_A0_bptt",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "base_flags": SET2_BASE_FLAGS,
        "flags": ["--snnbp-eta=0.0"],
        "is_ref": True,
    },
    {
        "name": "S2_T8_paper",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "base_flags": SET2_BASE_FLAGS,
        "flags": [],
    },
    {
        "name": "S2_lr003_coupled",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "base_flags": SET2_BASE_FLAGS,
        "flags": ["--lr=0.003", "--snnbp-epsilon=0.21", "--snnbp-beta=13.3"],
    },
    {
        "name": "S2_lr003_fixed",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "base_flags": SET2_BASE_FLAGS,
        "flags": ["--lr=0.003"],
    },
    {
        "name": "S2_lr007_coupled",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "base_flags": SET2_BASE_FLAGS,
        "flags": ["--lr=0.007", "--snnbp-epsilon=0.49", "--snnbp-beta=5.7"],
    },
    {
        "name": "S2_lr007_fixed",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "base_flags": SET2_BASE_FLAGS,
        "flags": ["--lr=0.007"],
    },
    {
        "name": "S2_lr005_wd02",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "base_flags": SET2_BASE_FLAGS,
        "flags": ["--weight-decay=0.02"],
    },
    {
        "name": "S2_gates_tight",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "base_flags": SET2_BASE_FLAGS,
        "flags": [
            "--snnbp-epsilon=0.22",
            "--snnbp-beta=12.0",
            "--snnbp-alpha-mis=40.0",
            "--snnbp-margin-m=0.25",
            "--snnbp-eta=0.55",
        ],
    },
    {
        "name": "S2_gates_broad",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "base_flags": SET2_BASE_FLAGS,
        "flags": [
            "--snnbp-epsilon=0.50",
            "--snnbp-beta=5.0",
            "--snnbp-alpha-mis=15.0",
            "--snnbp-margin-m=0.10",
            "--snnbp-eta=0.20",
        ],
    },
    {
        "name": "S2_gates_wide_near_sharp_mis",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "base_flags": SET2_BASE_FLAGS,
        "flags": [
            "--snnbp-epsilon=0.50",
            "--snnbp-beta=6.0",
            "--snnbp-alpha-mis=35.0",
            "--snnbp-margin-m=0.25",
            "--snnbp-eta=0.35",
        ],
    },
    {
        "name": "S2_tau15",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "base_flags": SET2_BASE_FLAGS,
        "flags": ["--snnbp-tau=1.5"],
    },
    {
        "name": "S2_tau30",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "base_flags": SET2_BASE_FLAGS,
        "flags": ["--snnbp-tau=3.0"],
    },
    {
        "name": "S2_soft_reset",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "base_flags": SET2_BASE_FLAGS,
        "flags": ["--reset-mode=soft"],
    },
    {
        "name": "S2_surr_alpha6",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "base_flags": SET2_BASE_FLAGS,
        "flags": ["--surrogate-alpha=6.0"],
    },
    {
        "name": "S2_eta_decay",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "base_flags": SET2_BASE_FLAGS,
        "flags": ["--snnbp-eta=0.55", "--snnbp-eta-end=0.15"],
    },
    {
        "name": "S2_eta_warmup",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "base_flags": SET2_BASE_FLAGS,
        "flags": ["--snnbp-eta=0.45", "--snnbp-eta-warmup-epochs=10"],
    },
]


def _write_done_flag(path, completed_count, failed_list):
    if not path:
        return
    write_json_atomic(
        path,
        {
            "finished_at": datetime.datetime.now().isoformat(),
            "completed": completed_count,
            "failed": failed_list,
        },
    )


def main():
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)

    parser = argparse.ArgumentParser(description="Set 2 CIFAR10-DVS Improved C-Grad HP Tuning Scheduler")
    parser.add_argument("--data-path", default="/data/rboone/datasets/wg_dvst")
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--max-gpus", type=int, default=0)
    parser.add_argument("--seeds", default="42,43")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--cgrad-stats-prob", type=float, default=0.02)
    parser.add_argument("--out-dir", default="./logs/dvs_set2_tune")
    parser.add_argument("--prefix", default="d2_")
    parser.add_argument("--arms", default="")
    parser.add_argument("--skip-exps", default="")
    parser.add_argument("--done-flag", default="/tmp/dvs_set2_done.flag")
    parser.add_argument("--ignore-polite-cap", action="store_true")
    parser.add_argument("--stall-timeout", type=int, default=1200)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-online", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    cli = parser.parse_args()

    os.chdir(EVENT_DIR)

    candidate_gpus = [int(x) for x in cli.gpus.split(",") if x.strip()]
    arm_filter = {x.strip() for x in cli.arms.split(",") if x.strip()} if cli.arms else None
    skip_exps = {x.strip() for x in cli.skip_exps.split(",") if x.strip()} if cli.skip_exps else set()
    seeds = [int(x) for x in cli.seeds.split(",") if x.strip()]

    out_dir = os.path.abspath(cli.out_dir)
    run_logs_dir = os.path.join(out_dir, "run_logs")
    summaries_dir = os.path.join(out_dir, "summaries")
    os.makedirs(run_logs_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)
    master_log = os.path.join(out_dir, "master.log")
    status_path = os.path.join(out_dir, "status.json")

    if cli.workers is None:
        cli.workers = 4 if cli.jobs_per_gpu == 1 else 2

    selected_arms = [a for a in SET2_ARMS if (arm_filter is None or a["name"] in arm_filter)]
    tasks = []
    skipped = 0
    for seed in seeds:
        for arm in selected_arms:
            exp_name = f"{cli.prefix}{arm['name']}_seed{seed}"
            if exp_name in skip_exps:
                skipped += 1
                continue
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
                "base_flags": arm["base_flags"],
                "flags": arm["flags"],
                "summary_path": summary_path,
                "epochs_jsonl_path": f"{summary_path[:-5]}.epochs.jsonl",
                "log_path": os.path.join(run_logs_dir, f"{exp_name}.log"),
                "retries": 0,
            })

    if cli.dry_run:
        print(
            f"Set 2 tuning dry-run: {len(tasks)} pending tasks ({skipped} already finished), "
            f"candidate_gpus={candidate_gpus}, jobs_per_gpu={cli.jobs_per_gpu}, "
            f"workers={cli.workers}, seeds={seeds}, epochs={cli.epochs}"
        )
        for idx, t in enumerate(tasks, 1):
            print(f"{idx:2d}. {t['exp_name']} | {' '.join(t['flags']) or '[default paper T8]'}")
        return

    log_msg(
        master_log,
        f"Set 2 tuning scheduler starting: {len(tasks)} pending tasks ({skipped} already finished), "
        f"candidate_gpus={candidate_gpus}, jobs_per_gpu={cli.jobs_per_gpu}, "
        f"workers={cli.workers}, seeds={seeds}, epochs={cli.epochs}, done_flag={cli.done_flag}",
    )

    active_jobs = []
    completed_list = []
    failed_list = []

    while tasks or active_jobs:
        still_active = []
        now = time.time()
        for job in active_jobs:
            t = job["task"]
            ret = job["proc"].poll()
            summary = load_summary_status(t["summary_path"])
            summary_fresh = bool(
                summary
                and os.path.exists(t["summary_path"])
                and os.path.getmtime(t["summary_path"]) >= job["launch_time"] - 1.0
            )
            if ret is None:
                if summary_fresh and summary.get("status") in ("completed", "nan_collapsed"):
                    teardown_age = now - os.path.getmtime(t["summary_path"])
                    if teardown_age > 60.0:
                        log_msg(
                            master_log,
                            f"TEARDOWN {t['exp_name']} on GPU {job['gpu']} (summary completed {teardown_age:.0f}s ago)",
                        )
                        try:
                            os.killpg(job["proc"].pid, signal.SIGTERM)
                            job["proc"].wait(timeout=5)
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
                        ret = 0
                elif cli.stall_timeout > 0:
                    last_activity = job["launch_time"]
                    if os.path.exists(t["epochs_jsonl_path"]):
                        last_activity = max(last_activity, os.path.getmtime(t["epochs_jsonl_path"]))
                    idle = now - last_activity
                    if idle > cli.stall_timeout:
                        log_msg(
                            master_log,
                            f"STALL {t['exp_name']} on GPU {job['gpu']} (idle {idle:.0f}s > {cli.stall_timeout}s) — killing",
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
            summary_fresh = bool(
                summary
                and os.path.exists(t["summary_path"])
                and os.path.getmtime(t["summary_path"]) >= job["launch_time"] - 1.0
            )
            st = summary.get("status") if summary else ("completed" if ret == 0 else "failed")
            if (ret == 0 and st == "completed") or (summary_fresh and st == "completed"):
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
                    log_msg(
                        master_log,
                        f"RETRY {t['exp_name']} after exit={ret} on GPU {job['gpu']} (attempt {t['retries'] + 1})",
                    )
                    tasks.insert(0, t)
                else:
                    log_msg(master_log, f"FAIL  {t['exp_name']} permanently after retries (exit={ret})")
                    failed_list.append(t["exp_name"])
        active_jobs = still_active

        while tasks:
            next_task = tasks[0]
            if both_prior_seeds_collapsed(next_task["arm"], next_task["seed"], seeds, summaries_dir, cli.prefix):
                tasks.pop(0)
                skipped += 1
                log_msg(master_log, f"SKIP  {next_task['exp_name']} (prior 2 seeds collapsed)")
                continue
            gpu_id, max_slots, ext_busy = pick_free_gpu(
                active_jobs,
                candidate_gpus,
                jobs_per_gpu=cli.jobs_per_gpu,
                max_gpus_override=cli.max_gpus,
                ignore_polite_cap=cli.ignore_polite_cap,
            )
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
                f"--seed={t['seed']}",
                f"--experiment={t['exp_name']}",
                f"--summary-json={t['summary_path']}",
                "--log-cgrad-stats=True",
                f"--cgrad-stats-prob={cli.cgrad_stats_prob}",
                "--metric-window=10",
                "--early-stop-patience=-1",
                "--device=cuda:0",
            ]
            if not cli.no_wandb:
                cmd.append("--log-wandb")
            cmd.extend(t["base_flags"])
            cmd.extend(t["flags"])

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["PYTHONUNBUFFERED"] = "1"
            if not cli.wandb_online and not cli.no_wandb:
                env["WANDB_MODE"] = "offline"

            log_fp = open(t["log_path"], "a", encoding="utf-8")
            log_fp.write(f"\n=== Launching on GPU {gpu_id} at {datetime.datetime.now().isoformat()} ===\n")
            log_fp.write("CMD: " + " ".join(cmd) + "\n")
            log_fp.flush()

            proc = subprocess.Popen(
                cmd,
                cwd=EVENT_DIR,
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
                "candidate_gpus": candidate_gpus,
                "active": [{"exp": j["task"]["exp_name"], "gpu": j["gpu"], "pid": j["proc"].pid} for j in active_jobs],
                "pending_count": len(tasks),
                "completed_count": len(completed_list) + skipped,
                "failed": failed_list,
            },
        )
        if tasks or active_jobs:
            time.sleep(15.0)

    _write_done_flag(cli.done_flag, len(completed_list) + skipped, failed_list)
    log_msg(
        master_log,
        f"Set 2 tuning scheduler finished! completed={len(completed_list) + skipped}, failed={len(failed_list)} {failed_list}; "
        f"wrote done_flag={cli.done_flag}",
    )
    if failed_list:
        sys.exit(1)


if __name__ == "__main__":
    main()

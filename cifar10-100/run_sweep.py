import subprocess
import time
import pynvml
import argparse
import os
import sys

# --- Configuration ---
GPUS_TO_USE = [4,5,6,7]  # GPUs you are willing to use
GPU_MEMORY_THRESHOLD_MB = 50000  # Minimum free memory (in MB) to be considered "available"
MAX_AGENTS_PER_GPU = 1

def get_available_gpus():
    """Checks for available GPUs based on memory usage."""
    available_gpus = []
    pynvml.nvmlInit()
    for gpu_id in GPUS_TO_USE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_memory_mb = info.free / 1024**2
            if free_memory_mb > GPU_MEMORY_THRESHOLD_MB:
                available_gpus.append(gpu_id)
        except pynvml.NVMLError:
            continue
    pynvml.nvmlShutdown()
    return available_gpus

def main(sweep_id):
    """Main function to launch sweep agents."""
    active_agents = {}  # {gpu_id: [subprocess.Popen]}

    print(f"Starting sweep manager for sweep: {sweep_id}")
    try:
        while True:
            available_gpus = get_available_gpus()

            # Launch new agents on available GPUs
            for gpu_id in available_gpus:
                if gpu_id not in active_agents:
                    active_agents[gpu_id] = []

                if len(active_agents[gpu_id]) < MAX_AGENTS_PER_GPU:
                    print(f"Launching new agent on GPU {gpu_id}")
                    
                    # --- 2. CHANGE COMMAND CONSTRUCTION ---
                    # Instead of a string "wandb agent ...", use a list.
                    # sys.executable points to: /home/rboone/.conda/envs/maxformer/bin/python
                    # "-m wandb" runs the library module directly.
                    command = [sys.executable, "-m", "wandb", "agent", sweep_id]
                    
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                    
                    # --- 3. REMOVE shell=True ---
                    # We are passing the executable directly, so we don't need the shell.
                    # This is safer and avoids PATH lookup issues.
                    proc = subprocess.Popen(command, env=env)
                    
                    active_agents[gpu_id].append(proc)

            # Clean up finished agents
            for gpu_id, agents in list(active_agents.items()):
                for agent in list(agents):
                    if agent.poll() is not None:
                        agents.remove(agent)
                        print(f"Agent on GPU {gpu_id} finished.")
                if not agents:
                    del active_agents[gpu_id]
            
            time.sleep(10)

    except KeyboardInterrupt:
        print("\nShutting down all agents...")
        for agents in active_agents.values():
            for agent in agents:
                agent.terminate()
        print("All agents terminated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated WandB Sweep Runner")
    parser.add_argument('--sweep_id', type=str, required=True, help='Full WandB sweep ID (e.g., username/project/sweep_id)')
    args = parser.parse_args()
    
    main(args.sweep_id)
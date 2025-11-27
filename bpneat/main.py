import os
from bpneat.config import TASK_NAME
from bpneat.core.hybrid_loop import run_evolution_loop

def ensure_dirs():
    os.makedirs(f"figs/{TASK_NAME}", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

def main():
    print(f"=== Starting Backprop-NEAT for task: {TASK_NAME} ===")
    ensure_dirs()
    run_evolution_loop(
        generations=5,
        mutation_rate=1,
        lr=0.1,
        steps=300,  # 使用 steps 而不是 train_steps
        save_dir="figs/hybrid_run"
    )


    print(f"=== Finished Task: {TASK_NAME} ===")

if __name__ == "__main__":
    main()

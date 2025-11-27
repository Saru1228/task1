import os
import random
import matplotlib.pyplot as plt

from bpneat.config import TASK_NAME
from bpneat.neat.genome import make_initial_genome
from bpneat.neat.mutation import mutate_add_connection, mutate_add_node
from bpneat.nn.builder import build_forward
from bpneat.nn.train import train_genome,load_data

class InnovationTracker:
    def __init__(self):
        self.counter = 100
        self.conn_history = {}

    def get_innovation(self, src, tgt):
        key = (src, tgt)
        if key in self.conn_history:
            return self.conn_history[key]
        else:
            self.counter += 1
            self.conn_history[key] = self.counter
            return self.counter



def run_evolution_loop(
    generations=25,
    mutation_rate=0.5,
    lr=0.1,
    steps=200,
    save_dir="figs/evolution"
):
    os.makedirs(save_dir, exist_ok=True)
    loss_history = []

    genome = make_initial_genome()
    tracker = InnovationTracker()
    X, Y = load_data(TASK_NAME)

    for gen in range(generations):
        print(f"\n=== Generation {gen} ===")

        # 可视化结构图
        fig_path = os.path.join(save_dir, f"gen_{gen}.png")
        genome.draw(fig_path)

        # 随机突变
        if random.random() < mutation_rate:
            if random.random() < 0.5:
                print("-> Applying mutate_add_connection")
                mutate_add_connection(genome, tracker)
            else:
                print("-> Applying mutate_add_node")
                mutate_add_node(genome, tracker)

        # 构建前向网络 + 训练
        params, loss = train_genome(
            genome,
            f_builder=build_forward,
            X=X,
            Y=Y,
            lr=lr,
            steps=steps,
            return_loss=True,
            plot_prefix=f"evolution_gen_{gen}"
        )
        print(f"[Gen {gen}] Final Loss: {loss:.6f}")
        loss_history.append(loss)

    # 画 Loss 曲线
    plt.figure()
    plt.plot(loss_history, marker="o")
    plt.title("Evolution Loss Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "evolution_loss.png"))
    plt.close()

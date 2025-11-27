import os
import random
import matplotlib.pyplot as plt

from bpneat.config import TASK_NAME
from bpneat.nn.train import load_data
from bpneat.neat.genome import make_initial_genome
from bpneat.neat.mutation import mutate_add_connection, mutate_add_node
from bpneat.nn.builder import build_forward
from bpneat.nn.train import train_genome

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
    generations=10,
    mutation_rate=0.5,
    lr=0.1,
    train_steps=200,
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

        #new mutation algorithm
        r = random.random()
        if r < mutation_rate:
            if r < 0.2:
                print("-> Applying mutate_add_connection")
                mutate_add_connection(genome, tracker)
            elif r < 0.6:
                print("-> Applying mutate_add_node")
                mutate_add_node(genome, tracker)
            elif r < 0.8:
                print("-> Applying mutate_add_node then mutate_add_connection")
                if mutate_add_node(genome, tracker):
                    mutate_add_connection(genome, tracker)

        # 构建前向网络 + 训练
        params, loss = train_genome(
            genome,
            f_builder=build_forward,
            X=X,
            Y=Y,
            lr=lr,
            steps=train_steps,
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
    
def train_individual_loop(
    genome,
    tracker,
    X, Y,
    generations=10,
    mutation_rate=0.5,
    lr=0.1,
    train_steps=200,
    save_dir=None
):
    loss_history = []

    for gen in range(generations):
        if save_dir:
            fig_path = os.path.join(save_dir, f"gen_{gen}.png")
            genome.draw(fig_path)

        # 突变
        if random.random() < mutation_rate:
            if random.random() < 0.2:
                mutate_add_connection(genome, tracker)
            else:
                mutate_add_node(genome, tracker)

        # 权重训练
        params, loss = train_genome(
            genome,
            f_builder=build_forward,
            X=X,
            Y=Y,
            lr=lr,
            steps=train_steps,
            return_loss=True,
            plot_prefix=f"{save_dir}/gen_{gen}" if save_dir else None
        )
        loss_history.append(loss)

    return genome, loss_history[-1]  # 返回最终个体和 loss

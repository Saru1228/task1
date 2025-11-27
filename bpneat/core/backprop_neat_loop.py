import os
import matplotlib.pyplot as plt
import jax.numpy as jnp

from bpneat.config import TASK_NAME
from bpneat.neat.genome import make_initial_genome
from bpneat.nn.builder import build_forward
from bpneat.nn.train import train_genome
from bpneat.nn.train import load_data


def count_complexity(genome):
    num_nodes = len(genome.nodes)
    num_connections = sum(1 for c in genome.connections if c.enabled)
    return num_nodes + num_connections


def backprop_neat_one_generation(pop_size=5, lambda_complexity=0.01):
    population = []
    results = []

    # 1. 加载数据集
    X, Y = load_data(TASK_NAME)

    # 2. 初始化种群
    for _ in range(pop_size):
        genome = make_initial_genome()
        population.append(genome)

    # 3. 逐个训练并评估
    for idx, genome in enumerate(population):
        print(f"\n=== Training Genome #{idx} ===")
        trained_params, final_loss = train_genome(
            genome,
            build_forward,
            X=X,
            Y=Y,
            steps=200,
            return_loss=True,
            plot_prefix=f"backprop_genome_{idx}"
        )

        # 可视化当前训练后的结构
        filename = f"figs/genome_{idx}_after_train.png"
        genome.draw(filename)
        print(f"Saved trained structure to: {filename}")

        complexity = count_complexity(genome)
        fitness = final_loss + lambda_complexity * complexity

        print(f"Genome #{idx} | Loss: {final_loss:.4f}, Complexity: {complexity}, Fitness: {fitness:.4f}")
        results.append((fitness, genome, trained_params))

    # 4. 选择最优个体
    results.sort(key=lambda x: x[0])  # 最小 fitness 最好
    best_fitness, best_genome, best_params = results[0]

    # 5. 输出可视化
    print(f"\n== Best Genome ==")
    best_genome.draw("figs/best_genome.png")
    print(f"Saved structure to: figs/best_genome.png")

    # 6. 测试输出
    f = build_forward(best_genome)
    print("\n== Final Predictions (Best Genome) ==")
    for x in X:
        y = f(x, best_params)
        y_clean = [0.0 if abs(v) < 1e-18 else float(v) for v in y.tolist()]
        print(f"x={x.tolist()} -> y={y_clean}")


if __name__ == "__main__":
    os.makedirs("figs", exist_ok=True)
    backprop_neat_one_generation()

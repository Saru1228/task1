import os
import copy
import random
import matplotlib.pyplot as plt

from bpneat.config import TASK_NAME
from bpneat.neat.genome import make_initial_genome
from bpneat.neat.mutation import mutate, crossover
from bpneat.nn.train import train_genome
from bpneat.nn.builder import build_forward
from bpneat.nn.train import load_data


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


class Population:
    def __init__(self, size=10, elite_size=2, tracker=None, lr=0.1, steps=300, crossover_rate=0.5):
        self.size = size
        self.elite_size = elite_size
        self.lr = lr
        self.steps = steps
        self.tracker = tracker or InnovationTracker()
        self.crossover_rate = crossover_rate

        self.genomes = [make_initial_genome() for _ in range(size)]
        self.mutate_population()
        self.loss_history = []

        # 数据集加载一次即可
        self.X, self.Y = load_data(TASK_NAME)

    def mutate_population(self):
        for genome in self.genomes:
            mutate(genome, self.tracker)

    def evaluate(self):
        results = []
        for genome in self.genomes:
            params, loss = train_genome(
                genome,
                build_forward,
                X=self.X,
                Y=self.Y,
                lr=self.lr,
                steps=self.steps,
                return_loss=True,
                plot_prefix=f"pop_{TASK_NAME}"
            )
            results.append((genome, loss))
        return results

    def evolve(self, generation):
        results = self.evaluate()
        results.sort(key=lambda x: x[1])

        best_genome, best_loss = results[0]
        print(f"[Gen {generation}] Best Loss: {best_loss:.6f}")
        self.loss_history.append(best_loss)

        os.makedirs("figs/evo", exist_ok=True)
        best_genome.draw(f"figs/evo/pop_gen_{generation}.png")

        elites = [copy.deepcopy(g) for g, _ in results[:self.elite_size]]

        new_genomes = []
        while len(new_genomes) + len(elites) < self.size:
            if random.random() < self.crossover_rate and len(elites) >= 2:
                p1, p2 = random.sample(elites, 2)
                child = crossover(p1, p2)
            else:
                parent = random.choice(elites)
                child = copy.deepcopy(parent)
                mutate(child, self.tracker)
            new_genomes.append(child)

        self.genomes = elites + new_genomes

    def save_summary(self, out_dir="figs/evo"):
        os.makedirs(out_dir, exist_ok=True)
        plt.plot(self.loss_history, marker="o")
        plt.title("Best Loss per Generation")
        plt.xlabel("Generation")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(f"{out_dir}/loss_curve.png")
        plt.close()

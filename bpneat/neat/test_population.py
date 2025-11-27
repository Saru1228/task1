from bpneat.neat.population import Population

if __name__ == "__main__":
    pop = Population(size=2, elite_size=2, steps=200, lr=0.2)
    for gen in range(5):
        pop.evolve(generation=gen)
    pop.save_summary()

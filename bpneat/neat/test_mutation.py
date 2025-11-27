from bpneat.neat.genome import make_fixed_xor_genome
from bpneat.neat.mutation import InnovationTracker, mutate_add_connection, mutate_add_node

def test_mutate():
    genome = make_fixed_xor_genome()
    tracker = InnovationTracker()

    print("== Before Mutation ==")
    genome.draw("figs/before_mutation.png")

    mutate_add_connection(genome, tracker)
    mutate_add_node(genome, tracker)

    print("== After Mutation ==")
    genome.draw("figs/after_mutation.png")

if __name__ == "__main__":
    import os
    os.makedirs("figs", exist_ok=True)
    test_mutate()

from bpneat.neat.genome import make_initial_genome
from bpneat.nn.builder import build_forward
from bpneat.nn.train import train_genome


def test_train():
    genome = make_initial_genome()
    print("== Begin Training ==")
    trained_params = train_genome(genome, f_builder=build_forward, steps=300)


if __name__ == "__main__":
    test_train()

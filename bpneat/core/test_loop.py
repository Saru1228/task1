from bpneat.core.loop import run_evolution_loop

def test_loop():
    run_evolution_loop(
        generations=10,
        mutation_rate=1,
        lr=0.1,
        train_steps=200,
        save_dir="figs/evolution"
    )

if __name__ == "__main__":
    test_loop()

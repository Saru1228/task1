# test/test_builder.py

import jax
import jax.numpy as jnp
from bpneat.nn.builder import build_forward
from bpneat.neat.genome import make_fixed_xor_genome



def test_builder():
    genome = make_fixed_xor_genome()
    f = build_forward(genome)

    # 权重与结构保持一致
    params = {
        (0, 3): 5.0,
        (1, 3): 5.0,
        (3, 2): -10.0
    }

    # XOR 输入输出对
    inputs = [
        jnp.array([0.0, 0.0]),
        jnp.array([0.0, 1.0]),
        jnp.array([1.0, 0.0]),
        jnp.array([1.0, 1.0])
    ]

    print("== XOR Network Output ==")
    for x in inputs:
        y = f(x, params)
        y_clean = [0.0 if abs(v) < 1e-18 else float(v) for v in y.tolist()]
        print(f"x={x.tolist()} -> y={y_clean}")

if __name__ == "__main__":
    test_builder()

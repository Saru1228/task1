import jax.numpy as jnp
import jax
from bpneat.config import TASK_NAME

def get_dataset():
    if TASK_NAME == 'xor':
        return get_xor_data()
    elif TASK_NAME == 'circle':
        return get_circle_data()
    elif TASK_NAME == 'spiral':
        return get_spiral_data()
    else:
        raise ValueError(f"Unsupported TASK_NAME: {TASK_NAME}")

def get_xor_data():
    X = jnp.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ])
    Y = jnp.array([
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ])
    return X, Y

def get_circle_data(n_samples=200, noise=0.1, inner_radius=0.5, outer_radius=1.0, seed=42):
    import numpy as np
    rng = np.random.RandomState(seed)

    # 内圈
    theta1 = 2 * np.pi * rng.rand(n_samples // 2)
    r1 = inner_radius + noise * rng.randn(n_samples // 2)
    x1 = r1 * np.cos(theta1)
    y1 = r1 * np.sin(theta1)
    data1 = np.stack([x1, y1], axis=1)
    label1 = np.zeros((n_samples // 2, 1))

    # 外圈
    theta2 = 2 * np.pi * rng.rand(n_samples // 2)
    r2 = outer_radius + noise * rng.randn(n_samples // 2)
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)
    data2 = np.stack([x2, y2], axis=1)
    label2 = np.ones((n_samples // 2, 1))

    # 合并
    X = np.vstack([data1, data2])
    Y = np.vstack([label1, label2])

    # 转为 JAX 格式
    return jax.numpy.array(X), jax.numpy.array(Y)


def get_spiral_data(n_samples=200, noise=0.2, rotations=3, seed=42):
    import numpy as np
    rng = np.random.RandomState(seed)

    n_class = 2
    samples_per_class = n_samples // n_class
    X = []
    Y = []

    for label in range(n_class):
        r = np.linspace(0.0, 1, samples_per_class)
        theta = r * rotations * 2 * np.pi + (np.pi if label == 1 else 0)
        x = r * np.cos(theta) + noise * rng.randn(samples_per_class)
        y = r * np.sin(theta) + noise * rng.randn(samples_per_class)
        X.append(np.stack([x, y], axis=1))
        Y.append(np.full((samples_per_class, 1), label))

    X = np.vstack(X)
    Y = np.vstack(Y)

    return jax.numpy.array(X), jax.numpy.array(Y)


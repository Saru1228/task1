import jax
import jax.numpy as jnp
import os
import optax
import matplotlib.pyplot as plt

from bpneat.config import TASK_NAME
from bpneat.data import datasets
from bpneat.nn.builder import topo_sort


def mse_loss(y_pred, y_true):
    return jnp.mean((y_pred - y_true) ** 2)


def apply_threshold(y, eps=1e-18):
    return [0.0 if abs(v) < eps else float(v) for v in y.tolist()]


def load_data(task=None):
    if task is None:
        task = TASK_NAME
    if task == "xor":
        return datasets.get_xor_data()
    elif task == "circle":
        return datasets.get_circle_data()
    elif task == "spiral":
        return datasets.get_spiral_data()
    else:
        raise ValueError(f"[train.py] Unsupported task: {task}")


def get_input_output_dim(task=None):
    X, Y = load_data(task)
    return X.shape[1], Y.shape[1]


def train_genome(genome, f_builder, task=None, X=None, Y=None, lr=0.1, steps=500, return_loss=False, plot_prefix=None):
    """
    对单个 genome 使用 backpropagation 训练任意数据集。
    """
    if task is None:
        task = TASK_NAME

    if X is None or Y is None:
        X, Y = load_data(task)

    if plot_prefix is None:
        plot_prefix = task

    try:
        f = f_builder(genome)
    except ValueError as e:
        print(f"[Skipping Genome] {e}")
        dummy_params = {}
        return (dummy_params, float("inf")) if return_loss else dummy_params
    
    output_ids = [n.id for n in genome.nodes.values() if n.type == 'output']
    topo_ids = topo_sort(genome.nodes.values(), genome.connections)
    for out_id in output_ids:
        if out_id not in topo_ids:
            print(f"[Skipping Genome] Output node {out_id} unreachable")
            return ({}, float("inf")) if return_loss else {}

    # 初始化连接权重参数
    params = {
        (conn.in_node, conn.out_node): jnp.array(conn.weight)
        for conn in genome.connections if conn.enabled
    }

    optimizer = optax.sgd(lr)
    opt_state = optimizer.init(params)

    loss_list = []

    def loss_fn(params, x, y_true):
        y_pred = f(x, params)
        return mse_loss(y_pred, y_true)

    @jax.jit
    def update(params, opt_state, x, y_true):
        grads = jax.grad(loss_fn)(params, x, y_true)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    for step in range(steps):
        total_loss = 0.0
        for x, y_true in zip(X, Y):
            params, opt_state = update(params, opt_state, x, y_true)
            total_loss += loss_fn(params, x, y_true)
        avg_loss = total_loss / len(X)
        loss_list.append(float(avg_loss))

        if step % 50 == 0 or step == steps - 1:
            print(f"[Step {step}] Loss: {avg_loss:.6f}")

    os.makedirs("figs", exist_ok=True)
    plt.plot(loss_list)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Training Loss ({task})")
    save_path = f"figs/{plot_prefix}_loss.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    print("\n== Final Predictions ==")
    for x in X:
        y = f(x, params)
        print(f"x={x.tolist()} -> y={apply_threshold(y)}")

    # 回写训练后的参数到 genome 结构中
    for (src, tgt), w in params.items():
        for conn in genome.connections:
            if conn.enabled and conn.in_node == src and conn.out_node == tgt:
                conn.weight = float(w)

    return (params, float(avg_loss)) if return_loss else params

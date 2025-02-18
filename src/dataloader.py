import jax.numpy as jnp
import jax
import pickle
from hungarian import hungarian_cover_matcher
from distance_on_torus import dist2_on_torus_matrix
from functools import partial


def batch_align(batch0, batch1):
    # center the solute
    batch0 = (batch0 - batch0[:, 0:1]) % 1
    batch1 = (batch1 - batch1[:, 0:1]) % 1

    def align_one(b0, b1):
        cost_matrix = dist2_on_torus_matrix(b0[1:], b1[1:])
        assignment = hungarian_cover_matcher(jnp.expand_dims(cost_matrix, 0))
        assignment = assignment[0, 1]
        b1 = jnp.concatenate((b1[0:1], b1[1 + assignment]))
        return b0, b1

    return jax.vmap(align_one)(batch0, batch1)


class paired_dataloader:
    def __init__(self, cfg):
        with open(f"../data/{cfg.data0}", "rb") as pickle_file:
            self.X0 = pickle.load(pickle_file)
        with open(f"../data/{cfg.data1}", "rb") as pickle_file:
            self.X1 = pickle.load(pickle_file)

        self.batch_size = cfg.batch_size

    @partial(jax.jit, static_argnames=["self"])
    def next(self, key):
        key1, key2 = jax.random.split(key, 2)
        batch0 = jax.random.choice(key1, self.X0, (self.batch_size,), replace=False)
        batch1 = jax.random.choice(key2, self.X1, (self.batch_size,), replace=False)
        batch0, batch1 = batch_align(batch0, batch1)
        return batch0, batch1

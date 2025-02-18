import jax


def dR_on_torus_matrix(x, y=None):
    if y is None:
        y = x

    def diff_fn(a, b):
        return a - b

    diff_fn = jax.vmap(jax.vmap(diff_fn, in_axes=(None, 0)), in_axes=(0, None))
    dR = diff_fn(x, y)
    return ((dR + 0.5) % 1) - 0.5


def dist2_on_torus_matrix(x, y=None):
    if y is None:
        y = x
    return (dR_on_torus_matrix(x, y) ** 2).sum(-1)


def dR_on_torus_paired(x, y):
    dR = x - y
    return ((dR + 0.5) % 1) - 0.5


def dist2_on_torus_paired(x, y):
    dR = x - y
    return (dR_on_torus_paired(x, y) ** 2).sum(-1)

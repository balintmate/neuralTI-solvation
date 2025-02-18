import jax
import hydra
import jax.numpy as jnp
import os
import pickle
from data.canonical_mcmc import Canonical_Sampler


@hydra.main(version_base=None, config_path=".", config_name="config.yaml")
def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_devices
    #### jax flags ###
    for cfg_name, cfg_value in cfg.jax_config.items():
        jax.config.update(cfg_name, cfg_value)

    print("devices: ", *jax.devices())

    target_system = hydra.utils.instantiate(cfg.target_system)
    key = jax.random.PRNGKey(cfg.PRNGKey)
    sampler = Canonical_Sampler(target_system)

    slices = {0.0: None, 1.0: None}

    def TI_estimate_at_t(t, key):
        sampler.sample(
            key=key,
            U=lambda x: target_system.Ut(t, x),
            data_path=f"../data/LJ3Dwithsolute_{t:.3f}",
        )

        with open(f"../data/LJ3Dwithsolute_{t:.3f}", "rb") as pickle_file:
            data = pickle.load(pickle_file)

        dUdt_list = []
        for i, chunk in enumerate(jnp.split(data, (len(data) // 500))):
            dUdt = jax.jit(jax.vmap(jax.grad(target_system.Ut, 0), in_axes=(None, 0)))(
                t, chunk
            ).mean()
            dUdt_list.append(dUdt)
            print(
                f"Computing dudt for t={t:.3f}: {jnp.stack(dUdt_list).mean():.3f}",
                end="          \r",
            )
        print()
        return jnp.stack(dUdt_list).mean()

    key1, key2, key = jax.random.split(key, 3)
    slices[0.0] = TI_estimate_at_t(0.0, key1)
    slices[1.0] = TI_estimate_at_t(1.0, key2)

    def print_estimate():
        print(80 * "-")
        print(f"TI estimate with {len(slices.keys())} slices: ", end="")
        print(f"{jnp.stack(list(slices.values())).mean():.3f}")
        print(80 * "-")

    print_estimate()
    for i in range(6):
        new_slices = [k + 2 ** (-i - 1) for k in sorted(slices.keys())[:-1]]
        for s in new_slices:
            key = jax.random.split(key)[0]
            slices[s] = TI_estimate_at_t(s, key)
        print_estimate()


if __name__ == "__main__":
    main()

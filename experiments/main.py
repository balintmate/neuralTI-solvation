import jax
import wandb
import hydra
import omegaconf
import jax.numpy as jnp
import os
import optax
import time
import pickle
from dataloader import paired_dataloader
from data.canonical_mcmc import Canonical_Sampler


@hydra.main(version_base=None, config_path=".", config_name="config.yaml")
def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_devices
    #### jax flags ###
    for cfg_name, cfg_value in cfg.jax_config.items():
        jax.config.update(cfg_name, cfg_value)
    try:
        wandb_key = open("./wandb.key", "r").read()
        wandb.login(key=wandb_key)
        run = wandb.init(project=cfg.wandb_project_name)
    except:
        print("Weights and biases key not found or not valid. Will be logging locally.")
        run = wandb.init(project=cfg.wandb_project_name, mode="offline")
    wandb.run.log_code("..")
    wandb.config.update(
        omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )

    print("devices: ", *jax.devices())

    target_system = hydra.utils.instantiate(cfg.target_system)

    run.tags = run.tags + (f"{type(target_system).__name__}",)
    key = jax.random.PRNGKey(cfg.PRNGKey)

    sampler = Canonical_Sampler(target_system)
    key1, key2, key = jax.random.split(key, 3)
    sampler.sample(key=key1, U=target_system.U0, data_path=f"../data/{cfg.data0}")
    sampler.sample(key=key2, U=target_system.U1, data_path=f"../data/{cfg.data1}")
    dataloader_train = paired_dataloader(cfg)

    print(80 * "-")

    SI = hydra.utils.instantiate(cfg.model)
    SI.num_features = target_system.num_dim
    SI.U_model = hydra.utils.instantiate(cfg.U_model)
    SI.init_params(key=key)
    key = jax.random.split(key, 2)[0]

    num_params = sum(x.size for x in jax.tree_util.tree_leaves(SI.params))
    print(f"num params: {num_params/1000:.1f}K")

    ## train
    optim = hydra.utils.instantiate(cfg.optim)
    opt_state = optim.init(SI.params)
    params = SI.params

    target_name = cfg.target_system._target_.split(".")[-1]
    if "model_name" in cfg.keys():
        target_name += f'.{cfg["model_name"]}'

    def update_step(key, params, batch_pair, opt_state):
        grad_fn = jax.grad(SI.loss_fn, has_aux=True)
        grad, aux = grad_fn(params, batch_pair, key)
        updates, opt_state = optim.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, grad, opt_state, aux

    @jax.jit
    def body_fun(i, carry):
        params, _, opt_state, key, dUdt, _, data = carry
        key, _ = jax.random.split(key)
        params, grad, opt_state, aux = update_step(key, params, data, opt_state)
        _, _, dUdt_ = aux
        dUdt = dUdt.at[i].set(dUdt_)

        return params, grad, opt_state, key, dUdt, aux[:2], data

    start = time.time()
    dUdt_log = []
    step = opt_state[0].count
    while step < cfg.num_train_steps:
        key1, key2 = jax.random.split(key)
        batch = dataloader_train.next(key1)
        step = opt_state[0].count
        if step > 0 and step % cfg.ckpt_every_n_step == 0:
            ckpt_dict = {"params": params, "opt_state": opt_state, "cfg": cfg}
            save_and_push(ckpt_dict, "LJsolute.model")
            save_and_push({"train": jnp.stack(dUdt_log)}, "dUdt_hist")

        carry = (
            params,
            params,
            opt_state,
            key2,
            jnp.zeros(cfg.new_batch_every_n_step),
            (0, 0),
            batch,
        )
        params, grad, opt_state, key, dUdt, aux, _ = jax.lax.fori_loop(
            0, cfg.new_batch_every_n_step, body_fun, carry
        )
        score_loss, dudt_loss = aux
        dUdt_log += [dUdt]

        def tree_flat(tree):
            tree = jax.tree.map(lambda a: a.reshape(-1), tree)
            tree = jax.tree.flatten(tree)[0]
            tree = jnp.concatenate(tree)
            return tree

        params_flat = tree_flat(params)
        grad_flat = tree_flat(grad)
        wandb.log(
            {
                "grad steps": opt_state[0].count,
                "score loss": score_loss,
                "dudt loss": dudt_loss,
                "running dF estimate": dUdt.mean(),
                "params norm": (params_flat**2).mean(),
                "params max": (jnp.abs(params_flat)).max(),
                "grad norm": (grad_flat**2).mean(),
                "grad max": (jnp.abs(grad_flat)).max(),
            }
        )
        print(f"training progress:  {step/cfg.num_train_steps:.3f}", end="    ")
        print(f"time: {convert_seconds(time.time()-start)}", end="    ")
        print(f"score loss: {score_loss:.1f}", end="    \r")
    print()
    print(80 * "=")
    run.finish()


def save_and_push(dict, filename):
    file = open(filename, "wb")
    pickle.dump(dict, file)
    file.close()
    wandb.log_artifact(filename)


def convert_seconds(seconds):
    seconds = int(seconds)
    hours = seconds // 3600  # 1 hour = 3600 seconds
    minutes = (seconds % 3600) // 60  # Remaining seconds converted to minutes
    secs = seconds % 60  # Remaining seconds after extracting hours and minutes

    result = []

    if hours > 0:
        result.append(f"{hours:0}h")
    if minutes > 0:
        result.append(f"{minutes:0}m")
    result.append(f"{secs:0}s")

    return " ".join(result)


if __name__ == "__main__":
    main()

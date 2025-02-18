import jax
import jax.numpy as jnp
from dataclasses import dataclass
import os, pickle
from distance_on_torus import dist2_on_torus_matrix
import wandb
import optax


@dataclass
class Canonical_Sampler:
    def __init__(self, target_system):
        self.target_system = target_system

    def propose(self, x, key):
        z = jax.random.normal(key, x.shape) * jnp.sqrt(
            2 * self.target_system.sampling_dt
        )
        x = x + z
        x = x % 1
        return x

    def sample(self, key, U, data_path):
        if os.path.isfile(data_path):
            return

        x0 = jax.random.uniform(
            key,
            shape=(
                self.target_system.num_walkers,
                self.target_system.N,
                self.target_system.num_dim,
            ),
        )

        def loss_one(x):
            D = dist2_on_torus_matrix(x)
            mask = 1 - jnp.eye(self.target_system.N)
            return ((1 / (D + 1e-4)) * mask).sum()

        def loss(x):
            return jax.vmap(loss_one)(x).mean()

        optim = optax.adam(learning_rate=1e-4)
        opt_state = optim.init(x0)

        @jax.jit
        def move(x0, opt_state):
            grad = jax.grad(loss)(x0)
            updates, opt_state = optim.update(grad, opt_state, x0)
            x0 = optax.apply_updates(x0, updates) % 1
            D = jax.vmap(dist2_on_torus_matrix)(x0)
            mask = jnp.eye(self.target_system.N).reshape(
                1, self.target_system.N, self.target_system.N
            )
            D = D / self.target_system.sigma**2
            D_min = (D + mask).min()
            return D_min, x0, opt_state

        D_min = 0
        while D_min < 0.7:
            D_min, x0, opt_state = move(x0, opt_state)

        optim = optax.adam(learning_rate=1e-4)
        opt_state = optim.init(x0)

        x_curr = x0
        samples_list = []
        i = 0

        @jax.jit
        def body_fn(i, carry):
            x_traj, U_traj, acc_prob_traj, key = carry
            key1, key2 = jax.random.split(key)
            x_curr = jax.tree.map(lambda x: x[i], x_traj)
            U_curr = U_traj[i]
            x_prop = jax.vmap(self.propose)(
                x_curr, jax.random.split(key1, self.target_system.num_walkers)
            )
            U_prop = jax.vmap(U)(x_prop)
            U_diff = U_prop - U_curr
            acc_prob = jnp.exp(-U_diff)
            acc_prob_traj = acc_prob_traj.at[i + 1].set(jnp.clip(acc_prob, max=1))
            take_new = (
                jax.random.uniform(key2, (self.target_system.num_walkers,)) < acc_prob
            )

            U_new = take_new * U_prop + (1 - take_new) * U_curr
            U_traj = U_traj.at[i + 1].set(U_new)
            take_new = take_new.reshape(self.target_system.num_walkers, 1, 1)
            x_new = take_new * x_prop + (1 - take_new) * x_curr
            x_traj = x_traj.at[i + 1].set(x_new)

            key = jax.random.split(key)[0]
            return x_traj, U_traj, acc_prob_traj, key

        NUM_TO_SAMPLE = self.target_system.num_samples + self.target_system.burn_in
        while i < NUM_TO_SAMPLE:
            N = 100 * self.target_system.keep_every_n_step
            x_traj = jnp.zeros((N,) + x_curr.shape)
            x_traj = x_traj.at[0].set(x_curr)
            U_traj = jnp.zeros((N, self.target_system.num_walkers))
            U_traj = U_traj.at[0].set(jax.vmap(U)(x_curr))
            acc_prob_traj = jnp.zeros_like(U_traj)
            carry = (x_traj, U_traj, acc_prob_traj, key)
            x_traj, U_traj, acc_prob_traj, key = jax.lax.fori_loop(
                0, N - 1, body_fn, carry
            )
            x_curr = x_traj[-1]

            if wandb.run is not None:
                wandb.log(
                    {
                        f"acceptance rate/{data_path}": acc_prob_traj.mean(),
                        f"U/{data_path}": U_traj.mean(),
                    }
                )
            print(f"Sampling progress {i/NUM_TO_SAMPLE:.2f}", end="\r")
            i += N
            if i > self.target_system.burn_in:
                samples_list.append(x_traj[::100].reshape((-1,) + x_curr.shape[1:]))

        i = 0

        samples = jnp.concatenate(samples_list)
        key = jax.random.split(key)[0]
        samples = jax.random.permutation(key, samples, axis=0)
        with open(data_path, "wb") as file:
            pickle.dump(samples, file)
        if wandb.run is not None:
            wandb.log({f"acceptance rate/{data_path}": acc_prob_traj.mean()})

    def __hash__(self):
        return 0

    def __eq__(self, other):
        return True

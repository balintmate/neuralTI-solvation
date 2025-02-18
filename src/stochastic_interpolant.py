import jax
import jax.numpy as jnp
from functools import partial
from U_GNN import MLP


class stochastic_interpolant:
    def __init__(self, target_system, loss_t_weight, loss_x_weight):
        self.target_system = target_system
        self.loss_t_weight = loss_t_weight
        self.loss_x_weight = eval(loss_x_weight)
        self.LJ_soft_model = MLP((256, 256, 4))

    def init_params(self, key):
        key1, key2, key3 = jax.random.split(key, 3)
        params_U = self.U_model.init(
            rngs=key1,
            t=jnp.ones(1),
            x=jax.random.uniform(key2, (47, 3)),
        )

        self.params = {
            "U": params_U,
            "LJ": self.LJ_soft_model.init(key3, jnp.ones(1)),
        }

    def getLJ_softening(self, params, t):
        LJsoft = self.LJ_soft_model.apply(params, t)
        t = t.reshape()
        LJsoft = LJsoft.at[0].set(jnp.exp(LJsoft[0]) * (t * (1 - t)))
        LJsoft = LJsoft.at[2].set(jnp.exp(LJsoft[2]) * (t * (1 - t)))
        LJsoft = LJsoft.at[1].set(jnp.exp(t * LJsoft[1]) * (1 - t))
        LJsoft = LJsoft.at[3].set(jnp.exp(t * LJsoft[3]) * (1 - t))
        return LJsoft

    def U(self, params, x, t):
        U_NN = self.U_model.apply(params["U"], x, t).reshape()
        LJsoft = self.getLJ_softening(params["LJ"], t)
        U_base = self.target_system.U(x, LJsoft)
        return ((t * (1 - t)) * U_NN + U_base).reshape()

    def loss_fn(self, params, batch_pair, key):
        batch0, batch1 = batch_pair
        t = jax.random.uniform(key, (len(batch0), 1))

        def loss_one(x0, x1, t):
            dx = ((x1 - x0) + 0.5) % 1 - 0.5
            xt = x0 + t[0] * dx
            xt = xt % 1
            U, (dUdx, dUdt) = jax.value_and_grad(partial(self.U, params), (0, 1))(xt, t)

            dUdx0 = jax.grad(self.target_system.U0)(x0)
            dUdx1 = jax.grad(self.target_system.U1)(x1)

            # score_loss = (dUdx - (dUdx0 + dUdx1)) ** 2
            TSM_left = (dUdx - dUdx0 / jnp.clip(1 - t[0], min=0.1)) ** 2
            TSM_right = (dUdx - dUdx1 / jnp.clip(t[0], min=0.1)) ** 2
            score_loss = (t[0] < 0.5) * TSM_left + (t[0] >= 0.5) * TSM_right

            return score_loss.mean(), dUdt

        L_score, dudt = jax.vmap(loss_one)(batch0, batch1, t)
        dudt_loss = dudt**2
        loss = (
            self.loss_x_weight(t.reshape(-1)) * L_score
        ).mean() + self.loss_t_weight * (dudt_loss).mean()
        aux = (L_score.mean(), dudt_loss.mean(), dudt.mean())
        return loss, aux

import jax.numpy as jnp
from distance_on_torus import dist2_on_torus_matrix


class LJ3D_with_solute:
    num_dim: int = 3
    sigma: float = 0.111  # box size: 9 sigma
    eps: float = 2.0
    N0: int = 512

    sigma_big_ratio: float = 2
    eps_big_ratio: float = 2

    ### sampling
    data_path: str = "../data/LJ3Dwithsolute"
    num_samples: int = int(1e5)
    keep_every_n_step: int = 100
    burn_in: int = int(2e5)
    sampling_dt: float = 8e-8
    num_walkers: int = 32

    @property
    def sigma_AB(self):
        return 0.5 * (self.sigma + self.sigma_big_ratio * self.sigma)

    @property
    def eps_AB(self):
        return self.eps * jnp.sqrt(self.eps_big_ratio)

    @property
    def N(self):
        return self.N0 + 1

    def U0(self, x):
        return self.U(x, a=jnp.array([0.0, 1.0, 0.0, 1.0]))

    def U1(self, x):
        return self.U(x, a=jnp.array([0.0, 0.0, 0.0, 0.0]))

    def Ut(self, t, x):
        return self.U(x, a=(1 - t) * jnp.array([0.0, 1.0, 0.0, 1.0]))

    def U(self, x, a):

        r2 = dist2_on_torus_matrix(x)
        r2 = r2 + jnp.eye(len(x))
        U = self.U_ij(r2=r2, a=a)

        mask = (1 - jnp.eye(len(x))) * 0.5  # prevent double counting
        interaction = 0.5 * (U * mask).sum()
        return interaction

    def U_ij(self, r2, a):
        assert len(r2.shape) == 2
        ## solute is the 1st particle
        r2 += jnp.eye(len(r2))

        sigma = jnp.full_like(r2, self.sigma)
        sigma = sigma.at[:, 0].set(self.sigma_AB)
        sigma = sigma.at[0, :].set(self.sigma_AB)

        eps = jnp.full_like(r2, self.eps)
        eps = eps.at[:, 0].set(self.eps_AB)
        eps = eps.at[0, :].set(self.eps_AB)

        a_ = jnp.full_like(r2, a[0])
        a_ = a_.at[:, 0].set(a[1])
        a_ = a_.at[0, :].set(a[1])
        sr6 = (sigma**2 / (a_ * sigma**2 + r2)) ** 3

        U_ij = 4 * eps * (sr6**2 - sr6)
        U_mul = jnp.full_like(U_ij, 1 - a[2])
        U_mul = U_mul.at[:, 0].set(1 - a[3])
        U_mul = U_mul.at[0, :].set(1 - a[3])

        return U_ij * U_mul

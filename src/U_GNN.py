import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Any
from distance_on_torus import dR_on_torus_matrix
from functools import partial


class U_model(nn.Module):
    target_system: Any
    NN: Sequence[int]
    cutoff: float
    size_to_pad: int
    num_features: int
    num_vec_features: int
    agg_norm: int
    num_layers: int

    @nn.compact
    def __call__(self, x, t):
        num_dim = x.shape[-1]
        dR = dR_on_torus_matrix(x) / self.cutoff
        D2 = (dR**2).sum(-1)
        D2 = D2 + 10 * jnp.eye(len(x))  # no self loops
        divideBy = (self.target_system.sigma_AB / self.target_system.sigma) ** 2
        D2 = D2.at[:, 0].divide(divideBy)
        D2 = D2.at[0, :].divide(divideBy)
        ################

        edges = jnp.stack(
            jnp.where(
                (D2 < 1),
                size=self.size_to_pad,
                fill_value=-42,
            )
        )
        senders, receivers = edges[0], edges[1]

        edge_dist2 = D2.reshape(-1)[senders * len(D2) + receivers]
        mask_edge = senders != -42
        edge_dR = dR.reshape(-1, num_dim)[senders * len(D2) + receivers]
        edge_dR = jnp.expand_dims(edge_dR, 1)

        ## first particle is the solute
        h = (jnp.arange(len(x)) == 0).reshape(-1, 1)
        h = jnp.concatenate((h, D2[:, 0:1]), -1)

        h_vec = jnp.zeros((len(x), self.num_vec_features, num_dim))

        edge_embedder = nn.Dense(self.num_vec_features, use_bias=False)
        edge_embedder = jax.vmap(edge_embedder, in_axes=-1, out_axes=-1)
        h_edge_vec = jax.vmap(edge_embedder)(edge_dR)

        h_edge = MLP(self.NN + (self.num_features,))(
            jnp.concatenate(
                (
                    edge_dist2.reshape(-1, 1),
                    h[senders],
                    h[receivers],
                    jnp.tile(t.reshape(-1, 1), (len(edge_dist2), 1)),
                ),
                -1,
            )
        )
        h = jax.vmap(nn.Dense(self.num_features, use_bias=False))(
            jnp.concatenate(
                (h.reshape(-1, 2), jnp.tile(t.reshape(-1, 1), (len(h), 1))), -1
            )
        )

        ######
        for _ in range(self.num_layers):
            dh, dh_vec, dh_edge, dh_edge_vec = Layer(self.NN, self.agg_norm)(
                t,
                h,
                h_vec,
                h_edge,
                h_edge_vec,
                edge_dist2,
                edge_dR,
                mask_edge,
                senders,
                receivers,
            )

            h += dh
            h_vec += dh_vec
            h_edge += dh_edge
            h_edge_vec += dh_edge_vec
        return h.mean()


class Layer(nn.Module):
    NN: Sequence[int]
    agg_norm: float

    @nn.compact
    def __call__(
        self,
        t,
        h,
        h_vec,
        h_edge,
        h_edge_vec,
        edge_dist2,
        edge_dR,
        mask_edge,
        senders,
        receivers,
    ):

        inp = jnp.concatenate(
            [
                jnp.einsum("nfx,nfx->nf", h_vec[receivers], h_edge_vec),
                jnp.einsum("nfx,nfx->nf", h_vec[senders], h_edge_vec),
                jnp.einsum("nfx,nfx->nf", h_vec[senders], h_vec[receivers]),
                jnp.einsum("nfx,nfx->nf", h_vec[senders], h_vec[senders]),
                jnp.einsum("nfx,nfx->nf", h_vec[receivers], h_vec[receivers]),
                jnp.einsum("nfx,nfx->nf", h_edge_vec, h_edge_vec),
                h[senders],
                h[receivers],
                h_edge,
            ],
            -1,
        )
        ## Message passing

        message_w_model = MessageWeight(self.NN + (h.shape[1] + h_vec.shape[1],))
        message_w_model = partial(message_w_model, t)
        message_w_model = jax.vmap(message_w_model)

        mw, mw_vec = jnp.split(message_w_model(inp), [h.shape[1]], axis=-1)

        ## smooth cutoff
        cutoff = 0.5 * (jnp.cos(edge_dist2 * jnp.pi) + 1)
        mw = jnp.einsum("nf,n->nf", mw, cutoff)
        mw_vec = jnp.einsum("nf,n->nf", mw_vec, cutoff)

        m = jnp.einsum("efx,ef,e->efx", h_edge_vec, mw_vec, mask_edge)
        h_vec = jnp.zeros(h_vec.shape).at[receivers].add(m) / self.agg_norm

        m = jnp.einsum("ef,ef,e->ef", mw, h[senders], mask_edge)
        h = jnp.zeros(h.shape).at[receivers].add(m) / self.agg_norm

        ########################
        ### atom update
        h = jax.vmap(MLP(self.NN + (h.shape[1],)))(h)
        hvec_update = nn.Dense(h_vec.shape[1], use_bias=False)
        hvec_update = jax.vmap(hvec_update, in_axes=-1, out_axes=-1)
        h_vec = jax.vmap(hvec_update)(h_vec)
        ## edge update
        edge_update = nn.Dense(h_edge_vec.shape[1], use_bias=False)
        edge_update = jax.vmap(edge_update, in_axes=-1, out_axes=-1)
        h_edge_vec = jnp.concatenate((h_edge_vec, h_vec[senders], h_vec[receivers]), 1)
        h_edge_vec = jax.vmap(edge_update)(h_edge_vec)

        h_edge = jnp.concatenate((h_edge, h[senders], h[receivers]), 1)
        h_edge = MLP(self.NN + (h.shape[1],))(h_edge)

        return h, h_vec, h_edge, h_edge_vec


class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat)(x)
            if i != len(self.features) - 1:
                x = nn.swish(x)
        return x


class MessageWeight(nn.Module):
    NN: Sequence[int]

    @nn.compact
    def __call__(self, t, x):
        x = jnp.concatenate((x.reshape(-1), t.reshape(-1)))
        return MLP(self.NN)(x)

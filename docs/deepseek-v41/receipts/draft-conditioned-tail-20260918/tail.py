"""Draft-only suffix from an already-generated native five-token prefix.

The target and its verifier are never called here. Prefix proposals, never
teacher-future tokens, condition the second pass. Draft attention is read-only.
"""
import mlx.core as mx
from mtplx.models import deepseek_v41_dspark as ds


class TailProposal:
    def __init__(self, owner, *, conditioned):
        self.owner = owner
        self.mtp = owner.mtp
        self.width = self.mtp.block_size
        if self.width not in (7, 13):
            raise ValueError('only the two statically bounded draft shapes are installed')
        self.prefix_length = 5
        self.input_ids = self.conditioned_inputs if conditioned else self.masked_inputs
        self.last = self.mtp.layers[-1]
        self.markov = ds._draft_markov_step(self.last.markov_head)
        self.confidence = ds._draft_confidence(self.last.confidence_head)
        self.head_dtype = ds._draft_head_source_dtype(owner.head)

    def conditioned_inputs(self, root_ids, prefix):
        first = root_ids.reshape(1, 1)
        noise = mx.full((1, self.width - 6), self.mtp.layers[0].noise_token_id,
                        dtype=first.dtype)
        return mx.concatenate([first, prefix, noise], axis=1)

    def masked_inputs(self, root_ids, prefix):
        first = root_ids.reshape(1, 1)
        noise = mx.full((1, self.width - 1), self.mtp.layers[0].noise_token_id,
                        dtype=first.dtype)
        return mx.concatenate([first, noise], axis=1)

    def __call__(self, main_hidden, root_ids, prefix, caches):
        draft_input_ids = self.input_ids(root_ids, prefix)
        main_x = self.mtp.layers[0].main_project(main_hidden)
        x = self.owner.model.embed_tokens(draft_input_ids)
        x = mx.broadcast_to(x[:, :, None, :], (1, self.width, self.mtp.hc_mult, x.shape[-1]))
        pre_mix = self.mtp._identity_pre_mix(1, self.width)
        for stage, cache in zip(self.mtp.layers, caches):
            x, pre_mix = stage(x, pre_mix, main_x, cache, seed_only=False)
        # Keep the native first five decisions. Project only the remaining rows;
        # seed their Markov recurrence with the last native proposal.
        x = self.last._hc_pre(x, pre_mix)[:, self.prefix_length:, :]
        hidden_n = ds._rmsnorm(x, self.last.norm_weight, self.last.norm_eps)
        base = self.owner.head(hidden_n.astype(self.head_dtype)).astype(mx.float32)
        previous = [prefix[:, -1]]
        for i in range(self.width - self.prefix_length):
            _, _, nxt = self.markov(previous[-1], base[:, i, :],
                self.last.markov_head.embed.weight, self.last.markov_head.head.weight)
            previous.append(nxt.reshape(1))
        out = mx.stack(previous[1:], axis=1)
        embeds = self.last.markov_head.embed(mx.stack(previous[:-1], axis=1))
        confidence = self.confidence(x, embeds, self.last.confidence_head.proj.weight)
        return out, confidence

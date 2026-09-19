"""One-request adaptive draft lane; target verification remains authoritative."""
import copy
from dataclasses import replace


class AdaptiveDraft:
    def __init__(self, native, *, tree_flatten):
        if native.block_size != 7 or len(native.layers) != 3:
            raise RuntimeError('adaptive draft requires the proved native D7/M8 envelope')
        if any(stage.block_size != 7 for stage in native.layers):
            raise RuntimeError('all native draft stages must be constructed at depth7')
        if 'draft_block' in vars(native) or 'seed_main' in vars(native):
            raise RuntimeError('draft instance already has an installed route')
        parameters = dict(tree_flatten(native.parameters()))
        self.heads = {}
        for width in (3, 5, 7):
            head = copy.copy(native)
            head.args = replace(native.args, dspark_block_size=width)
            head.block_size = width
            head.layers = [copy.copy(stage) for stage in native.layers]
            for stage in head.layers:
                stage.block_size = width
            copied = dict(tree_flatten(head.parameters()))
            if set(copied) != set(parameters) or any(copied[k] is not v for k, v in parameters.items()):
                raise RuntimeError('draft views must share every installed parameter')
            if any(a is b for a, b in zip(head.layers, native.layers)):
                raise RuntimeError('draft stage width owners were not separated')
            self.heads[width] = head
        if native.block_size != 7 or any(stage.block_size != 7 for stage in native.layers):
            raise RuntimeError('draft view construction mutated the source head')
        self._drafts = {width: head.draft_block for width, head in self.heads.items()}
        self._seed = self.heads[7].seed_main
        self.reset()

    def reset(self):
        self.width = 5
        self._draft = self._drafts[5]

    def draft_block(self, *args):
        return self._draft(*args)

    def seed_main(self, hidden, caches):
        self._seed(hidden, caches)
        # The existing commit path passes exactly primary + accepted drafts.
        # This is the sole causal policy decision; no future tokens are used.
        self.width = 7 if hidden.shape[1] == self.width + 1 else 3
        self._draft = self._drafts[self.width]

    def close(self):
        self._seed = self._draft = None
        self._drafts.clear()
        self.heads.clear()


def installed_draft_length(conf, k, threshold):
    # The entry boundary proves max depth7, no confidence trimming and fixed
    # width views. The chosen proposal shape is the runtime width authority.
    return conf.shape[-1]


def install(model, decode, *, tree_flatten):
    native = model.mtp
    lane = AdaptiveDraft(native, tree_flatten=tree_flatten)
    original_cycles = decode._decode_cycles
    original_length = decode._effective_draft_len
    used = False

    def cycles(**kwargs):
        nonlocal used
        if used or kwargs['model'] is not model or kwargs['k_request'] != 7:
            raise RuntimeError('adaptive lane is installed for one depth7 request')
        if kwargs['confidence_threshold'] is not None or float(kwargs['sampler'].temperature) != 0:
            raise RuntimeError('adaptive lane requires native greedy verification without confidence trimming')
        if kwargs['verify_chunks'] not in (None, (8,), [8]):
            raise RuntimeError('adaptive lane requires the existing single M<=8 verify route')
        used = True
        native.draft_block = lane.draft_block
        native.seed_main = lane.seed_main
        decode._effective_draft_len = installed_draft_length
        try:
            return original_cycles(**kwargs)
        finally:
            del native.draft_block
            del native.seed_main
            decode._effective_draft_len = original_length
            decode._decode_cycles = original_cycles
            lane.close()

    decode._decode_cycles = cycles
    return {'policy': 'previous-full-acceptance-depth7-else-depth3',
            'initial_depth': 5, 'maximum_depth': 7, 'draft_widths': [3, 5, 7],
            'parameter_storage': 'all views alias the installed compact native parameters',
            'target_arithmetic': 'unchanged native verification',
            'scope': 'one greedy request; installed after weight validation; prefill uses native methods'}

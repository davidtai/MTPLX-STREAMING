"""Minimal DeepSeek-V4.1 model test double for the W3 loader tests.

Satisfies exactly the interface the streaming loader requires of
``mtplx.models.deepseek_v41`` (worker W1), and nothing more, so
``bind_streamed_switches`` and the resident construction path can run end to end
on the real artifact manifest with no GPU and no real weights:

- ``Model(model_args, *, engram_bank_path=None)`` -- the constructor contract,
  including the engram bank-path keyword argument the loader passes.
- ``ModelArgs.from_dict(config)`` -- built from the checkpoint config dict.
- ``model.model.layers[i].mlp.switch_mlp`` for every routed layer i -- the seam
  ``bind_streamed_switches`` reassigns (it walks ``model.model.layers`` first,
  then ``model.layers``; each routed layer must expose ``.mlp`` with a
  ``switch_mlp`` attribute).  The placeholder switch is the real
  ``UnboundExpertSwitch`` hy3 installs pre-binding.

NOTE on naming: DeepSeek's *tensor* namespace uses ``layers.N.ffn.*`` for the
MoE block, but the MTPLX runtime seam ``bind_streamed_switches`` requires is
``layer.mlp.switch_mlp`` (the hy3 convention).  W1's real model therefore
exposes the routed FFN as ``layer.mlp`` with a ``switch_mlp`` seam (its
``sanitize`` remaps the ``ffn.*`` resident tensor keys accordingly), and this
double mirrors that.
"""

from __future__ import annotations

from mtplx.models.expert_mlx import UnboundExpertSwitch


class ModelArgs:
    def __init__(self, *, num_hidden_layers: int, **extra) -> None:
        self.num_hidden_layers = int(num_hidden_layers)
        for key, value in extra.items():
            setattr(self, key, value)

    @classmethod
    def from_dict(cls, config: dict) -> "ModelArgs":
        text = config.get("text_config") or config
        return cls(num_hidden_layers=int(text.get("num_hidden_layers", 40)))


class _SwitchMLP:
    """Stand-in MoE block exposing the ``switch_mlp`` seam."""

    def __init__(self, layer_index: int) -> None:
        self.switch_mlp = UnboundExpertSwitch(layer_index)


class _Layer:
    def __init__(self, layer_index: int) -> None:
        self.mlp = _SwitchMLP(layer_index)


class _Inner:
    def __init__(self, num_layers: int) -> None:
        self.layers = [_Layer(i) for i in range(num_layers)]


class Model:
    def __init__(self, args: ModelArgs, *, engram_bank_path=None) -> None:
        self.args = args
        self.model = _Inner(args.num_hidden_layers)
        # Exposes the engram bank path handed in by the loader constructor arg.
        self.engram_bank_path = engram_bank_path

    @property
    def layers(self):
        return self.model.layers


def model_classes() -> tuple[type, type]:
    """A ``model_class_resolver`` returning (Model, ModelArgs) for the double."""

    return Model, ModelArgs

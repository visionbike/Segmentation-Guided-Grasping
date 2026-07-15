# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Inference-only subset of DETR's util/misc.py.

Kept: NestedTensor (used by backbone.py / position_encoding.py) and
is_main_process (backbone.py passes it as the `pretrained` flag when
building the torchvision ResNet).

Dropped from the original: MetricLogger / SmoothedValue, all_gather,
reduce_dict, collate_fn, nested_tensor_from_tensor_list, distributed
setup, accuracy, interpolate — all training/eval-time helpers that the
ACT inference path never touches.
"""
from typing import Optional, OrderedDict
import torch.distributed as dist
from torch import Tensor
from torch import nn


class IntermediateLayerGetter(nn.ModuleDict):
    """Module wrapper that returns intermediate layers from a model.

    Vendored from torchvision.models._utils (BSD-3) to avoid importing a
    private torchvision module. State-dict keys are identical because
    nn.ModuleDict registers children under their original names.
    """
    _version = 2
    __annotations__ = {"return_layers": dict[str, str]}

    def __init__(self, model: nn.Module, return_layers: dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

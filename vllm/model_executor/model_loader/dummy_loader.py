# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch.nn as nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.reload.layerwise import (
    _layerwise_process,
    get_layerwise_info,
)
from vllm.model_executor.model_loader.reload.utils import get_layer_tensors
from vllm.model_executor.model_loader.weight_utils import (
    initialize_single_dummy_weight,
)


class DummyModelLoader(BaseModelLoader):
    """Model loader that will set model weights to random values."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )

    def download_model(self, model_config: ModelConfig) -> None:
        pass  # Nothing to download

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        # NOTE(woosuk): For accurate performance evaluation, we assign
        # random values to the weights.
        for layer in model.modules():
            info = get_layerwise_info(layer)
            if info.can_load():
                # Online quant layer (meta device): materialize the layer,
                # apply dummy weights, and run quantization processing.
                _layerwise_process(layer, info)
            else:
                # Regular layer: apply dummy weights to direct parameters.
                for tensor in get_layer_tensors(layer).values():
                    initialize_single_dummy_weight(tensor)

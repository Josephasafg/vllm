# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, replace

import torch

from vllm.v1.attention.backend import AttentionBackend, CommonAttentionMetadata
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)


class Mamba1AttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "MAMBA1_ATTN"

    @staticmethod
    def get_builder_cls() -> type["Mamba1AttentionMetadataBuilder"]:
        return Mamba1AttentionMetadataBuilder


@dataclass
class Mamba1AttentionMetadata(BaseMambaAttentionMetadata):
    # Chunk-related metadata (only for prefill with APC)
    cu_chunk_seqlen_p: torch.Tensor | None = None
    last_chunk_indices_p: torch.Tensor | None = None


class Mamba1AttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[Mamba1AttentionMetadata]
):
    metadata_cls = Mamba1AttentionMetadata
    supports_update_block_table: bool = False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Mamba1AttentionMetadata:
        common = self._compute_common_metadata(common_attn_metadata)

        cu_chunk_seqlen_p = None
        last_chunk_indices_p = None

        if (
            common.num_prefills > 0
            and self.vllm_config.cache_config.mamba_cache_mode == "all"
        ):
            cu_chunk_seqlen_p, _, last_chunk_indices_p = (
                self._build_chunk_metadata_tensors(
                    self.kv_cache_spec.block_size, common,
                    common_attn_metadata,
                )
            )

        return replace(
            common,
            cu_chunk_seqlen_p=cu_chunk_seqlen_p,
            last_chunk_indices_p=last_chunk_indices_p,
        )

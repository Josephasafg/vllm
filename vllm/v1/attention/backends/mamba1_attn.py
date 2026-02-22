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
            num_reqs = common.num_reqs
            num_prefills = common.num_prefills
            num_decode_tokens = common.num_decode_tokens

            num_computed_tokens_cpu = (
                common_attn_metadata.compute_num_computed_tokens().cpu()
            )
            num_computed_tokens_p_cpu = num_computed_tokens_cpu[
                num_reqs - num_prefills : num_reqs
            ]
            query_start_loc_p_cpu = (
                common_attn_metadata.query_start_loc_cpu[-num_prefills - 1 :]
                - num_decode_tokens
            )

            chunk_size = self.kv_cache_spec.block_size
            cu_chunk_seqlen, _, last_chunk_indices = (
                self._compute_chunk_metadata(
                    chunk_size,
                    num_prefills,
                    num_computed_tokens_p_cpu,
                    query_start_loc_p_cpu,
                )
            )

            device = common_attn_metadata.query_start_loc.device
            cu_chunk_seqlen_p = torch.as_tensor(
                cu_chunk_seqlen, device=device, dtype=torch.int32,
            )
            last_chunk_indices_p = torch.as_tensor(
                last_chunk_indices, device=device, dtype=torch.int32,
            )

        return replace(
            common,
            cu_chunk_seqlen_p=cu_chunk_seqlen_p,
            last_chunk_indices_p=last_chunk_indices_p,
        )

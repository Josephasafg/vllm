# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Optional

import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.utils import cdiv
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadataBuilder)
from vllm.v1.attention.backends.utils import (CommonAttentionMetadata,
                                              split_decodes_and_prefills)


class Mamba1AttentionBackend(AttentionBackend):

    @staticmethod
    def get_builder_cls() -> type["Mamba1AttentionMetadataBuilder"]:
        return Mamba1AttentionMetadataBuilder


@dataclass
class Mamba1AttentionMetadata:
    query_start_loc: torch.Tensor
    context_lens_tensor: torch.Tensor
    state_indices_tensor: torch.Tensor
    has_initial_states: Optional[torch.Tensor]
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_padded_decodes: int
    cache_spec: Optional[object] = None
    seq_lens: Optional[torch.Tensor] = None
    current_first_idx_d: Optional[torch.Tensor] = None
    current_first_idx_p: Optional[torch.Tensor] = None
    current_last_idx_d: Optional[torch.Tensor] = None
    current_last_idx_p: Optional[torch.Tensor] = None
    last_computed_offset_d: Optional[torch.Tensor] = None
    last_computed_offset_p: Optional[torch.Tensor] = None
    seq_lens_completed_d: Optional[torch.Tensor] = None
    seq_lens_completed_p: Optional[torch.Tensor] = None
    last_state_idx_d: Optional[torch.Tensor] = None
    last_state_idx_p: Optional[torch.Tensor] = None

class Mamba1AttentionMetadataBuilder(
        BaseMambaAttentionMetadataBuilder[Mamba1AttentionMetadata]):
    
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Mamba1AttentionMetadata:
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens

        state_indices_tensor = common_attn_metadata.block_table_tensor[:, 0]
        context_lens_tensor = common_attn_metadata.num_computed_tokens_cpu.to(
            query_start_loc.device)
        


        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold))

        has_initial_states = None
        padded_decodes = num_decodes
        mamba_block_size = self.kv_cache_spec.block_size

        if self.kv_cache_spec.cache_strategy == "disabled":
            # Always return just a single block per each request:
            state_indices_tensor = common_attn_metadata.block_table_tensor[:,
                                                                           0]
            current_first_idx_d = None
            current_first_idx_p = None
            current_last_idx_d = None
            current_last_idx_p = None
            last_computed_offset_d = None
            last_computed_offset_p = None
            seq_lens_completed_d = None
            seq_lens_completed_p = None
            last_state_idx_d = None
            last_state_idx_p = None
        else:
            # Return a tensor of shape (#requests, #blocks for longest request)
            # filled in with cached and newly allocated blocks for each request
            state_indices_tensor = common_attn_metadata.block_table_tensor
        
            seq_lens_pending = (
                torch.roll(query_start_loc, -1, -1) -
                query_start_loc)[:-1]
            seq_lens_completed = (seq_lens - seq_lens_pending)
            last_computed_token_block_offset = seq_lens_completed % mamba_block_size
            # e.g. 16 blocks computed; 0th based indexing -> state[15]
            last_computed_token_block_idx = \
                seq_lens_completed // mamba_block_size - 1
            # -1 in case it's non-computed and causes later issues with indexing
            last_computed_token_block_idx = last_computed_token_block_idx.clamp(
                min=0)
            current_first_token_block_idx = cdiv(seq_lens_completed + 1,
                                                 mamba_block_size) - 1
            current_last_token_block_idx = cdiv(
                seq_lens_completed + seq_lens_pending, mamba_block_size) - 1

            seq_lens_completed_d, seq_lens_completed_p = torch.split(
                seq_lens_completed, [num_decodes, num_prefills],
                dim=0)
            last_state_idx_d, last_state_idx_p = torch.split(
                last_computed_token_block_idx, [num_decodes, num_prefills],
                dim=0)
            last_computed_offset_d, last_computed_offset_p = torch.split(
                last_computed_token_block_offset, [num_decodes, num_prefills],
                dim=0)
            current_first_idx_d, current_first_idx_p = torch.split(
                current_first_token_block_idx, [num_decodes, num_prefills],
                dim=0)
            current_last_idx_d, current_last_idx_p = torch.split(
                current_last_token_block_idx, [num_decodes, num_prefills],
                dim=0)

        if num_prefills > 0:
            has_initial_states = context_lens_tensor > 0
        elif (num_decodes > 0 and num_decodes <= self.decode_cudagraph_max_bs
              and self.compilation_config.full_cuda_graph):
            state_indices_for_decode = state_indices_tensor[:num_decodes]
            padded_decodes = self.vllm_config.pad_for_cudagraph(num_decodes)
            self.state_indices_tensor[:num_decodes].copy_(
                state_indices_for_decode, non_blocking=True)
            state_indices_tensor = self.state_indices_tensor[:padded_decodes]
            state_indices_tensor[num_decodes:] = PAD_SLOT_ID

        return Mamba1AttentionMetadata(
            query_start_loc=query_start_loc,
            context_lens_tensor=context_lens_tensor,
            has_initial_states=has_initial_states,
            state_indices_tensor=state_indices_tensor,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_padded_decodes=padded_decodes,
            cache_spec=self.kv_cache_spec,
            seq_lens=seq_lens,
            current_first_idx_d=current_first_idx_d,
            current_first_idx_p=current_first_idx_p,
            current_last_idx_d=current_last_idx_d,
            current_last_idx_p=current_last_idx_p,
            last_computed_offset_d=last_computed_offset_d,
            last_computed_offset_p=last_computed_offset_p,
            seq_lens_completed_d=seq_lens_completed_d,
            seq_lens_completed_p=seq_lens_completed_p,
            last_state_idx_d=last_state_idx_d,
            last_state_idx_p=last_state_idx_p,
        )

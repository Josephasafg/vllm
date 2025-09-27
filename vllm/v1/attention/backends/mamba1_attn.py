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
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec
from vllm.config import VllmConfig


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
    
    current_last_token_block_idx: torch.Tensor | None
    current_first_token_block_idx: torch.Tensor | None
    last_computed_token_block_idx: torch.Tensor | None
    seq_lens_completed: torch.Tensor | None
    last_computed_token_block_offset: torch.Tensor | None
    cache_spec: Optional[object] = None
    seq_lens: Optional[torch.Tensor] = None


class Mamba1AttentionMetadataBuilder(
        BaseMambaAttentionMetadataBuilder[Mamba1AttentionMetadata]):
    
    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],
                 vllm_config: VllmConfig, device: torch.device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        assert isinstance(kv_cache_spec, MambaSpec)
        if kv_cache_spec.cache_strategy == "all":
            self.state_indices_tensor = torch.empty(
                (self.decode_cudagraph_max_bs,
                 cdiv(vllm_config.model_config.max_model_len,
                      kv_cache_spec.block_size)),
                dtype=torch.int32,
                device=device,
            )
            self.current_last_token_block_idx = torch.empty(
                (self.decode_cudagraph_max_bs, ),
                dtype=torch.int32,
                device=device,
            )
            self.current_first_token_block_idx = torch.empty(
                (self.decode_cudagraph_max_bs, ),
                dtype=torch.int32,
                device=device,
            )
            self.last_computed_token_block_idx = torch.empty(
                (self.decode_cudagraph_max_bs, ),
                dtype=torch.int32,
                device=device,
            )
            self.seq_lens_completed = torch.empty(
                (self.decode_cudagraph_max_bs, ),
                dtype=torch.int32,
                device=device,
            )
            self.last_computed_token_block_offset = torch.empty(
                (self.decode_cudagraph_max_bs, ),
                dtype=torch.int32,
                device=device,
            )
    
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
            current_last_token_block_idx = None
            current_first_token_block_idx = None
            last_computed_token_block_idx = None
            last_computed_token_block_offset = None
            seq_lens_completed = None
        else:
            # Return a tensor of shape (#requests, #blocks for longest request)
            # filled in with cached and newly allocated blocks for each request
            state_indices_tensor = common_attn_metadata.block_table_tensor
            mamba_block_size = self.kv_cache_spec.block_size
            seq_lens_pending = (
                torch.roll(common_attn_metadata.query_start_loc, -1, -1) -
                common_attn_metadata.query_start_loc)[:-1]
            seq_lens_completed = common_attn_metadata.seq_lens - \
                                 seq_lens_pending
            last_computed_token_block_offset = \
                seq_lens_completed % mamba_block_size
            
            # Indices: last_computed <= current_first <= current_last
            # Cases:
            #  last_computed == current_first  if last state was partially
            #                                  computed and needs to be updated
            #  current_first == current_last   if no block crossing occurs, and
            #                                  only one state will be stored
            # 0th based indexing leads to "-1" -> e.g. 16 computed -> state[15]:
            current_last_token_block_idx = cdiv(
                seq_lens_completed + seq_lens_pending, mamba_block_size) - 1
            current_first_token_block_idx = cdiv(seq_lens_completed + 1,
                                                 mamba_block_size) - 1
            last_computed_token_block_idx = cdiv(seq_lens_completed,
                                                 mamba_block_size) - 1
            # -1 in case it's non-computed and causes later issues with indexing
            last_computed_token_block_idx = \
                last_computed_token_block_idx.clamp(min=0)
        

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

            if self.kv_cache_spec.cache_strategy != 'disabled':
                self.current_last_token_block_idx[:num_decodes].copy_(
                    current_last_token_block_idx, non_blocking=True)
                current_last_token_block_idx = \
                    self.current_last_token_block_idx[:padded_decodes]
                current_last_token_block_idx[num_decodes:] = 0

                self.current_first_token_block_idx[:num_decodes].copy_(
                    current_first_token_block_idx, non_blocking=True)
                current_first_token_block_idx = \
                    self.current_first_token_block_idx[:padded_decodes]
                current_first_token_block_idx[num_decodes:] = 0

                self.last_computed_token_block_idx[:num_decodes].copy_(
                    last_computed_token_block_idx, non_blocking=True)
                last_computed_token_block_idx = \
                    self.last_computed_token_block_idx[:padded_decodes]
                last_computed_token_block_idx[num_decodes:] = 0

                self.seq_lens_completed[:num_decodes].copy_(seq_lens_completed,
                                                            non_blocking=True)
                seq_lens_completed = self.seq_lens_completed[:padded_decodes]
                seq_lens_completed[num_decodes:] = 0

                self.last_computed_token_block_offset[:num_decodes].copy_(
                    last_computed_token_block_offset, non_blocking=True)
                last_computed_token_block_offset = \
                    self.last_computed_token_block_offset[:padded_decodes]
                last_computed_token_block_offset[num_decodes:] = 0

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
            current_last_token_block_idx=current_last_token_block_idx,
            current_first_token_block_idx=current_first_token_block_idx,
            last_computed_token_block_idx=last_computed_token_block_idx,
            seq_lens_completed=seq_lens_completed,
            last_computed_token_block_offset=last_computed_token_block_offset,
        )

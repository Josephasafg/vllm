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
    
    # last_chunk_indices_p is a tensor of shape (batch,) that contains the
    # index of the last chunk for every sequence in the (prefill) batch.
    last_chunk_indices_p: Optional[torch.Tensor]

    state_indices_tensor: torch.Tensor  # shape: [batch,]
    current_last_idx: torch.Tensor
    current_first_idx_p: torch.Tensor
    last_state_idx: torch.Tensor
    context_lens_p: torch.Tensor
    last_computed_offset_p: torch.Tensor


class Mamba1AttentionMetadataBuilder(
        BaseMambaAttentionMetadataBuilder[Mamba1AttentionMetadata]):
    
    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],
                 vllm_config: VllmConfig, device: torch.device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        assert isinstance(kv_cache_spec, MambaSpec)
        if self.vllm_config.cache_config.enable_prefix_caching:
            self.state_indices_tensor = torch.empty(
                (self.decode_cudagraph_max_bs,
                 cdiv(vllm_config.model_config.max_model_len,
                      kv_cache_spec.block_size)),
                dtype=torch.int32,
                device=device,
            )
            self.current_last_idx = torch.empty(
                (self.decode_cudagraph_max_bs, ),
                dtype=torch.int32,
                device=device,
            )
            self.last_state_idx = torch.empty(
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
        num_reqs = common_attn_metadata.num_reqs
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

        if self.vllm_config.cache_config.enable_prefix_caching:
            # Return a tensor of shape (#requests, #max blocks)
            state_indices_tensor = common_attn_metadata.block_table_tensor

            # Additional cache-related varaiables:
            mamba_block_size = self.kv_cache_spec.block_size
            seq_lens_pending = (
                torch.roll(common_attn_metadata.query_start_loc, -1, -1) -
                common_attn_metadata.query_start_loc)[:-1]
            context_lens = common_attn_metadata.seq_lens - \
                                 seq_lens_pending
            last_computed_offset = \
                context_lens % mamba_block_size
            # Indices: last_computed <= current_first <= current_last
            # Cases:
            #  last_computed == current_first  if last state was partially
            #                                  computed and needs to be updated
            #  current_first == current_last   if no block crossing occurs, and
            #                                  only one state will be stored
            # 0th based indexing leads to "-1" -> e.g. 16 computed -> state[15]:
            current_last_idx = cdiv(context_lens + seq_lens_pending,
                                    mamba_block_size) - 1
            current_first_idx = cdiv(context_lens + 1, mamba_block_size) - 1
            last_state_idx = cdiv(context_lens, mamba_block_size) - 1
            # -1 in case it's non-computed and causes later issues with indexing
            last_state_idx = \
                last_state_idx.clamp(min=0)
        else:
            # Always return just a single block per each request:
            state_indices_tensor = common_attn_metadata.block_table_tensor[:,
                                                                           0]
            current_last_idx = None
            last_state_idx = None

        if num_prefills > 0:
            has_initial_states = context_lens_tensor > 0

            if self.vllm_config.cache_config.enable_prefix_caching:
                assert context_lens is not None
                context_lens_p = context_lens[num_reqs - num_prefills:num_reqs]
                assert last_computed_offset is not None
                last_computed_offset_p = last_computed_offset[
                    num_reqs - num_prefills:num_reqs]
                assert current_first_idx is not None
                current_first_idx_p = current_first_idx[num_reqs -
                                                        num_prefills:num_reqs]

        elif (num_decodes > 0 and num_decodes <= self.decode_cudagraph_max_bs
              and self.compilation_config.full_cuda_graph):
            state_indices_for_decode = state_indices_tensor[:num_decodes]
            padded_decodes = self.vllm_config.pad_for_cudagraph(num_decodes)
            self.state_indices_tensor[:num_decodes].copy_(
                state_indices_for_decode, non_blocking=True)
            state_indices_tensor = self.state_indices_tensor[:padded_decodes]
            state_indices_tensor[num_decodes:] = PAD_SLOT_ID

            if self.vllm_config.cache_config.enable_prefix_caching:
                self.current_last_idx[:num_decodes].copy_(current_last_idx,
                                                          non_blocking=True)
                current_last_idx = \
                    self.current_last_idx[:padded_decodes]
                current_last_idx[num_decodes:] = 0

                self.last_state_idx[:num_decodes].copy_(last_state_idx,
                                                        non_blocking=True)
                last_state_idx = \
                    self.last_state_idx[:padded_decodes]
                last_state_idx[num_decodes:] = 0

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
            current_last_idx=current_last_idx,
            current_first_idx_p=current_first_idx_p,
            last_state_idx=last_state_idx,
            context_lens_p=context_lens_p,
            last_computed_offset_p=last_computed_offset_p,
        )

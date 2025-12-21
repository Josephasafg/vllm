# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import abc
import copy
from dataclasses import dataclass
from typing import ClassVar, TypeVar

import torch

from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import (
    PAD_SLOT_ID,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    compute_causal_conv1d_metadata,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

# Default chunk alignment for Mamba1 models that don't have explicit chunk_size.
# This alignment is needed for proper state handling during chunked prefill.
# Mamba2 models will override this with their model config chunk_size.
DEFAULT_MAMBA_CHUNK_ALIGNMENT = 8

M = TypeVar("M", bound="BaseMambaAttentionMetadata")


@dataclass
class BaseMambaAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_reqs: int

    # The following tensors only contain prefill requests and will be None if
    # the batch has no prefill request.
    has_initial_states_p: torch.Tensor | None
    query_start_loc_p: torch.Tensor | None
    num_computed_tokens_p: torch.Tensor | None

    state_indices_tensor: torch.Tensor

    # The following tensors are only used for prefix caching and are None if disabled
    block_idx_last_scheduled_token: torch.Tensor | None
    block_idx_first_scheduled_token_p: torch.Tensor | None
    block_idx_last_computed_token: torch.Tensor | None

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None

    # Chunk alignment fields for proper state handling during chunked prefill.
    # These ensure state is saved/loaded at consistent chunk boundaries.
    chunk_size: int = 0
    prep_initial_states: bool = False
    # cu_chunk_seqlen_p: cumulative chunk sequence lengths (nchunks+1,)
    # seq_idx_p: sequence index for each logical chunk (nchunks,)
    # last_chunk_indices_p: index of last chunk for each sequence (batch,)
    cu_chunk_seqlen_p: torch.Tensor | None = None
    seq_idx_p: torch.Tensor | None = None
    last_chunk_indices_p: torch.Tensor | None = None


class BaseMambaAttentionMetadataBuilder(AttentionMetadataBuilder[M], abc.ABC):
    metadata_cls: type[M]
    reorder_batch_threshold: int = 1
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )
    supports_update_block_table: bool = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        assert isinstance(kv_cache_spec, MambaSpec)
        self.compilation_config = vllm_config.compilation_config
        self.decode_cudagraph_max_bs = min(
            self.vllm_config.scheduler_config.max_num_seqs,
            self.compilation_config.max_cudagraph_capture_size,
        )

        # Get chunk_size from model config if available (Mamba2), otherwise use
        # default alignment (Mamba1). Subclasses can override this.
        model_chunk_size = vllm_config.model_config.get_mamba_chunk_size()
        self.chunk_size = (
            model_chunk_size
            if model_chunk_size is not None
            else DEFAULT_MAMBA_CHUNK_ALIGNMENT
        )

        if self.vllm_config.cache_config.enable_prefix_caching:
            self.state_indices_tensor = torch.empty(
                (
                    self.decode_cudagraph_max_bs,
                    cdiv(
                        self.vllm_config.model_config.max_model_len,
                        self.kv_cache_spec.block_size,
                    ),
                ),
                dtype=torch.int32,
                device=device,
            )
            self.block_idx_last_scheduled_token = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
            self.block_idx_last_computed_token = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
        else:
            self.state_indices_tensor = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> M:
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        m = common_attn_metadata

        assert m.num_reqs == m.num_actual_tokens, (
            "Mamba only supports decode-only full CUDAGraph capture. "
            "Make sure all cudagraph capture sizes <= max_num_seq."
        )

        m.max_query_len = 1  # decode-only

        return self.build(0, m)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> M:
        """
        Default build implementation for Mamba-like attention backends.
        Subclasses (e.g., Mamba2) can override to add additional metadata.
        """
        return self._compute_common_metadata(common_attn_metadata)

    def _compute_prefix_caching_block_indices(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        mamba_block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_computed_tokens = common_attn_metadata.num_computed_tokens_cpu.to(
            self.device
        )
        # Block index of the last computed token
        block_idx_last_computed_token = cdiv(num_computed_tokens, mamba_block_size) - 1
        # which is <= block index for the first scheduled token
        block_idx_first_scheduled_token = (
            cdiv(num_computed_tokens + 1, mamba_block_size) - 1
        )
        # which is <= block index of the last scheduled token
        block_idx_last_scheduled_token = (
            cdiv(common_attn_metadata.seq_lens, mamba_block_size) - 1
        )
        # -1 in case it's non-computed and causes later issues with indexing
        block_idx_last_computed_token = block_idx_last_computed_token.clamp(min=0)
        # -1 in the case we have a padded request (0 seq-len)
        block_idx_last_scheduled_token = block_idx_last_scheduled_token.clamp(min=0)

        return (
            block_idx_last_computed_token,
            block_idx_first_scheduled_token,
            block_idx_last_scheduled_token,
        )

    def _compute_chunk_metadata(
        self,
        num_prefills: int,
        num_computed_tokens_p_cpu: torch.Tensor,
        query_start_loc_p_cpu: torch.Tensor,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Compute chunk-aligned metadata for Mamba models.

        This method constructs chunks such that:
        1. Chunks contain tokens from a *single* sequence only.
        2. For every sequence, we can retrieve the mamba state every
           chunk_size tokens, which is critical for proper state handling
           during chunked prefill.

        The chunking handles the interaction between num_computed_tokens
        (which may not be chunk-aligned due to chunked prefill) and the
        new tokens being processed.

        Returns:
            cu_chunk_seqlen: Cumulative chunk sequence lengths (nchunks+1,)
            seq_idx: Sequence index for each logical chunk (nchunks,)
            last_chunk_indices: Index of last chunk for each sequence (batch,)
        """
        cu_chunk_seqlen: list[int] = []
        seq_idx: list[int] = []
        last_chunk_indices: list[int] = []
        seqlen_pos = 0
        chunk_size = self.chunk_size

        for req_idx in range(num_prefills):
            this_num_computed = int(num_computed_tokens_p_cpu[req_idx].item())
            this_new_tokens = int(
                query_start_loc_p_cpu[req_idx + 1].item()
                - query_start_loc_p_cpu[req_idx].item()
            )

            # If computed tokens are not chunk-aligned, use the first
            # chunk to finish it off (bring total to chunk boundary)
            if this_num_computed % chunk_size != 0:
                seq_idx.append(req_idx)
                cu_chunk_seqlen.append(seqlen_pos)
                # How many tokens to finish the chunk?
                chunk_len = (
                    cdiv(this_num_computed, chunk_size) * chunk_size
                    - this_num_computed
                )
                # We can only use at most this_new_tokens
                chunk_len = min(chunk_len, this_new_tokens)
                seqlen_pos += chunk_len
                this_new_tokens -= chunk_len

            # Process remaining tokens in full chunks
            n_chunks = cdiv(this_new_tokens, chunk_size)
            for _ in range(n_chunks):
                seq_idx.append(req_idx)
                cu_chunk_seqlen.append(seqlen_pos)
                chunk_len = min(chunk_size, this_new_tokens)
                seqlen_pos += chunk_len
                this_new_tokens -= chunk_len

            assert this_new_tokens == 0
            last_chunk_indices.append(len(cu_chunk_seqlen) - 1)

        cu_chunk_seqlen.append(seqlen_pos)

        return cu_chunk_seqlen, seq_idx, last_chunk_indices

    def _compute_common_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> M:
        """
        Compute metadata common to both Mamba1 and Mamba2.
        """
        num_reqs = common_attn_metadata.num_reqs

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=self.reorder_batch_threshold
            )
        )

        # Need flags to indicate if there are initial states
        has_initial_states_p = None
        query_start_loc_p = None
        num_computed_tokens = None
        num_computed_tokens_p = None

        # for prefix caching
        block_idx_first_scheduled_token = None
        block_idx_first_scheduled_token_p = None
        block_idx_last_computed_token = None
        block_idx_last_scheduled_token = None

        # for causal_conv1d
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None

        # for chunk alignment (needed for chunked prefill)
        prep_initial_states = False
        cu_chunk_seqlen_p = None
        seq_idx_p = None
        last_chunk_indices_p = None

        if self.vllm_config.cache_config.enable_prefix_caching:
            # Return a tensor of shape (#requests, #max blocks)
            state_indices_tensor = common_attn_metadata.block_table_tensor
            # Additional cache-related varaiables:
            mamba_block_size = self.kv_cache_spec.block_size
            num_computed_tokens = common_attn_metadata.num_computed_tokens_cpu.to(
                self.device
            )
            (
                block_idx_last_computed_token,
                block_idx_first_scheduled_token,
                block_idx_last_scheduled_token,
            ) = self._compute_prefix_caching_block_indices(
                common_attn_metadata, mamba_block_size
            )
        else:
            # Always return just a single block per each request:
            state_indices_tensor = common_attn_metadata.block_table_tensor[:, 0]

        if num_prefills > 0:
            query_start_loc_p = (
                common_attn_metadata.query_start_loc[-num_prefills - 1 :]
                - num_decode_tokens
            )
            has_initial_states_cpu = (
                common_attn_metadata.num_computed_tokens_cpu[
                    num_reqs - num_prefills : num_reqs
                ]
                > 0
            )
            has_initial_states_p = has_initial_states_cpu.to(
                common_attn_metadata.query_start_loc.device
            )

            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(query_start_loc_p)
            )

            # Always compute num_computed_tokens_p for prefills (needed for
            # chunk alignment even without prefix caching)
            num_computed_tokens_p_cpu = common_attn_metadata.num_computed_tokens_cpu[
                num_reqs - num_prefills : num_reqs
            ]
            num_computed_tokens_p = num_computed_tokens_p_cpu.to(self.device)

            # Compute chunk-aligned metadata for proper state handling
            query_start_loc_p_cpu = (
                common_attn_metadata.query_start_loc_cpu[-num_prefills - 1 :]
                - num_decode_tokens
            )
            cu_chunk_seqlen, seq_idx, last_chunk_indices = self._compute_chunk_metadata(
                num_prefills, num_computed_tokens_p_cpu, query_start_loc_p_cpu
            )
            cu_chunk_seqlen_p = torch.as_tensor(
                cu_chunk_seqlen,
                device=common_attn_metadata.query_start_loc.device,
                dtype=torch.int32,
            )
            seq_idx_p = torch.as_tensor(
                seq_idx,
                device=common_attn_metadata.query_start_loc.device,
                dtype=torch.int32,
            )
            last_chunk_indices_p = torch.as_tensor(
                last_chunk_indices,
                device=common_attn_metadata.query_start_loc.device,
                dtype=torch.int32,
            )

            # prep_initial_states indicates whether any prefill request has
            # initial states to load (continuation from previous chunk)
            prep_initial_states = bool(torch.any(has_initial_states_cpu).item())

            if self.vllm_config.cache_config.enable_prefix_caching:
                assert block_idx_first_scheduled_token is not None
                block_idx_first_scheduled_token_p = block_idx_first_scheduled_token[
                    num_reqs - num_prefills : num_reqs
                ]
        elif (
            num_decodes <= self.decode_cudagraph_max_bs
            and self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            self.state_indices_tensor[:num_decodes].copy_(
                state_indices_tensor, non_blocking=True
            )
            state_indices_tensor = self.state_indices_tensor[:num_decode_tokens]
            state_indices_tensor[num_decodes:] = PAD_SLOT_ID

            if self.vllm_config.cache_config.enable_prefix_caching:
                self.block_idx_last_scheduled_token[:num_decodes].copy_(
                    block_idx_last_scheduled_token, non_blocking=True
                )
                block_idx_last_scheduled_token = self.block_idx_last_scheduled_token[
                    :num_decode_tokens
                ]

                self.block_idx_last_computed_token[:num_decodes].copy_(
                    block_idx_last_computed_token, non_blocking=True
                )
                block_idx_last_computed_token = self.block_idx_last_computed_token[
                    :num_decode_tokens
                ]

        return self.metadata_cls(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            query_start_loc_p=query_start_loc_p,
            has_initial_states_p=has_initial_states_p,
            state_indices_tensor=state_indices_tensor,
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
            block_idx_first_scheduled_token_p=block_idx_first_scheduled_token_p,
            block_idx_last_computed_token=block_idx_last_computed_token,
            num_computed_tokens_p=num_computed_tokens_p,
            num_reqs=num_reqs,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
            # chunk alignment fields
            chunk_size=self.chunk_size,
            prep_initial_states=prep_initial_states,
            cu_chunk_seqlen_p=cu_chunk_seqlen_p,
            seq_idx_p=seq_idx_p,
            last_chunk_indices_p=last_chunk_indices_p,
        )

    def update_block_table(
        self,
        metadata: M,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> M:
        new_metadata = copy.copy(metadata)
        prefix_caching = self.vllm_config.cache_config.enable_prefix_caching
        state_indices_t = blk_table if prefix_caching else blk_table[:, 0]
        num_reqs = blk_table.shape[0]

        # For CUDA graphs, copy to persistent buffer
        if (
            metadata.num_prefills == 0
            and num_reqs <= self.decode_cudagraph_max_bs
            and self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            persistent_state_indices_t = self.state_indices_tensor[:num_reqs]
            persistent_state_indices_t.copy_(state_indices_t, non_blocking=True)
            state_indices_t = persistent_state_indices_t

        new_metadata.state_indices_tensor = state_indices_t
        return new_metadata

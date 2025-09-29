# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, NamedTuple, Optional

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

import torch
from torch import nn
from torch.nn.parameter import Parameter

from vllm import envs
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.config import CacheConfig, ModelConfig, get_current_vllm_config
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator, MambaStateShapeCalculator)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn, causal_conv1d_update)
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    selective_scan_fn, selective_state_update)
from vllm.model_executor.models.mamba_cache import MambaCacheParams
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils import cdiv, direct_register_custom_op
from vllm.v1.attention.backends.mamba1_attn import Mamba1AttentionMetadata


# Adapted from transformers.models.mamba.modeling_mamba.MambaMixer
@CustomOp.register("mamba_mixer")
class MambaMixer(MambaBase, CustomOp):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute
    the `contextualized_states`. A, D are input independent
    (see Mamba paper [1] Section 3.5.2 "Interpretation of A"
    for why A isn't selective) ∆, B, C are input-dependent
    (this is a key difference between Mamba and the linear time
    invariant S4, and is why Mamba is called
    **selective** state spaces)
    """

    def __init__(self,
                 hidden_size: int,
                 ssm_state_size: int,
                 conv_kernel_size: int,
                 intermediate_size: int,
                 time_step_rank: int,
                 use_conv_bias: bool,
                 use_bias: bool,
                 use_rms_norm: bool,
                 rms_norm_has_weight: bool = True,
                 rms_norm_eps: float = 1e-5,
                 activation="silu",
                 is_lora_enabled: bool = False,
                 model_config: Optional[ModelConfig] = None,
                 cache_config: Optional[CacheConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.time_step_rank = time_step_rank
        self.ssm_state_size = ssm_state_size
        self.use_rms_norm = use_rms_norm
        self.activation = activation
        self.is_lora_enabled = is_lora_enabled
        self.conv_kernel_size = conv_kernel_size
        self.intermediate_size = intermediate_size

        self.conv1d = ColumnParallelLinear(
            input_size=conv_kernel_size,
            output_size=intermediate_size,
            bias=use_conv_bias,
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        self.in_proj = MergedColumnParallelLinear(hidden_size,
                                                  [intermediate_size] * 2,
                                                  bias=use_bias)

        # selective projection used to make dt, B and C input dependent
        self.x_proj = RowParallelLinear(
            intermediate_size,
            time_step_rank + ssm_state_size * 2,
            bias=False,
        )
        # time step projection (discretization) -
        # In the forward we need to apply dt_proj without the bias,
        # as the bias is added in the selective scan kernel.
        self.dt_proj = ColumnParallelLinear(time_step_rank,
                                            intermediate_size,
                                            bias=True,
                                            skip_bias_add=True)

        def weight_loader(param: Parameter, loaded_weight: torch.Tensor):
            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()
            param.data.copy_(
                loaded_weight.data.split(loaded_weight.shape[0] // tp_size,
                                         dim=0)[tp_rank])

        def A_weight_loader(param: Parameter, loaded_weight: torch.Tensor):
            weight_loader(param, -torch.exp(loaded_weight.float()))

        tp_size = get_tensor_model_parallel_world_size()
        self.A = nn.Parameter(
            torch.empty(
                intermediate_size // tp_size,
                ssm_state_size,
                dtype=torch.float32,
            ))
        self.D = nn.Parameter(torch.ones(intermediate_size // tp_size))

        set_weight_attrs(self.D, {"weight_loader": weight_loader})
        set_weight_attrs(self.A, {"weight_loader": A_weight_loader})

        self.out_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=use_bias,
            input_is_parallel=True,
        )

        self.dt_layernorm = RMSNorm(
            time_step_rank,
            eps=rms_norm_eps,
            has_weight=rms_norm_has_weight,
        ) if use_rms_norm else None

        self.b_layernorm = RMSNorm(
            ssm_state_size,
            eps=rms_norm_eps,
            has_weight=rms_norm_has_weight,
        ) if use_rms_norm else None

        self.c_layernorm = RMSNorm(
            ssm_state_size,
            eps=rms_norm_eps,
            has_weight=rms_norm_has_weight,
        ) if use_rms_norm else None

        if envs.VLLM_USE_V1:
            compilation_config = get_current_vllm_config().compilation_config
            if prefix in compilation_config.static_forward_context:
                raise ValueError(f"Duplicate layer name: {prefix}")
            compilation_config.static_forward_context[prefix] = self
            # The outer list is for v0 PP virtual engine. Though this code path
            # only runs for v1, we have to do this to unify with the interface
            # of Attention + v0 PP.
            # The inner tuple is (conv_state, ssm_state)
            self.kv_cache = [(torch.tensor([]), torch.tensor([]))]

        self.model_config = model_config
        self.cache_config = cache_config
        self.prefix = prefix

    def _ssm_transform(
            self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.is_lora_enabled:
            #  Lora kernel requires contiguous tensor.
            ssm_params = self.x_proj(x.contiguous())[0]
        else:
            ssm_params = self.x_proj(x)[0]
        time_step, B, C = torch.split(
            ssm_params,
            [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
            dim=-1)
        if self.use_rms_norm:
            assert self.dt_layernorm is not None
            assert self.b_layernorm is not None
            assert self.c_layernorm is not None
            time_step = self.dt_layernorm(time_step.contiguous())
            B = self.b_layernorm(B.contiguous())
            C = self.c_layernorm(C.contiguous())
        discrete_time_step = self.dt_proj(time_step)[0].transpose(-2, -1)
        return discrete_time_step, B, C

    def forward(self,
                hidden_states: torch.Tensor,
                output: torch.Tensor,
                mamba_cache_params: Optional[MambaCacheParams] = None):
        if not envs.VLLM_USE_V1:
            CustomOp.forward(self, hidden_states, output, mamba_cache_params)
        else:
            torch.ops.vllm.mamba_mixer(
                hidden_states,
                output,
                self.prefix,
            )

    def forward_native(self,
                       hidden_states: torch.Tensor,
                       output: torch.Tensor,
                       mamba_cache_params: Optional[MambaCacheParams] = None):
        pass

    def forward_cuda(self,
                     hidden_states: torch.Tensor,
                     output: torch.Tensor,
                     mamba_cache_params: Optional[MambaCacheParams] = None):
        """
        Run the Mamba-1 SSM pipeline.

        Steps
        -----
        1. Apply the gated-MLP linear projection to the raw input.
        2. Pass the projected sequence through the convolutional mixing layer.
        3. Feed the result into the State-Space Model (SSM) blocks.
        4. Perform the recurrence y ← SSM(A, B, C, Δ)(x)
           to produce contextual representations.
        5. Project the contextualised sequence back
           to the output embedding dimension.

        Batch handling
        --------------
        Prefill and decode tokens are processed by dedicated CUDA
        kernels for both the convolutional (conv1d) and SSM stages.
        In the case of a mixed batch (containing both prefill and
        decode tokens), both sets of kernels are executed independently
        and their outputs are concatenated before the final output projection.
        """

        forward_context: ForwardContext = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if envs.VLLM_USE_V1:
            if attn_metadata is not None:
                assert isinstance(attn_metadata, dict)
                attn_metadata = attn_metadata[self.prefix]
                mamba1_metadata = attn_metadata
                assert isinstance(mamba1_metadata, Mamba1AttentionMetadata)
                query_start_loc = mamba1_metadata.query_start_loc
                state_indices_tensor = mamba1_metadata.state_indices_tensor
                self_kv_cache = self.kv_cache[forward_context.virtual_engine]
                conv_state = self_kv_cache[0].transpose(-1, -2)
                ssm_state = self_kv_cache[1]
                has_initial_states = mamba1_metadata.has_initial_states
                num_padded_decodes = mamba1_metadata.num_padded_decodes
                mamba_block_size = attn_metadata.cache_spec.block_size
                cache_strategy = attn_metadata.cache_spec.cache_strategy
                cache_enabled = (cache_strategy != "disabled")
        else:
            assert isinstance(attn_metadata, AttentionMetadata)
            assert mamba_cache_params is not None
            conv_state = mamba_cache_params.conv_state
            ssm_state = mamba_cache_params.ssm_state
            state_indices_tensor = mamba_cache_params.state_indices_tensor
            query_start_loc = attn_metadata.query_start_loc
            context_lens_tensor = attn_metadata.context_lens_tensor
            has_initial_states = None
            if context_lens_tensor is not None:
                has_initial_states = context_lens_tensor > 0
            num_padded_decodes = attn_metadata.num_decode_tokens

        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states)[0].transpose(-2, -1)
        hidden_states_BC, gate = projected_states.chunk(2, dim=-2)

        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0),
                                               self.conv1d.weight.size(2))

        if envs.VLLM_USE_V1 and attn_metadata is None:
            # V1 profile run
            hidden_states_BC = hidden_states_BC.contiguous()
            return self.out_proj(hidden_states_BC.transpose(-2, -1))[0]


        num_prefill_tokens = attn_metadata.num_prefill_tokens  # token count
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefills = attn_metadata.num_prefills  # request count
        num_decodes = attn_metadata.num_decodes  # request count (not token count)
        has_prefill = num_prefill_tokens > 0
        has_decode = num_decode_tokens > 0
        num_actual_tokens = num_prefill_tokens + num_decode_tokens
            
        prefill_decode_split = split_batch_to_prefill_and_decode(
            hidden_states_BC,
            gate,
            state_indices_tensor,
            query_start_loc,
            has_initial_states,
            num_prefill_tokens,
            num_decode_tokens,
            num_prefills,
            num_decodes,
            num_padded_decodes,
        )
        hidden_states_BC_p = prefill_decode_split.hidden_states_BC_p
        hidden_states_BC_d = prefill_decode_split.hidden_states_BC_d
        gate_p = prefill_decode_split.gate_p
        gate_d = prefill_decode_split.gate_d
        state_indices_tensor_p = prefill_decode_split.state_indices_tensor_p
        state_indices_tensor_d = prefill_decode_split.state_indices_tensor_d
        query_start_loc_p = prefill_decode_split.query_start_loc_p
        has_initial_states_p = prefill_decode_split.has_initial_states_p

        if envs.VLLM_USE_V1 and cache_enabled:
            # Split decodes and prefills:
            seq_lens_completed_d, seq_lens_completed_p = torch.split(
                attn_metadata.seq_lens_completed, [num_decodes, num_prefills],
                dim=0)
            last_state_idx_d, last_state_idx_p = torch.split(
                attn_metadata.last_computed_token_block_idx,
                [num_decodes, num_prefills],
                dim=0)
            last_computed_offset_d, last_computed_offset_p = torch.split(
                attn_metadata.last_computed_token_block_offset,
                [num_decodes, num_prefills],
                dim=0)
            current_first_idx_d, current_first_idx_p = torch.split(
                attn_metadata.current_first_token_block_idx,
                [num_decodes, num_prefills],
                dim=0)
            current_last_idx_d, current_last_idx_p = torch.split(
                attn_metadata.current_last_token_block_idx,
                [num_decodes, num_prefills],
                dim=0)

        
        ssm_outputs = []

        if has_prefill:
            if cache_enabled:
                n_blocks_to_fill = current_last_idx_p - current_first_idx_p
                stride_state_indices = state_indices_tensor_p.shape[-1]
            else:
                current_first_idx_p = None
                current_last_idx_p = None
                seq_lens_completed_p = None
                last_state_idx_p = None
                n_blocks_to_fill = None
                stride_state_indices = 1
            

            # 2. Convolution sequence transformation
            conv_out_p = causal_conv1d_fn(
                hidden_states_BC_p,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_states_p,
                cache_indices=state_indices_tensor_p,
                n_blocks_to_fill=n_blocks_to_fill,
                current_first_idx=current_first_idx_p,
                current_last_idx=current_last_idx_p,
                last_state_idx=last_state_idx_p,
                seq_lens_completed=seq_lens_completed_p,
                stride_cache_chunk=mamba_block_size // 1,
                stride_state_indices=stride_state_indices,
                query_start_loc=query_start_loc_p)
            
            # 3. State Space Model sequence transformations.
            discrete_time_step_p, B_p, C_p = self._ssm_transform(
                conv_out_p.transpose(-2, -1))
            time_proj_bias = self._time_proj_bias()
            
            # APC parameters
            if (has_initial_states_p is not None):
                # making a copy of the states
                if envs.VLLM_USE_V1:
                    kernel_ssm_indices = state_indices_tensor_p
                    if cache_enabled:
                        kernel_ssm_indices = state_indices_tensor_p.gather(
                1, current_last_idx_p.unsqueeze(1)).squeeze(1)
            
            scan_result = selective_scan_fn(
                conv_out_p,
                ssm_state,
                discrete_time_step_p,
                self.A,
                B_p.transpose(-2, -1),
                C_p.transpose(-2, -1),
                self.D.float(),
                gate_p,
                time_proj_bias,
                delta_softplus=True,
                cache_indices=kernel_ssm_indices,
                has_initial_state=has_initial_states_p,
                query_start_loc=query_start_loc_p,
                return_intermediate_states=cache_enabled,
                cache_enabled=cache_enabled,
                block_size=mamba_block_size)

            if cache_enabled:
                # When cache is enabled, selective_scan_fn returns (output, intermediate_states)
                scan_out_p, intermediate_states = scan_result

                # Store intermediate states back into ssm_state for future runs
                # Similar to how mamba_mixer2 handles it
                n_blocks_to_fill = current_last_idx_p - current_first_idx_p

                # Save states for sequences with more than just the final state
                for seq_idx in (n_blocks_to_fill > 0).nonzero().squeeze(1):
                    cache_blocks_to_fill = state_indices_tensor_p[
                        seq_idx, current_first_idx_p[seq_idx]:
                        current_first_idx_p[seq_idx] + n_blocks_to_fill[seq_idx]]

                    # Calculate which blocks from intermediate_states to use
                    # intermediate_states shape: [batch, max_blocks, dim, dstate]
                    # We need to map the blocks correctly based on sequence position
                    blocks_to_copy = n_blocks_to_fill[seq_idx].item()

                    # Copy the intermediate states to the appropriate cache blocks
                    ssm_state[cache_blocks_to_fill] = intermediate_states[
                        seq_idx, :blocks_to_copy]

                # For all sequences, the final state is already updated in ssm_state
                # by the kernel itself, so we don't need to update it again
            else:
                # When cache is disabled, selective_scan_fn returns just the output
                scan_out_p = scan_result

            ssm_outputs.append(scan_out_p)


        if has_decode:

            if cache_enabled:
                state_indices_tensor_d_input = \
                    state_indices_tensor_d.gather(1,
                        last_state_idx_d.unsqueeze(1)).squeeze(1)
                state_indices_tensor_d_output = \
                    state_indices_tensor_d.gather(1,
                        current_last_idx_d.unsqueeze(1)).squeeze(1)
                #Note:
                # for decode always: current_first_idx_d == current_last_idx_d
                # at block boundaries: current_first_idx_d > last_state_idx_d
            else:
                # Without caching, read and write in-place to the same blocks:
                state_indices_tensor_d_input = state_indices_tensor_d
                state_indices_tensor_d_output = state_indices_tensor_d
                current_last_idx_d = None
                last_state_idx_d = None
                
            # 2. Convolution sequence transformation
            conv_out_d = causal_conv1d_update(
                hidden_states_BC_d.transpose(0, 1),
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=state_indices_tensor_d,
                current_last_idx=current_last_idx_d,
                last_state_idx=last_state_idx_d).transpose(0, 1)

            # 3. State Space Model sequence transformation.
            discrete_time_step_d, B_d, C_d = self._ssm_transform(
                conv_out_d.transpose(-2, -1))
            time_proj_bias = self._time_proj_bias()

            # 4. Perform the recurrence y ← SSM(A, B, C, Δ)(x)
            scan_outputs_d = torch.empty_like(
                hidden_states_BC_d.transpose(0, 1))
            selective_state_update(ssm_state,
                                   conv_out_d.transpose(0, 1),
                                   discrete_time_step_d.transpose(0, 1),
                                   self.A,
                                   B_d,
                                   C_d,
                                   self.D,
                                   gate_d.transpose(0, 1),
                                   time_proj_bias,
                                   dt_softplus=True,
                                   state_batch_indices=state_indices_tensor_d_input,
                                   dst_state_batch_indices=state_indices_tensor_d_output,
                                   out=scan_outputs_d)
            scan_outputs_d = scan_outputs_d.transpose(0, 1)

            if envs.VLLM_USE_V1:
                ssm_outputs.insert(0, scan_outputs_d)
            else:
                ssm_outputs.append(scan_outputs_d)

        scan_outputs_combined = ssm_outputs[0] if len(
            ssm_outputs) == 1 else torch.cat(ssm_outputs, dim=-1)

        # 5. Final output projection
        if self.is_lora_enabled:  # Lora kernel requires contiguous tensor.
            scan_outputs_combined = scan_outputs_combined.transpose(
                -2, -1).contiguous()
            out = self.out_proj(scan_outputs_combined)[0]
        else:
            out = self.out_proj(scan_outputs_combined.transpose(-2, -1))[0]

        output[:num_actual_tokens] = out

    def get_state_dtype(self) -> tuple[torch.dtype]:
        assert self.model_config is not None
        assert self.cache_config is not None
        return MambaStateDtypeCalculator.mamba1_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.mamba1_state_shape(
            tp_world_size=get_tensor_model_parallel_world_size(),
            intermediate_size=self.intermediate_size,
            state_size=self.ssm_state_size,
            conv_kernel=self.conv_kernel_size,
        )

    @property
    def mamba_type(self) -> str:
        return "mamba1"

    def get_attn_backend(self) -> type["AttentionBackend"]:
        from vllm.v1.attention.backends.mamba1_attn import (
            Mamba1AttentionBackend)
        return Mamba1AttentionBackend

    def _time_proj_bias(self) -> Optional[torch.Tensor]:
        if hasattr(self.dt_proj, "bias") and self.dt_proj.bias is not None:
            return self.dt_proj.bias.float()
        return None


class PrefillDecodeSplit(NamedTuple):
    hidden_states_BC_p: torch.Tensor
    hidden_states_BC_d: torch.Tensor
    gate_p: torch.Tensor
    gate_d: torch.Tensor
    state_indices_tensor_p: torch.Tensor
    state_indices_tensor_d: torch.Tensor
    query_start_loc_p: Optional[torch.Tensor]
    has_initial_states_p: Optional[torch.Tensor]


def split_batch_to_prefill_and_decode(
    hidden_states_BC: torch.Tensor,
    gate: torch.Tensor,
    state_indices_tensor: torch.Tensor,
    query_start_loc: torch.Tensor,
    has_initial_states: Optional[torch.Tensor],
    num_prefill_tokens: int,
    num_decode_tokens: int,
    num_prefills: int,
    num_decodes: int,
    num_padded_decodes: int,
) -> PrefillDecodeSplit:
    num_actual_tokens = num_prefill_tokens + num_padded_decodes

    if envs.VLLM_USE_V1:
        # In v1, decode tokens come first, then prefill tokens.
        hidden_states_BC_d, hidden_states_BC_p = torch.split(
            hidden_states_BC[..., :num_actual_tokens],
            [num_padded_decodes, num_prefill_tokens],
            dim=-1)
        gate_d, gate_p = torch.split(gate[..., :num_actual_tokens],
                                     [num_padded_decodes, num_prefill_tokens],
                                     dim=-1)

        # num_padded_decodes accounts for CUDA graph padding when applicable
        state_indices_tensor_d, state_indices_tensor_p = torch.split(
            state_indices_tensor[:num_padded_decodes + num_prefills],
            [num_padded_decodes, num_prefills],
            dim=0)
        query_start_loc_p = (query_start_loc[-num_prefills - 1:] -
                             num_padded_decodes if num_prefills > 0 else None)
        has_initial_states_p = has_initial_states[-num_prefills:] if (
            has_initial_states is not None and num_prefills > 0) else None
    else:
        # In v0, prefill tokens come first, then decode tokens.
        hidden_states_BC_p, hidden_states_BC_d = torch.split(
            hidden_states_BC, [num_prefill_tokens, num_decode_tokens], dim=-1)
        gate_p, gate_d = torch.split(gate,
                                     [num_prefill_tokens, num_decode_tokens],
                                     dim=-1)
        state_indices_tensor_p, state_indices_tensor_d = torch.split(
            state_indices_tensor, [num_prefills, num_decodes], dim=0)
        query_start_loc_p = (query_start_loc[:num_prefills +
                                             1] if num_prefills > 0 else None)
        has_initial_states_p = has_initial_states[:num_prefills] if (
            has_initial_states is not None and num_prefills > 0) else None

    return PrefillDecodeSplit(
        hidden_states_BC_p=hidden_states_BC_p,
        hidden_states_BC_d=hidden_states_BC_d,
        gate_p=gate_p,
        gate_d=gate_d,
        state_indices_tensor_p=state_indices_tensor_p,
        state_indices_tensor_d=state_indices_tensor_d,
        query_start_loc_p=query_start_loc_p,
        has_initial_states_p=has_initial_states_p,
    )


def _copy_cached_states_for_requests(state_tensor, state_indices_tensor, 
                                   last_computed_indices, current_last_indices, 
                                   has_initial_states, req_count):
    """Copy cached states from previous blocks for requests with initial states."""
      # Only copy if there are requests with initial states
    if has_initial_states is None or has_initial_states.sum() == 0:
        return

    # Get exact indices using index_select + diag pattern
    state_idx_input = state_indices_tensor.index_select(
        1, last_computed_indices).diag()
    state_idx_output = state_indices_tensor.index_select(
        1, current_last_indices).diag()

    # Only copy for requests that have initial states
    state_tensor[state_idx_output[has_initial_states]] = \
        state_tensor[state_idx_input[has_initial_states]]

def _get_kernel_indices(state_indices_tensor, current_last_indices):
    """Get kernel indices for computation, with fallback to default."""
    # If no per-request last indices are provided, fall back to full tensor
    if current_last_indices is None or current_last_indices.numel() == 0:
        return state_indices_tensor

    # Ensure correct dtype/device for gather
    indices = current_last_indices.to(device=state_indices_tensor.device)
    # Bounds check: if any index is out of range, fall back to full tensor
    if indices.max() >= state_indices_tensor.shape[1]:
        return state_indices_tensor

    gathered = torch.gather(state_indices_tensor, 1, indices.unsqueeze(1))
    return gathered.squeeze(1)


def _get_kernel_indices_range(state_indices_tensor, first_indices, last_indices):
    """Get kernel indices for a range of blocks from first to last (inclusive).
    
    For cache strategies that need to update multiple blocks, this function
    returns the appropriate indices based on the strategy.
    """
    if (first_indices is None or last_indices is None or 
        first_indices.numel() == 0 or last_indices.numel() == 0):
        return state_indices_tensor
    
    # Ensure correct device/dtype
    first_indices = first_indices.to(device=state_indices_tensor.device)
    last_indices = last_indices.to(device=state_indices_tensor.device)
    
    # For multiple blocks per request, we need to gather all indices in the range
    # However, kernels typically expect a single index per request
    # So we return the last index as the primary cache target
    # The range information is used by the metadata builder to allocate the right blocks
    return _get_kernel_indices(state_indices_tensor, last_indices)

def _handle_decode_state_indices(cache_enabled, state_indices_tensor_d, num_decodes,
                               attn_metadata, mamba_block_size, conv_state, ssm_state):
    """Handle state indices computation for decode phase (capture-safe).

    Returns two 1D tensors of length `num_decodes` with input and output
    state indices, computed in a vectorized way suitable for CUDA graph capture.
    """
    if not cache_enabled:
        return state_indices_tensor_d, state_indices_tensor_d
    
    seq_lens = attn_metadata.seq_lens

    # Align seq_lens to decode rows (supports mixed prefill+decode batches)
    batch_count = state_indices_tensor_d.shape[0] if state_indices_tensor_d.ndim >= 1 else 0
    if batch_count == 0:
        return state_indices_tensor_d, state_indices_tensor_d
    
    if seq_lens.numel() > batch_count:
        # Since decode requests come first, we need to truncate seq_lens to 
        # the number of decode requests
        seq_lens = seq_lens[:batch_count]

    # Compute 0-based input/output block indices per request
    input_block_idx = torch.div(seq_lens - 1,
                                mamba_block_size,
                                rounding_mode='floor')
    output_block_idx = torch.div(seq_lens,
                                 mamba_block_size,
                                 rounding_mode='floor')

    # Clamp to valid block column range
    max_col = state_indices_tensor_d.shape[1] - 1 if state_indices_tensor_d.ndim == 2 else 0
    if state_indices_tensor_d.ndim == 1:
        # Already a per-request vector of indices
        state_indices_input = state_indices_tensor_d
        state_indices_output = state_indices_tensor_d
        return state_indices_input, state_indices_output

    input_block_idx = input_block_idx.clamp(min=0, max=max_col)
    output_block_idx = output_block_idx.clamp(min=0, max=max_col)

    # Gather per-request indices from the state index table
    state_indices_input = state_indices_tensor_d.gather(1, input_block_idx.unsqueeze(1)).squeeze(1)
    state_indices_output = state_indices_tensor_d.gather(1, output_block_idx.unsqueeze(1)).squeeze(1)

    # Vectorized copy of states for requests moving to a new block
    # Use torch.where for CUDA graph-safe conditional execution
    move_mask = output_block_idx > input_block_idx
    
    # Create indices arrays that handle the conditional copy
    # torch.where is graph-safe and doesn't require CPU sync
    src_indices = torch.where(move_mask, state_indices_input, state_indices_output)
    dst_indices = state_indices_output
    
    # Always perform the copy - when move_mask is False, we copy from dst to dst (no-op)
    # This avoids any conditional branching while being graph-capture safe
    conv_state[dst_indices] = conv_state[src_indices]
    ssm_state[dst_indices] = ssm_state[src_indices]

    return state_indices_input, state_indices_output

def mamba_mixer(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self.forward_cuda(hidden_states=hidden_states,
                      output=output,
                      mamba_cache_params=None)


def mamba_mixer_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="mamba_mixer",
    op_func=mamba_mixer,
    mutates_args=["output"],
    fake_impl=mamba_mixer_fake,
    dispatch_key=current_platform.dispatch_key,
)

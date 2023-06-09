import torch
import math
from torch import nn
from .utils import freeze_module
from typing import List, Optional, Tuple, Union
import transformers.models.opt.modeling_opt
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)


def opt_adapter_set_kwargs(before, **kwargs):
    dtype = next(before.parameters()).dtype
    return {
        "embed_dim": before.embed_dim,
        "num_heads": before.num_heads,
        "num_heads": before.num_heads,
        "dropout": before.dropout,
        "is_decoder": before.is_decoder,
        "adapter_len": kwargs.pop("adapter_len", 10),
        "dtype": dtype,
    }


class OPTNaiveAdapter(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        is_decoder=False,
        adapter_len=10,
        bias=True,
        dtype=torch.float,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.adapter_prefix = nn.Embedding(adapter_len, embed_dim)
        self.gate = nn.Parameter(torch.zeros(num_heads, 1, 1))

        freeze_module(self.k_proj)
        freeze_module(self.v_proj)
        freeze_module(self.q_proj)
        freeze_module(self.out_proj)
        self.to(dtype)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        # --------------------------------------------------
        # Add adapter prefix here
        adapter_prefix = self.adapter_prefix.weight
        prefix_len = adapter_prefix.shape[0]
        key_prefixes = self._shape(
            self.k_proj(adapter_prefix).repeat(bsz, 1, 1), -1, bsz
        )
        value_prefixes = self._shape(
            self.v_proj(adapter_prefix).repeat(bsz, 1, 1), -1, bsz
        )

        key_states = torch.cat([key_prefixes, key_states], dim=2)
        value_states = torch.cat([value_prefixes, value_states], dim=2)
        attention_mask = torch.cat(
            [
                torch.zeros(bsz, 1, tgt_len, prefix_len).to(attention_mask),
                attention_mask,
            ],
            dim=3,
        )
        # --------------------------------------------------

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # --------------------------------------------------
        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        prefix_weights = attn_weights[..., :prefix_len]
        token_weights = attn_weights[..., prefix_len:]

        if attn_weights.dtype == torch.float16:
            gates = self.gate.tanh().half().repeat(bsz, 1, 1)
            prefix_weights = nn.functional.softmax(
                prefix_weights, dim=-1, dtype=torch.float32
            ).to(torch.float16)
            token_weights = nn.functional.softmax(
                token_weights, dim=-1, dtype=torch.float32
            ).to(torch.float16)
        else:
            gates = self.gate.tanh().repeat(bsz, 1, 1)
            prefix_weights = nn.functional.softmax(prefix_weights, dim=-1)
            token_weights = nn.functional.softmax(token_weights, dim=-1)
        attn_weights = torch.cat([gates * prefix_weights, token_weights], dim=-1)

        # --------------------------------------------------
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


def llama_adapter_set_kwargs(before, **kwargs):
    dtype = next(before.parameters()).dtype
    return {
        "config": before.config,
        "adapter_len": kwargs.pop("adapter_len", 10),
        "dtype": dtype,
    }


class LlamaNaiveAdapter(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, adapter_len, dtype=torch.float):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = freeze_module(
            nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        )
        self.k_proj = freeze_module(
            nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        )
        self.v_proj = freeze_module(
            nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        )
        self.o_proj = freeze_module(
            nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        )
        self.rotary_emb = freeze_module(
            LlamaRotaryEmbedding(
                self.head_dim, max_position_embeddings=self.max_position_embeddings
            )
        )

        self.adapter_prefix = nn.Embedding(adapter_len, self.hidden_size)
        self.gate = nn.Parameter(torch.zeros(self.num_heads, 1, 1))

        self.to(dtype)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = (
            self.q_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # --------------------------------------------------
        # add adapter here
        adapter_prefix = self.adapter_prefix.weight
        prefix_len = adapter_prefix.shape[0]
        key_prefixes = (
            self.k_proj(adapter_prefix)
            .view(bsz, prefix_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_prefixes = (
            self.v_proj(adapter_prefix)
            .view(bsz, prefix_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        key_states = torch.cat([key_prefixes, key_states], dim=2)
        value_states = torch.cat([value_prefixes, value_states], dim=2)
        attention_mask = torch.cat(
            [
                torch.zeros(bsz, 1, q_len, prefix_len).to(attention_mask),
                attention_mask,
            ],
            dim=-1,
        )
        kv_seq_len += prefix_len
        # --------------------------------------------------

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )

        # upcast attention to fp32
        prefix_weights = attn_weights[..., :prefix_len]
        token_weights = attn_weights[..., prefix_len:]
        gates = self.gate.tanh().repeat(bsz, 1, 1).to(prefix_weights.dtype)

        prefix_weights = nn.functional.softmax(
            prefix_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        token_weights = nn.functional.softmax(
            token_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = torch.cat([gates * prefix_weights, token_weights], dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

import dataclasses
import enum
from typing import Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionType(enum.Enum):
    GLOBAL = 1
    LOCAL_SLIDING = 2

@dataclasses.dataclass
class TurkishConfig:
    # The number of tokens in the vocabulary.
    vocab_size: int = 256000
    # The maximum sequence length that this model might ever be used with.
    max_position_embeddings: int = 8192
    # The number of blocks in the model.
    num_hidden_layers: int = 28
    # The number of attention heads used in the attention layers of the model.
    num_attention_heads: int = 16
    # The number of key-value heads for implementing attention.
    num_key_value_heads: int = 16
    # The hidden size of the model.
    hidden_size: int = 3072
    # The dimension of the MLP representations.
    intermediate_size: int = 24576
    # The number of head dimensions.
    head_dim: int = 256
    # The epsilon used by the rms normalization layers.
    rms_norm_eps: float = 1e-6
    # The dtype of the weights.
    dtype: str = 'float32'
    # The types of attention used in the layers of the model.
    attn_types: Optional[Sequence[AttentionType]] = None
    # The size of the sliding window used for local attention.
    sliding_window_size: Optional[int] = None
    # If provided, the query vector is normalized using the
    # inverse square root of this value instead of head_dim.
    query_pre_attn_scalar: Optional[int] = None
    # The wave length of the rotary embedding.
    rope_wave_length: dict[AttentionType, int] | None = None
    # Whether to use QK normalization in the attention blocks.
    use_qk_norm: bool = False
    # The factor by which the rope wave length is divided for global layers.
    rope_scaling_factor: int| None = None

    def get_dtype(self) -> torch.dtype:
        """Convert string dtype to torch.dtype."""
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
        }
        return dtype_map.get(self.dtype, torch.float32)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies the rotary embedding to the query and key tensors."""
    x_ = torch.view_as_complex(
        torch.stack(torch.chunk(x.transpose(1, 2).float(), 2, dim=-1),
                    dim=-1))
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2],
                          -1).transpose(1, 2)
    return x_out

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, add_unit_offset: bool = True):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = self._norm(x.float())
        if self.add_unit_offset:
            output = output * (1 + self.weight.float())
        else:
            output = output * self.weight.float()
        return output.type_as(x)

class TurkishMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh")
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        return outputs

class TurkishAttention(nn.Module):
    def __init__(self, config: TurkishConfig, attn_type: AttentionType):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        if config.query_pre_attn_scalar is not None:
            self.scaling = config.query_pre_attn_scalar**-0.5
        else:
            self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self.query_norm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if config.use_qk_norm
            else None
        )
        self.key_norm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if config.use_qk_norm
            else None
        )

        self.sliding_window_size = config.sliding_window_size
        self.attn_type = attn_type

    def forward(self,
                hidden_states: torch.Tensor,
                freqs_cis: torch.Tensor,
                mask: torch.Tensor,
                local_mask: torch.Tensor = None) -> torch.Tensor:
        hidden_states_shape = hidden_states.shape
        assert len(hidden_states_shape) == 3

        batch_size, input_len, _ = hidden_states_shape

        xq = self.q_proj(hidden_states)
        xk = self.k_proj(hidden_states)
        xv = self.v_proj(hidden_states)

        xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)

        if self.query_norm is not None and self.key_norm is not None:
            xq = self.query_norm(xq)
            xk = self.key_norm(xk)

        # Positional embedding.
        xq = apply_rotary_emb(xq, freqs_cis=freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis=freqs_cis)
  
        if self.num_kv_heads != self.num_heads:
            # [batch_size, max_seq_len, n_local_heads, head_dim]
            xk = torch.repeat_interleave(xk, self.num_queries_per_kv, dim=2)
            xv = torch.repeat_interleave(xv,
                                            self.num_queries_per_kv,
                                            dim=2)

        # [batch_size, n_local_heads, input_len, head_dim]
        q = xq.transpose(1, 2)
        # [batch_size, n_local_heads, max_seq_len, head_dim]
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)

        # [batch_size, n_local_heads, input_len, max_seq_len]
        q.mul_(self.scaling)
        scores = torch.matmul(q, k.transpose(2, 3))
        if (
            self.attn_type == AttentionType.LOCAL_SLIDING
            and self.sliding_window_size is not None
            and local_mask is not None
        ):
            mask = local_mask

        scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)

        # [batch_size, n_local_heads, input_len, head_dim]
        output = torch.matmul(scores, v)

        # [batch_size, input_len, hidden_dim]
        output = (output.transpose(1, 2).contiguous().view(
            batch_size, input_len, -1))
        output = self.o_proj(output)
        return output

class TurkishDecoderLayer(nn.Module):
    def __init__(self, config: TurkishConfig, attn_type: AttentionType):
        super().__init__()
        self.self_attn = TurkishAttention(config=config, attn_type=attn_type)
        self.mlp = TurkishMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, 
                                                 eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, 
                                                  eps=config.rms_norm_eps)

    def forward(self,
                hidden_states: torch.Tensor,
                freqs_cis: torch.Tensor,
                mask: torch.Tensor,
                local_mask: torch.Tensor = None) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            mask=mask,
            local_mask=local_mask,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class TurkishModel(nn.Module):
    def __init__(self, config: TurkishConfig, attn_type: AttentionType):
        super().__init__()

        self.embedder = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            attn_type = (
                config.attn_types[i % len(config.attn_types)]
                if config.attn_types is not None
                else AttentionType.GLOBAL
            )
            self.layers.append(TurkishDecoderLayer(config, attn_type))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.config = config

    def forward(
        self,
        input_token_ids: torch.Tensor,
        freqs_cis: Mapping[AttentionType, torch.Tensor],
        mask: torch.Tensor,
        local_mask: torch.Tensor,
    ) -> torch.Tensor:
        # [batch_size, input_len, hidden_size]
        hidden_states = self.embedder(input_token_ids)
        # Gemma normalizes the embedding by sqrt(hidden_size).
        # Gemma2 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = hidden_states * normalizer
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis.get(layer.attn_type),
                mask=mask,
                local_mask=local_mask,
            )
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.index_select(
        1, output_positions).squeeze(dim=1)
        logits = torch.matmul(hidden_states, self.embedder.weight.t())
        return logits



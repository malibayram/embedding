import copy
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class Gemma3Config:
    vocab_size: int = 256000
    hidden_size: int = 3072
    intermediate_size: int = 24576
    num_hidden_layers: int = 28
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_local_base_freq: float = 10.0
    sliding_window: int = 4096
    attention_dropout: float = 0.0
    attention_bias: bool = False
    hidden_activation: str = "gelu_pytorch_tanh"
    query_pre_attn_scalar: float = 128.0
    attn_logit_softcapping: Optional[float] = 50.0
    final_logit_softcapping: Optional[float] = 30.0
    layer_types: list = None
    pad_token_id: int = 0
    use_cache: bool = True
    
    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers


class Gemma3ScaledWordEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)

class Gemma3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * (1.0 + self.weight.float())


class Gemma3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=8192, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Gemma3MLP(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Gemma3Attention(nn.Module):
    def __init__(self, config: Gemma3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = config.query_pre_attn_scalar ** -0.5
        self.sliding_window = config.sliding_window if self.is_sliding else None
        self.attn_logit_softcapping = config.attn_logit_softcapping

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        self.q_norm = Gemma3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if self.attn_logit_softcapping is not None:
            attn_weights = attn_weights / self.attn_logit_softcapping
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * self.attn_logit_softcapping

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, -1)

        return self.o_proj(attn_output)


class Gemma3DecoderLayer(nn.Module):
    def __init__(self, config: Gemma3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.self_attn = Gemma3Attention(config, layer_idx)
        self.mlp = Gemma3MLP(config)
        
        self.input_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings_global, position_embeddings_local, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        position_embeddings = position_embeddings_local if self.self_attn.is_sliding else position_embeddings_global
        
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


def create_causal_mask(batch_size, seq_len, device, dtype):
    mask = torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)


class Gemma3Model(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = Gemma3ScaledWordEmbedding(
            config.vocab_size, 
            config.hidden_size, 
            config.pad_token_id,
            embed_scale=config.hidden_size ** 0.5
        )
        
        self.layers = nn.ModuleList([
            Gemma3DecoderLayer(config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3RotaryEmbedding(config.head_dim, config.max_position_embeddings, config.rope_theta)
        
        local_config = copy.deepcopy(config)
        local_config.rope_theta = config.rope_local_base_freq
        self.rotary_emb_local = Gemma3RotaryEmbedding(
            config.head_dim, 
            config.max_position_embeddings, 
            local_config.rope_theta
        )

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        batch_size, seq_len = input_ids.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        inputs_embeds = self.embed_tokens(input_ids)
        
        if attention_mask is None:
            attention_mask = create_causal_mask(batch_size, seq_len, input_ids.device, inputs_embeds.dtype)
        
        position_embeddings_global = self.rotary_emb(inputs_embeds, position_ids)
        position_embeddings_local = self.rotary_emb_local(inputs_embeds, position_ids)
        
        hidden_states = inputs_embeds
        
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
                attention_mask=attention_mask,
            )
        
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Gemma3ForCausalLM(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.config = config
        self.model = Gemma3Model(config)
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, position_ids=None, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        
        logits = torch.matmul(outputs, self.model.embed_tokens.weight.T)
        # logits = self.lm_head(outputs)
        
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs
        }

    def generate(self, input_ids, max_length=50, temperature=1.0, do_sample=True):
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                outputs = self(generated)
                logits = outputs['logits']
                next_token_logits = logits[:, -1, :] / temperature
                
                if do_sample:
                    probs = nn.functional.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated = torch.cat([generated, next_token], dim=-1)
                
                if next_token.item() == self.config.pad_token_id:
                    break
        
        return generated


def create_gemma3_model(vocab_size=32768, hidden_size=640, num_layers=18, num_heads=4, num_kv_heads=1):
    config = Gemma3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=256,
        intermediate_size=2048,
        max_position_embeddings=32768,
    )
    return Gemma3ForCausalLM(config)


if __name__ == "__main__":
    model = create_gemma3_model()
    
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(input_ids)
        print(f"Model output shape: {outputs['logits'].shape}")
        print("Model created successfully!")
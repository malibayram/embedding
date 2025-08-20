import copy
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    output_attentions: bool = False
    output_hidden_states: bool = False
    
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

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Gemma3 style: (x * w).to(dtype) instead of x.to(dtype) * w
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Gemma3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=8192, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        # Ensure position_ids is on the same device as x
        position_ids = position_ids.to(x.device)
        
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) 
    to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def create_sliding_window_causal_mask(batch_size, seq_len, device, dtype, sliding_window):
    """Create sliding window causal mask for attention."""
    mask = torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    
    # Add sliding window constraint
    for i in range(seq_len):
        start_idx = max(0, i - sliding_window + 1)
        mask[i, start_idx:i+1] = 0
    
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)


def create_causal_mask(batch_size, seq_len, device, dtype):
    """Create standard causal mask for attention."""
    mask = torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)


class Gemma3MLP(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        
        # Use proper activation function mapping
        if config.hidden_activation == "gelu_pytorch_tanh":
            self.act_fn = nn.GELU(approximate="tanh")
        elif config.hidden_activation == "gelu":
            self.act_fn = nn.GELU()
        elif config.hidden_activation == "swish":
            self.act_fn = lambda x: x * torch.sigmoid(x)
        else:
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
        self.attention_dropout = config.attention_dropout

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
            # Ensure attention_mask is on the same device and has the right shape
            attention_mask = attention_mask.to(attn_weights.device, dtype=attn_weights.dtype)
            if attention_mask.shape[-1] != key_states.shape[-2]:
                # Handle case where attention mask might be shorter than sequence
                attention_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            attn_weights = attn_weights + attention_mask

        # Upcast attention to fp32 for numerical stability
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
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

    def forward(self, input_ids, attention_mask=None, position_ids=None, output_attentions=False, output_hidden_states=False):
        batch_size, seq_len = input_ids.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        else:
            # Ensure position_ids is on the same device as input_ids
            position_ids = position_ids.to(input_ids.device)
        
        inputs_embeds = self.embed_tokens(input_ids)
        
        # Create appropriate attention mask based on layer types
        if attention_mask is None:
            # For now, use standard causal mask - in practice you'd want to create different masks for different layers
            attention_mask = create_causal_mask(batch_size, seq_len, input_ids.device, inputs_embeds.dtype)
        else:
            # Ensure attention_mask is on the correct device and dtype
            attention_mask = attention_mask.to(input_ids.device, dtype=inputs_embeds.dtype)
        
        position_embeddings_global = self.rotary_emb(inputs_embeds, position_ids)
        position_embeddings_local = self.rotary_emb_local(inputs_embeds, position_ids)
        
        hidden_states = inputs_embeds
        
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            hidden_states = layer(
                hidden_states,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
                attention_mask=attention_mask,
            )
        
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        return {
            'last_hidden_state': hidden_states,
            'hidden_states': all_hidden_states,
            'attentions': all_attentions
        }


class Gemma3ForCausalLM(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.config = config
        self.model = Gemma3Model(config)
        # Use tied weights approach like the official implementation
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def to(self, device):
        """Override to ensure proper device handling."""
        super().to(device)
        return self

    def cuda(self, device=None):
        """Move model to CUDA device."""
        return self.to(device if device is not None else torch.device('cuda'))

    def cpu(self):
        """Move model to CPU."""
        return self.to(torch.device('cpu'))

    def forward(self, input_ids, attention_mask=None, position_ids=None, labels=None, 
                output_attentions=False, output_hidden_states=False):
        # Ensure all inputs are on the same device as the model
        device = next(self.parameters()).device
        
        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if position_ids is not None:
            position_ids = position_ids.to(device)
        if labels is not None:
            labels = labels.to(device)
        
        outputs = self.model(
            input_ids, 
            attention_mask=attention_mask, 
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        
        hidden_states = outputs['last_hidden_state']
        
        # Use tied weights (embedding weight as output projection)
        logits = torch.matmul(hidden_states, self.model.embed_tokens.weight.T)
        
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping
        
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs['hidden_states'],
            'attentions': outputs['attentions']
        }

    def generate(self, input_ids, max_length=50, temperature=1.0, do_sample=True, 
                 top_k=50, top_p=0.9, repetition_penalty=1.0, pad_token_id=None):
        self.eval()
        
        # Ensure input_ids is on the same device as the model
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        
        generated = input_ids.clone()
        pad_token_id = pad_token_id or self.config.pad_token_id
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                outputs = self(generated)
                logits = outputs['logits']
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(generated.shape[0]):
                        for previous_token in set(generated[i].tolist()):
                            if previous_token != pad_token_id:
                                next_token_logits[i, previous_token] /= repetition_penalty
                
                if do_sample:
                    # Apply top-k filtering
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                        next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                        next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                    
                    # Apply top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Stop if pad token is generated
                if next_token.item() == pad_token_id:
                    break
        
        return generated


def create_gemma3_model(vocab_size=32768, hidden_size=640, num_layers=18, num_heads=4, num_kv_heads=1):
    # Calculate head_dim based on the original structure
    head_dim = 256  # This should be fixed based on the original model
    
    config = Gemma3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,  # Use fixed head_dim
        intermediate_size=2048,  # Use original intermediate size
        max_position_embeddings=32768,
    )
    return Gemma3ForCausalLM(config)


def create_gemma3_model_flexible(vocab_size=32768, hidden_size=640, num_layers=18, num_heads=4, num_kv_heads=1, head_dim=None, intermediate_size=None):
    """
    Create a Gemma3 model with flexible configuration.
    
    Args:
        vocab_size: Size of the vocabulary
        hidden_size: Size of the hidden layers
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        num_kv_heads: Number of key-value heads (for grouped-query attention)
        head_dim: Dimension of each attention head (if None, calculated as hidden_size // num_heads)
        intermediate_size: Size of the intermediate layer in MLP (if None, calculated as hidden_size * 4)
    """
    if head_dim is None:
        head_dim = hidden_size // num_heads
    
    if intermediate_size is None:
        intermediate_size = hidden_size * 4
    
    config = Gemma3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
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
        
        # Test generation
        generated = model.generate(input_ids, max_length=20, temperature=0.8, do_sample=True)
        print(f"Generated sequence shape: {generated.shape}")
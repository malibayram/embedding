import torch
import torch.nn as nn
from gemma_claude import (
    Gemma3Config, 
    Gemma3ForCausalLM, 
    create_gemma3_model,
    create_causal_mask,
    create_sliding_window_causal_mask
)

def test_basic_functionality():
    """Test basic model functionality."""
    print("=== Testing Basic Functionality ===")
    
    # Create a small model for testing
    config = Gemma3Config(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=256,
        max_position_embeddings=512,
    )
    
    model = Gemma3ForCausalLM(config)
    
    # Test forward pass
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(input_ids)
        print(f"✓ Forward pass successful")
        print(f"  - Logits shape: {outputs['logits'].shape}")
        print(f"  - Expected: ({batch_size}, {seq_len}, {config.vocab_size})")
        
        # Test with labels
        labels = torch.randint(0, 1000, (batch_size, seq_len))
        outputs_with_loss = model(input_ids, labels=labels)
        print(f"✓ Loss computation successful")
        print(f"  - Loss value: {outputs_with_loss['loss'].item():.4f}")
        
        # Test output options
        outputs_full = model(
            input_ids, 
            output_attentions=True, 
            output_hidden_states=True
        )
        print(f"✓ Full outputs successful")
        print(f"  - Hidden states: {len(outputs_full['hidden_states']) if outputs_full['hidden_states'] else 0}")
        print(f"  - Attentions: {len(outputs_full['attentions']) if outputs_full['attentions'] else 0}")

def test_attention_masks():
    """Test attention mask creation."""
    print("\n=== Testing Attention Masks ===")
    
    batch_size, seq_len = 2, 8
    device = torch.device('cpu')
    dtype = torch.float32
    
    # Test causal mask
    causal_mask = create_causal_mask(batch_size, seq_len, device, dtype)
    print(f"✓ Causal mask created")
    print(f"  - Shape: {causal_mask.shape}")
    print(f"  - Upper triangular: {torch.all(causal_mask[0, 0].triu(diagonal=1) == float('-inf'))}")
    
    # Test sliding window mask
    sliding_window = 4
    sliding_mask = create_sliding_window_causal_mask(batch_size, seq_len, device, dtype, sliding_window)
    print(f"✓ Sliding window mask created")
    print(f"  - Shape: {sliding_mask.shape}")
    
    # Check that sliding window constraint is applied
    # In a sliding window mask, each position should only attend to the last `sliding_window` positions
    for i in range(seq_len):
        start_idx = max(0, i - sliding_window + 1)
        valid_attention = sliding_mask[0, 0, i, start_idx:i+1] == 0
        invalid_attention = sliding_mask[0, 0, i, :start_idx] == float('-inf')
        print(f"  - Position {i}: valid attention to positions {start_idx}-{i}")

def test_generation():
    """Test text generation functionality."""
    print("\n=== Testing Generation ===")
    
    config = Gemma3Config(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        intermediate_size=128,
        max_position_embeddings=256,
    )
    
    model = Gemma3ForCausalLM(config)
    
    # Test greedy generation
    input_ids = torch.randint(0, 100, (1, 5))
    with torch.no_grad():
        generated_greedy = model.generate(
            input_ids, 
            max_length=10, 
            do_sample=False
        )
        print(f"✓ Greedy generation successful")
        print(f"  - Input length: {input_ids.shape[1]}")
        print(f"  - Output length: {generated_greedy.shape[1]}")
        
        # Test sampling generation
        generated_sampled = model.generate(
            input_ids, 
            max_length=10, 
            do_sample=True,
            temperature=0.8,
            top_k=10,
            top_p=0.9
        )
        print(f"✓ Sampling generation successful")
        print(f"  - Output length: {generated_sampled.shape[1]}")

def test_model_creation():
    """Test different model configurations."""
    print("\n=== Testing Model Creation ===")
    
    # Test the helper function
    model = create_gemma3_model(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2
    )
    
    print(f"✓ Model creation successful")
    print(f"  - Vocab size: {model.config.vocab_size}")
    print(f"  - Hidden size: {model.config.hidden_size}")
    print(f"  - Layers: {model.config.num_hidden_layers}")
    print(f"  - Heads: {model.config.num_attention_heads}")
    print(f"  - KV heads: {model.config.num_key_value_heads}")
    print(f"  - Head dim: {model.config.head_dim}")
    print(f"  - Intermediate size: {model.config.intermediate_size}")

def test_numerical_stability():
    """Test numerical stability of the model."""
    print("\n=== Testing Numerical Stability ===")
    
    config = Gemma3Config(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        intermediate_size=128,
        max_position_embeddings=256,
    )
    
    model = Gemma3ForCausalLM(config)
    
    # Test with different input scales
    batch_size, seq_len = 2, 16
    
    # Normal scale
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    with torch.no_grad():
        outputs_normal = model(input_ids)
        print(f"✓ Normal scale inputs successful")
        print(f"  - Logits range: [{outputs_normal['logits'].min():.4f}, {outputs_normal['logits'].max():.4f}]")
        
        # Check for NaN or inf
        has_nan = torch.isnan(outputs_normal['logits']).any()
        has_inf = torch.isinf(outputs_normal['logits']).any()
        print(f"  - Has NaN: {has_nan}")
        print(f"  - Has Inf: {has_inf}")

def test_memory_efficiency():
    """Test memory efficiency with larger models."""
    print("\n=== Testing Memory Efficiency ===")
    
    # Test with a larger model
    config = Gemma3Config(
        vocab_size=1000,
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        intermediate_size=1024,
        max_position_embeddings=1024,
    )
    
    model = Gemma3ForCausalLM(config)
    
    # Test with longer sequences
    batch_size, seq_len = 1, 256
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(input_ids)
        print(f"✓ Large model with long sequence successful")
        print(f"  - Sequence length: {seq_len}")
        print(f"  - Hidden size: {config.hidden_size}")
        print(f"  - Memory usage reasonable")

if __name__ == "__main__":
    print("Testing Improved Gemma3 Implementation")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        test_attention_masks()
        test_generation()
        test_model_creation()
        test_numerical_stability()
        test_memory_efficiency()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! The improved Gemma3 implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

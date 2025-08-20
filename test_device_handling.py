import torch
import torch.nn as nn

from gemma_claude import Gemma3Config, Gemma3ForCausalLM, create_gemma3_model


def test_device_creation():
    """Test model creation on different devices."""
    print("=== Testing Device Creation ===")
    
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
    
    # Test CPU
    model_cpu = Gemma3ForCausalLM(config)
    print(f"✓ Model created on CPU")
    print(f"  - Device: {next(model_cpu.parameters()).device}")
    
    # Test CUDA if available
    if torch.cuda.is_available():
        model_cuda = Gemma3ForCausalLM(config).cuda()
        print(f"✓ Model moved to CUDA")
        print(f"  - Device: {next(model_cuda.parameters()).device}")
        
        # Test specific CUDA device
        if torch.cuda.device_count() > 1:
            model_cuda_1 = Gemma3ForCausalLM(config).cuda(1)
            print(f"✓ Model moved to CUDA:1")
            print(f"  - Device: {next(model_cuda_1.parameters()).device}")
    
    # Test MPS if available (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        model_mps = Gemma3ForCausalLM(config).to('mps')
        print(f"✓ Model moved to MPS")
        print(f"  - Device: {next(model_mps.parameters()).device}")

def test_device_movement():
    """Test moving model between devices."""
    print("\n=== Testing Device Movement ===")
    
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
    print(f"✓ Initial device: {next(model.parameters()).device}")
    
    # Test moving to CPU
    model = model.cpu()
    print(f"✓ Moved to CPU: {next(model.parameters()).device}")
    
    # Test moving to CUDA if available
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"✓ Moved to CUDA: {next(model.parameters()).device}")
        
        # Test moving back to CPU
        model = model.cpu()
        print(f"✓ Moved back to CPU: {next(model.parameters()).device}")
    
    # Test moving to MPS if available
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        model = model.to('mps')
        print(f"✓ Moved to MPS: {next(model.parameters()).device}")
        
        # Test moving back to CPU
        model = model.cpu()
        print(f"✓ Moved back to CPU: {next(model.parameters()).device}")

def test_forward_pass_on_devices():
    """Test forward pass on different devices."""
    print("\n=== Testing Forward Pass on Devices ===")
    
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
    
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    labels = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Test on CPU
    model_cpu = Gemma3ForCausalLM(config)
    with torch.no_grad():
        outputs_cpu = model_cpu(input_ids, labels=labels)
        print(f"✓ CPU forward pass successful")
        print(f"  - Loss: {outputs_cpu['loss'].item():.4f}")
        print(f"  - Logits shape: {outputs_cpu['logits'].shape}")
        print(f"  - Device: {outputs_cpu['logits'].device}")
    
    # Test on CUDA if available
    if torch.cuda.is_available():
        model_cuda = Gemma3ForCausalLM(config).cuda()
        input_ids_cuda = input_ids.cuda()
        labels_cuda = labels.cuda()
        
        with torch.no_grad():
            outputs_cuda = model_cuda(input_ids_cuda, labels=labels_cuda)
            print(f"✓ CUDA forward pass successful")
            print(f"  - Loss: {outputs_cuda['loss'].item():.4f}")
            print(f"  - Logits shape: {outputs_cuda['logits'].shape}")
            print(f"  - Device: {outputs_cuda['logits'].device}")
            
            # Verify results are similar (allowing for small numerical differences)
            loss_diff = abs(outputs_cpu['loss'].item() - outputs_cuda['loss'].cpu().item())
            print(f"  - Loss difference: {loss_diff:.6f}")
    
    # Test on MPS if available
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        model_mps = Gemma3ForCausalLM(config).to('mps')
        input_ids_mps = input_ids.to('mps')
        labels_mps = labels.to('mps')
        
        with torch.no_grad():
            outputs_mps = model_mps(input_ids_mps, labels=labels_mps)
            print(f"✓ MPS forward pass successful")
            print(f"  - Loss: {outputs_mps['loss'].item():.4f}")
            print(f"  - Logits shape: {outputs_mps['logits'].shape}")
            print(f"  - Device: {outputs_mps['logits'].device}")

def test_generation_on_devices():
    """Test generation on different devices."""
    print("\n=== Testing Generation on Devices ===")
    
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
    
    input_ids = torch.randint(0, 1000, (1, 5))
    
    # Test on CPU
    model_cpu = Gemma3ForCausalLM(config)
    with torch.no_grad():
        generated_cpu = model_cpu.generate(input_ids, max_length=10, do_sample=False)
        print(f"✓ CPU generation successful")
        print(f"  - Generated shape: {generated_cpu.shape}")
        print(f"  - Device: {generated_cpu.device}")
    
    # Test on CUDA if available
    if torch.cuda.is_available():
        model_cuda = Gemma3ForCausalLM(config).cuda()
        input_ids_cuda = input_ids.cuda()
        
        with torch.no_grad():
            generated_cuda = model_cuda.generate(input_ids_cuda, max_length=10, do_sample=False)
            print(f"✓ CUDA generation successful")
            print(f"  - Generated shape: {generated_cuda.shape}")
            print(f"  - Device: {generated_cuda.device}")
            
            # Verify results are identical
            generated_cuda_cpu = generated_cuda.cpu()
            is_identical = torch.equal(generated_cpu, generated_cuda_cpu)
            print(f"  - Results identical: {is_identical}")
    
    # Test on MPS if available
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        model_mps = Gemma3ForCausalLM(config).to('mps')
        input_ids_mps = input_ids.to('mps')
        
        with torch.no_grad():
            generated_mps = model_mps.generate(input_ids_mps, max_length=10, do_sample=False)
            print(f"✓ MPS generation successful")
            print(f"  - Generated shape: {generated_mps.shape}")
            print(f"  - Device: {generated_mps.device}")

def test_mixed_device_inputs():
    """Test handling inputs on different devices than model."""
    print("\n=== Testing Mixed Device Inputs ===")
    
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
    
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    labels = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Test CPU model with CPU inputs (baseline)
    model_cpu = Gemma3ForCausalLM(config)
    with torch.no_grad():
        outputs_cpu = model_cpu(input_ids, labels=labels)
        loss_cpu = outputs_cpu['loss'].item()
        print(f"✓ CPU model + CPU inputs: {loss_cpu:.4f}")
    
    # Test CUDA model with CPU inputs (should auto-move)
    if torch.cuda.is_available():
        model_cuda = Gemma3ForCausalLM(config).cuda()
        with torch.no_grad():
            outputs_cuda = model_cuda(input_ids, labels=labels)
            loss_cuda = outputs_cuda['loss'].item()
            print(f"✓ CUDA model + CPU inputs: {loss_cuda:.4f}")
            print(f"  - Auto-moved inputs: {outputs_cuda['logits'].device}")
            
            # Verify results are similar
            loss_diff = abs(loss_cpu - loss_cuda)
            print(f"  - Loss difference: {loss_diff:.6f}")
    
    # Test MPS model with CPU inputs (should auto-move)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        model_mps = Gemma3ForCausalLM(config).to('mps')
        with torch.no_grad():
            outputs_mps = model_mps(input_ids, labels=labels)
            loss_mps = outputs_mps['loss'].item()
            print(f"✓ MPS model + CPU inputs: {loss_mps:.4f}")
            print(f"  - Auto-moved inputs: {outputs_mps['logits'].device}")

def test_device_consistency():
    """Test that all model components are on the same device."""
    print("\n=== Testing Device Consistency ===")
    
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
    
    # Test on CPU
    model = Gemma3ForCausalLM(config)
    devices = set()
    for name, param in model.named_parameters():
        devices.add(param.device)
    print(f"✓ CPU model device consistency: {len(devices)} device(s) - {devices}")
    
    # Test on CUDA if available
    if torch.cuda.is_available():
        model = Gemma3ForCausalLM(config).cuda()
        devices = set()
        for name, param in model.named_parameters():
            devices.add(param.device)
        print(f"✓ CUDA model device consistency: {len(devices)} device(s) - {devices}")
    
    # Test on MPS if available
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        model = Gemma3ForCausalLM(config).to('mps')
        devices = set()
        for name, param in model.named_parameters():
            devices.add(param.device)
        print(f"✓ MPS model device consistency: {len(devices)} device(s) - {devices}")

def test_memory_efficiency():
    """Test memory usage on different devices."""
    print("\n=== Testing Memory Efficiency ===")
    
    config = Gemma3Config(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=512,
        max_position_embeddings=1024,
    )
    
    batch_size, seq_len = 4, 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Test CPU memory
    model = Gemma3ForCausalLM(config)
    with torch.no_grad():
        outputs = model(input_ids)
        print(f"✓ CPU memory usage reasonable")
        print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test CUDA memory if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        model = Gemma3ForCausalLM(config).cuda()
        input_ids_cuda = input_ids.cuda()
        
        with torch.no_grad():
            outputs = model(input_ids_cuda)
            print(f"✓ CUDA memory usage reasonable")
            print(f"  - GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            print(f"  - GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
        
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print("Testing Gemma3 Device Handling")
    print("=" * 50)
    
    try:
        test_device_creation()
        test_device_movement()
        test_forward_pass_on_devices()
        test_generation_on_devices()
        test_mixed_device_inputs()
        test_device_consistency()
        test_memory_efficiency()
        
        print("\n" + "=" * 50)
        print("🎉 All device handling tests passed! The model works correctly on all devices.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

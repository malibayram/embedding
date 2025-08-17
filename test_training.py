#!/usr/bin/env python3
"""
Test script to verify the training setup works correctly.
This script creates a small model and runs a few training steps.
"""

import os

import torch
from tokenizers import (Tokenizer, decoders, models, pre_tokenizers,
                        processors, trainers)
from transformers import PreTrainedTokenizerFast

from gemma_claude import create_gemma3_model
from gemma_trainer_claude import TrainingConfig, create_trainer
from text_dataset import create_or_load_dataset


def create_basic_tokenizer():
    """Create a basic tokenizer for testing."""
    # Create a new tokenizer
    tokenizer_obj = Tokenizer(models.BPE())
    tokenizer_obj.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer_obj.decoder = decoders.ByteLevel()
    tokenizer_obj.post_processor = processors.ByteLevel(trim_offsets=False)
    
    # Create trainer
    trainer = trainers.BpeTrainer(
        vocab_size=32000,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )
    
    # Wrap in PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>"
    )
    
    # Set pad_token_id
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")
    
    return tokenizer


def load_sample_data():
    """Load a small sample of data for testing."""
    try:
        with open("data.txt", 'r', encoding='utf-8') as f:
            text = f.read()
        # Take first 10000 characters for testing
        return text[:10000]
    except Exception as e:
        print(f"Error loading data: {e}")
        # Return some sample Turkish text
        return """
        Merhaba dünya! Bu bir test metnidir. Türkçe dilinde yazılmıştır.
        Bu metin, model eğitimi için kullanılacaktır. Gemma3 modeli ile
        Türkçe metin üretimi yapılacaktır. Bu test, eğitim sürecinin
        doğru çalışıp çalışmadığını kontrol etmek için yapılmaktadır.
        """


def main():
    print("=== Training Setup Test ===")
    
    # Create basic tokenizer
    print("Creating tokenizer...")
    tokenizer = create_basic_tokenizer()
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Load sample data
    print("Loading sample data...")
    text_data = load_sample_data()
    print(f"Sample text length: {len(text_data)} characters")
    
    # Split data
    split_idx = int(len(text_data) * 0.8)
    train_text = text_data[:split_idx]
    eval_text = text_data[split_idx:]
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = create_or_load_dataset(
        text=train_text,
        tokenizer=tokenizer,
        seq_len=128,  # Small sequence length for testing
        stride=64,
        device="cpu",
        cache_path="./test_cache/train"
    )
    
    eval_dataset = create_or_load_dataset(
        text=eval_text,
        tokenizer=tokenizer,
        seq_len=128,
        stride=64,
        device="cpu",
        cache_path="./test_cache/eval"
    )
    
    print(f"Train dataset: {len(train_dataset)} sequences")
    print(f"Eval dataset: {len(eval_dataset)} sequences")
    
    # Create tiny model
    print("Creating model...")
    model = create_gemma3_model(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Create training configuration
    config = TrainingConfig(
        learning_rate=1e-4,
        batch_size=2,
        gradient_accumulation_steps=2,
        max_epochs=1,
        seq_len=128,
        logging_steps=10,
        eval_steps=20,
        save_steps=50,
        output_dir="./test_checkpoints",
        device="auto",
        mixed_precision=False,  # Disable for testing
        warmup_steps=10,
        max_grad_norm=1.0,
        weight_decay=0.01
    )
    
    # Create trainer
    print("Creating trainer...")
    trainer = create_trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
        tokenizer=tokenizer
    )
    
    # Test a few training steps
    print("Testing training...")
    try:
        # Run just a few steps
        trainer.model.train()
        for i, batch in enumerate(trainer.train_dataloader):
            if i >= 5:  # Only test 5 batches
                break
            
            step_metrics = trainer.train_step(batch)
            print(f"Step {i+1}: Loss = {step_metrics['loss']:.4f}")
            
            if (i + 1) % trainer.config.gradient_accumulation_steps == 0:
                grad_norm = trainer.optimizer_step()
                print(f"  Grad norm: {grad_norm:.4f}")
        
        print("Training test completed successfully!")
        
        # Test evaluation
        print("Testing evaluation...")
        eval_metrics = trainer.evaluate()
        print(f"Evaluation metrics: {eval_metrics}")
        
        print("\n=== All Tests Passed! ===")
        print("The training setup is working correctly.")
        print("You can now run the full training script.")
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()

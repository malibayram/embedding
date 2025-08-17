#!/usr/bin/env python3
"""
Training script for Gemma3 model using Turkish text data.
This script automatically handles device selection, tokenizer loading, and dataset creation.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

from gemma_claude import create_gemma3_model
from gemma_trainer_claude import TrainingConfig, create_trainer
from text_dataset import create_or_load_dataset


def load_text_data(data_path: str) -> str:
    """Load text data from file."""
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Loaded {len(text)} characters from {data_path}")
        return text
    except Exception as e:
        print(f"Error loading text data: {e}")
        sys.exit(1)


def create_model(vocab_size: int, model_size: str = "tiny") -> torch.nn.Module:
    """Create a Gemma3 model with specified size."""
    model_configs = {
        "tiny": {
            "vocab_size": vocab_size,
            "hidden_size": 512,
            "num_layers": 6,
            "num_heads": 8,
            "num_kv_heads": 2,
        },
        "small": {
            "vocab_size": vocab_size,
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "num_kv_heads": 3,
        },
        "medium": {
            "vocab_size": vocab_size,
            "hidden_size": 1024,
            "num_layers": 16,
            "num_heads": 16,
            "num_kv_heads": 4,
        },
        "large": {
            "vocab_size": vocab_size,
            "hidden_size": 1536,
            "num_layers": 24,
            "num_heads": 24,
            "num_kv_heads": 6,
        }
    }
    
    if model_size not in model_configs:
        raise ValueError(f"Invalid model size: {model_size}. Choose from {list(model_configs.keys())}")
    
    config = model_configs[model_size]
    print(f"Creating {model_size} model with {config['num_layers']} layers and {config['hidden_size']} hidden size")
    
    return create_gemma3_model(**config)


def main():
    parser = argparse.ArgumentParser(description="Train Gemma3 model on Turkish text data")
    parser.add_argument("--data_path", type=str, default="data.txt", help="Path to text data file")
    parser.add_argument("--tokenizer_path", type=str, default="gemma-3-270m-tr-tokenizer", help="Path to tokenizer")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--model_size", type=str, default="tiny", choices=["tiny", "small", "medium", "large"], 
                       help="Model size to train")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length for training")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=3, help="Maximum epochs")
    parser.add_argument("--eval_ratio", type=float, default=0.1, help="Evaluation dataset ratio")
    parser.add_argument("--cache_dir", type=str, default="./dataset_cache", help="Directory to cache datasets")
    parser.add_argument("--resume_from", type=str, help="Resume training from checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cuda, mps, cpu)")
    
    args = parser.parse_args()
    
    print("=== Turkish Gemma3 Training Script ===")
    print(f"Data path: {args.data_path}")
    print(f"Tokenizer path: {args.tokenizer_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model size: {args.model_size}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Load tokenizer
    print("\n=== Loading Tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
    
    # Ensure pad_token_id is set
    if tokenizer.pad_token_id is None:
        print("Setting pad_token_id to 0")
        tokenizer.pad_token_id = 0
    
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Special tokens: {tokenizer.special_tokens_map}")
    print(f"Pad token ID: {tokenizer.pad_token_id}")
    
    # Load text data
    print("\n=== Loading Text Data ===")
    text_data = load_text_data(args.data_path)
    
    # Split data into train and eval
    split_idx = int(len(text_data) * (1 - args.eval_ratio))
    train_text = text_data[:split_idx]
    eval_text = text_data[split_idx:]
    
    print(f"Train text length: {len(train_text)} characters")
    print(f"Eval text length: {len(eval_text)} characters")
    
    # Create datasets
    print("\n=== Creating Datasets ===")
    train_cache_path = os.path.join(args.cache_dir, f"train_{args.model_size}_{args.seq_len}")
    eval_cache_path = os.path.join(args.cache_dir, f"eval_{args.model_size}_{args.seq_len}")
    
    # Test tokenization first
    print("Testing tokenization...")
    test_tokens = tokenizer.encode(train_text[:1000])
    print(f"Sample tokens: {test_tokens[:20]}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Pad token ID: {tokenizer.pad_token_id}")
    
    # Adjust sequence length if needed
    if args.seq_len > len(test_tokens):
        print(f"Warning: Sequence length ({args.seq_len}) is longer than available tokens ({len(test_tokens)})")
        args.seq_len = min(args.seq_len, len(test_tokens) - 1)
        print(f"Adjusted sequence length to: {args.seq_len}")
    
    # Use smaller stride for better coverage
    stride = max(1, args.seq_len // 4)  # 25% overlap instead of 50%
    
    train_dataset = create_or_load_dataset(
        text=train_text,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        stride=stride,
        device="cpu",  # Will be moved to device by trainer
        cache_path=train_cache_path
    )
    
    eval_dataset = create_or_load_dataset(
        text=eval_text,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        stride=stride,
        device="cpu",
        cache_path=eval_cache_path
    )
    
    print(f"Train dataset: {len(train_dataset)} sequences")
    print(f"Eval dataset: {len(eval_dataset)} sequences")
    
    # Check if datasets are empty
    if len(train_dataset) == 0:
        print("ERROR: Train dataset is empty!")
        print("This might be due to:")
        print("1. Text is too short for the sequence length")
        print("2. Tokenizer is not working properly")
        print("3. Sequence length is too large")
        print(f"Text length: {len(train_text)} characters")
        print(f"Tokenized length: {len(test_tokens)} tokens")
        print(f"Sequence length: {args.seq_len}")
        sys.exit(1)
    
    if len(eval_dataset) == 0:
        print("WARNING: Eval dataset is empty, using train dataset for evaluation")
        eval_dataset = train_dataset
    
    # Create model
    print("\n=== Creating Model ===")
    model = create_model(tokenizer.vocab_size, args.model_size)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create training configuration
    print("\n=== Training Configuration ===")
    config = TrainingConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_epochs=args.max_epochs,
        seq_len=args.seq_len,
        logging_steps=50,
        eval_steps=200,
        save_steps=500,
        output_dir=args.output_dir,
        device=args.device,
        mixed_precision=True,
        eval_dataset_ratio=args.eval_ratio,
        do_eval=True,
        warmup_steps=100,
        max_grad_norm=1.0,
        weight_decay=0.01
    )
    
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Max epochs: {config.max_epochs}")
    
    # Create trainer
    print("\n=== Creating Trainer ===")
    trainer = create_trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
        tokenizer=tokenizer
    )
    
    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"\n=== Resuming from checkpoint: {args.resume_from} ===")
        trainer.load_checkpoint(args.resume_from)
    
    # Start training
    print("\n=== Starting Training ===")
    try:
        trainer.train()
        print("\n=== Training Completed Successfully! ===")
    except KeyboardInterrupt:
        print("\n=== Training Interrupted by User ===")
        # Save final checkpoint
        final_checkpoint_dir = os.path.join(args.output_dir, "interrupted_model")
        trainer.save_checkpoint(final_checkpoint_dir)
        print(f"Checkpoint saved to: {final_checkpoint_dir}")
    except Exception as e:
        print(f"\n=== Training Failed: {e} ===")
        raise


if __name__ == "__main__":
    main()

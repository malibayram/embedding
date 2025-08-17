# Turkish Gemma3 Training

This directory contains scripts to train a Gemma3 model on Turkish text data using the `data.txt` file.

## Files Overview

- `train_script.py` - Main training script with full functionality
- `test_training.py` - Test script to verify the setup works
- `gemma_trainer_claude.py` - Enhanced trainer with automatic device selection
- `text_dataset.py` - Dataset handling for text data
- `gemma_claude.py` - Gemma3 model implementation
- `data.txt` - Turkish text data for training

## Quick Start

### 1. Test the Setup

First, test that everything works correctly:

```bash
python test_training.py
```

This will:

- Create a small test model
- Load a sample of your data
- Run a few training steps
- Verify the setup is working

### 2. Start Training

Once the test passes, start the full training:

```bash
python train_script.py --model_size tiny --max_epochs 3
```

## Training Script Options

The `train_script.py` supports many command-line options:

### Basic Options

```bash
python train_script.py \
    --data_path data.txt \
    --output_dir ./checkpoints \
    --model_size tiny \
    --max_epochs 3
```

### Advanced Options

```bash
python train_script.py \
    --data_path data.txt \
    --tokenizer_path gemma-3-270m-tr-tokenizer \
    --output_dir ./checkpoints \
    --model_size small \
    --seq_len 512 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --max_epochs 5 \
    --eval_ratio 0.1 \
    --device auto
```

### All Available Options

| Option                          | Default                     | Description                            |
| ------------------------------- | --------------------------- | -------------------------------------- |
| `--data_path`                   | `data.txt`                  | Path to text data file                 |
| `--tokenizer_path`              | `gemma-3-270m-tr-tokenizer` | Path to tokenizer                      |
| `--output_dir`                  | `./checkpoints`             | Output directory for checkpoints       |
| `--model_size`                  | `tiny`                      | Model size: tiny, small, medium, large |
| `--seq_len`                     | `512`                       | Sequence length for training           |
| `--batch_size`                  | `4`                         | Batch size                             |
| `--gradient_accumulation_steps` | `4`                         | Gradient accumulation steps            |
| `--learning_rate`               | `5e-5`                      | Learning rate                          |
| `--max_epochs`                  | `3`                         | Maximum epochs                         |
| `--eval_ratio`                  | `0.1`                       | Evaluation dataset ratio               |
| `--cache_dir`                   | `./dataset_cache`           | Directory to cache datasets            |
| `--resume_from`                 | None                        | Resume training from checkpoint        |
| `--device`                      | `auto`                      | Device to use (auto, cuda, mps, cpu)   |

## Model Sizes

The script supports different model sizes optimized for different hardware:

| Size     | Layers | Hidden Size | Parameters | Memory (GB) | Use Case                 |
| -------- | ------ | ----------- | ---------- | ----------- | ------------------------ |
| `tiny`   | 6      | 512         | ~2M        | 0.5         | Testing, learning        |
| `small`  | 12     | 768         | ~8M        | 1.5         | Development, prototyping |
| `medium` | 16     | 1024        | ~20M       | 3.5         | Small-scale production   |
| `large`  | 24     | 1536        | ~50M       | 8.0         | Production               |

## Device Selection

The script automatically selects the best available device:

- **CUDA (NVIDIA GPU)**: Primary choice with full mixed precision
- **MPS (Apple Silicon)**: For Macs with M1/M2 chips
- **CPU**: Fallback option

You can override with `--device`:

```bash
python train_script.py --device cuda  # Force CUDA
python train_script.py --device mps   # Force MPS
python train_script.py --device cpu   # Force CPU
```

## Training Features

### Automatic Features

- **Device Detection**: Automatically selects best available device
- **Mixed Precision**: Enabled automatically for supported devices
- **Gradient Clipping**: Prevents gradient explosion
- **Learning Rate Scheduling**: Cosine annealing with warmup
- **Checkpointing**: Automatic saving and resuming
- **Dataset Caching**: Fast reloading of processed datasets

### Monitoring

- **Real-time Logging**: Loss, learning rate, gradient norms
- **Evaluation**: Regular evaluation on validation set
- **Checkpoints**: Save best model and regular checkpoints
- **Metrics**: Loss, perplexity, training speed

## Resuming Training

To resume from a checkpoint:

```bash
python train_script.py \
    --resume_from ./checkpoints/checkpoint-1000 \
    --max_epochs 5
```

## Output Structure

After training, you'll find:

```
checkpoints/
├── checkpoint-500/          # Regular checkpoints
├── checkpoint-1000/
├── best_model/              # Best model based on eval loss
├── final_model/             # Final model after training
└── logs/
    └── training.log         # Training logs
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or model size

   ```bash
   python train_script.py --batch_size 2 --model_size tiny
   ```

2. **Slow Training**: Enable mixed precision and use GPU

   ```bash
   python train_script.py --device auto
   ```

3. **Tokenizer Issues**: The script will create a basic tokenizer if needed

4. **Dataset Loading**: Check that `data.txt` exists and is readable

### Performance Tips

- Use the largest model that fits in your memory
- Enable mixed precision (automatic with `--device auto`)
- Use gradient accumulation for larger effective batch sizes
- Cache datasets for faster reloading
- Use appropriate sequence length (512-1024 for most cases)

## Example Training Runs

### Quick Test (5 minutes)

```bash
python train_script.py --model_size tiny --max_epochs 1 --seq_len 256
```

### Development Training (30 minutes)

```bash
python train_script.py --model_size small --max_epochs 3 --seq_len 512
```

### Production Training (2+ hours)

```bash
python train_script.py \
    --model_size medium \
    --max_epochs 10 \
    --seq_len 1024 \
    --batch_size 8 \
    --gradient_accumulation_steps 8
```

## Requirements

Make sure you have the required packages:

```bash
pip install torch transformers tokenizers
```

For GPU support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For Apple Silicon:

```bash
pip install torch torchvision torchaudio
```

## Next Steps

After training, you can:

1. **Load the trained model** for inference
2. **Fine-tune** on specific tasks
3. **Export** to different formats
4. **Deploy** for production use

The trained model will be saved in the `checkpoints` directory and can be loaded using PyTorch's `torch.load()` function.

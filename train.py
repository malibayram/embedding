import os

import torch
# Load model directly
from transformers import AutoModelForCausalLM, AutoTokenizer

from gemma_claude import create_gemma3_model
from gemma_trainer_claude import TrainingConfig, create_trainer
from text_dataset import create_dataloader, create_dataset

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it")

model_claude = create_gemma3_model(vocab_size=262144)
# Automatic device detection
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

model_claude.load_state_dict(model.state_dict())


with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

seq_len = 512
stride = 1
batch_size = 16
shuffle = False

train_dataset = create_dataset(text[:10000], tokenizer, seq_len, stride, device)
# dataset = create_or_load_dataset(text[:50000], tr_tokenizer, seq_len, stride, device, cache_path="dataset/wiki1")
train_dataloader = create_dataloader(train_dataset, batch_size, shuffle)

valid_dataset = create_dataset(text[10000:11000], tokenizer, seq_len, stride, device)
valid_dataloader = create_dataloader(valid_dataset, batch_size, shuffle)


config = TrainingConfig(
    learning_rate=5e-5,
    batch_size=batch_size,
    gradient_accumulation_steps=2,
    max_epochs=10,
    seq_len=seq_len,
    logging_steps=50,
    eval_steps=200,
    save_steps=500,
    output_dir="./test_checkpoints",
    device="auto",
    mixed_precision=True,
    eval_dataset_ratio=0.1,
    do_eval=True,
    warmup_steps=100,
    max_grad_norm=1.0,
    weight_decay=0.01
)


trainer = create_trainer(
        model=model_claude,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        config=config,
        tokenizer=tokenizer
    )


try:
    trainer.train()
    print("\n=== Training Completed Successfully! ===")
except KeyboardInterrupt:
    print("\n=== Training Interrupted by User ===")
    # Save final checkpoint
    final_checkpoint_dir = os.path.join("./test_checkpoints", "interrupted_model")
    trainer.save_checkpoint(final_checkpoint_dir)
    print(f"Checkpoint saved to: {final_checkpoint_dir}")
except Exception as e:
    print(f"\n=== Training Failed: {e} ===")
    raise
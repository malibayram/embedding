import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrainingConfig:
    # Model parameters
    model_name: str = "gemma3"
    
    # Training parameters
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.01
    max_epochs: int = 10
    max_steps: Optional[int] = None
    warmup_steps: int = 1000
    warmup_ratio: float = 0.1
    
    # Sequence parameters
    seq_len: int = 512
    
    # Batch and gradient parameters
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Logging and saving
    logging_steps: int = 100
    eval_steps: Optional[int] = None
    save_steps: int = 1000
    save_total_limit: int = 3
    
    # Paths
    output_dir: str = "./checkpoints"
    logging_dir: Optional[str] = None
    
    # Device and precision
    device: str = "auto"
    mixed_precision: bool = True
    compile_model: bool = False
    
    # Evaluation
    eval_dataset_ratio: float = 0.1
    do_eval: bool = True
    
    # Optimizer
    optimizer_type: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    
    # Scheduler
    scheduler_type: str = "cosine_with_warmup"
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = self._get_best_device()
        
        if self.logging_dir is None:
            self.logging_dir = os.path.join(self.output_dir, "logs")
        
        if self.eval_steps is None:
            self.eval_steps = self.save_steps
    
    def _get_best_device(self) -> str:
        """Automatically select the best available device."""
        # Check for CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            # Get GPU info
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                # Get the first GPU's memory info
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                print(f"Found {gpu_count} CUDA device(s), using GPU 0 with {gpu_memory:.1f}GB memory")
                return "cuda"
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("Found Apple Silicon GPU (MPS), using MPS device")
            return "mps"
        
        # Check for ROCm (AMD GPU)
        if hasattr(torch.backends, 'rocm') and torch.backends.rocm.is_available():
            print("Found AMD GPU (ROCm), using ROCm device")
            return "cuda"  # ROCm uses CUDA API
        
        # Fallback to CPU
        print("No GPU found, using CPU")
        return "cpu"


class TrainingMetrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.losses = []
        self.learning_rates = []
        self.grad_norms = []
        self.step_times = []
        
    def update(self, loss: float, lr: float, grad_norm: float, step_time: float):
        self.losses.append(loss)
        self.learning_rates.append(lr)
        self.grad_norms.append(grad_norm)
        self.step_times.append(step_time)
    
    def get_averages(self, last_n: int = None) -> Dict[str, float]:
        if last_n is None:
            losses = self.losses
            step_times = self.step_times
        else:
            losses = self.losses[-last_n:]
            step_times = self.step_times[-last_n:]
        
        if not losses:
            return {}
        
        return {
            "avg_loss": sum(losses) / len(losses),
            "avg_step_time": sum(step_times) / len(step_times),
            "current_lr": self.learning_rates[-1] if self.learning_rates else 0.0,
            "last_grad_norm": self.grad_norms[-1] if self.grad_norms else 0.0,
        }


class Gemma3Trainer:
    def __init__(
        self,
        model,
        config: TrainingConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        tokenizer = None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        
        # Create output directories first
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.logging_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Setup device and model
        self.setup_device_and_model()
        
        # Create data loaders
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if config.device != "cpu" else False
        )
        
        if eval_dataset is not None:
            self.eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True if config.device != "cpu" else False
            )
        else:
            self.eval_dataloader = None
        
        # Setup optimizer and scheduler
        self.setup_optimizer_and_scheduler()
        
        # Mixed precision scaler
        if config.mixed_precision and config.device == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
        elif config.mixed_precision and config.device == "mps":
            # MPS doesn't have GradScaler, but we can still use autocast
            self.scaler = None
        else:
            self.scaler = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.metrics = TrainingMetrics()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.logging_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_device_and_model(self):
        # Move model to device
        self.model = self.model.to(self.config.device)
        
        # Log device information
        self._log_device_info()
        
        # Model compilation (only for CUDA devices)
        if self.config.compile_model and hasattr(torch, 'compile') and self.config.device == "cuda":
            try:
                self.model = torch.compile(self.model)
                self.logger.info("Model compiled successfully")
            except Exception as e:
                self.logger.warning(f"Model compilation failed: {e}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model moved to {self.config.device}")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def _log_device_info(self):
        """Log detailed information about the selected device."""
        device = self.config.device
        
        if device == "cuda":
            gpu_count = torch.cuda.device_count()
            current_gpu = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_gpu)
            gpu_memory = torch.cuda.get_device_properties(current_gpu).total_memory / (1024**3)
            self.logger.info(f"Using CUDA device {current_gpu}: {gpu_name} ({gpu_memory:.1f}GB)")
            
        elif device == "mps":
            self.logger.info("Using Apple Silicon GPU (MPS)")
            
        elif device == "cpu":
            import platform
            cpu_info = platform.processor()
            self.logger.info(f"Using CPU: {cpu_info}")
            
        # Log mixed precision availability
        if self.config.mixed_precision:
            if device == "cuda":
                self.logger.info("Mixed precision training enabled (CUDA AMP)")
            elif device == "mps":
                self.logger.info("Mixed precision training enabled (MPS)")
            else:
                self.logger.info("Mixed precision disabled (CPU)")
                self.config.mixed_precision = False
    
    def setup_optimizer_and_scheduler(self):
        # Setup optimizer
        if self.config.optimizer_type == "adamw":
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.epsilon
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")
        
        # Calculate total training steps
        if self.config.max_steps is not None:
            total_steps = self.config.max_steps
        else:
            steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
            total_steps = steps_per_epoch * self.config.max_epochs
        
        # Setup scheduler
        if self.config.scheduler_type == "cosine_with_warmup":
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=self.config.min_learning_rate / self.config.learning_rate,
                end_factor=1.0,
                total_iters=self.config.warmup_steps
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - self.config.warmup_steps,
                eta_min=self.config.min_learning_rate
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.config.warmup_steps]
            )
        else:
            self.scheduler = None
        
        self.logger.info(f"Optimizer: {self.config.optimizer_type}")
        self.logger.info(f"Scheduler: {self.config.scheduler_type}")
        self.logger.info(f"Total training steps: {total_steps}")
    
    def train_step(self, batch) -> Dict[str, float]:
        start_time = time.time()
        
        input_ids, labels = batch
        input_ids = input_ids.to(self.config.device)
        labels = labels.to(self.config.device)
        
        # Handle mixed precision for different devices
        if self.config.mixed_precision and self.config.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs = self.model(input_ids, labels=labels)
                loss = outputs['loss'] / self.config.gradient_accumulation_steps
            
            self.scaler.scale(loss).backward()
        elif self.config.mixed_precision and self.config.device == "mps":
            with torch.autocast(device_type='mps', dtype=torch.float16):
                outputs = self.model(input_ids, labels=labels)
                loss = outputs['loss'] / self.config.gradient_accumulation_steps
            
            loss.backward()
        else:
            outputs = self.model(input_ids, labels=labels)
            loss = outputs['loss'] / self.config.gradient_accumulation_steps
            loss.backward()
        
        step_time = time.time() - start_time
        
        return {
            'loss': loss.item() * self.config.gradient_accumulation_steps,
            'step_time': step_time
        }
    
    def optimizer_step(self) -> float:
        # Gradient clipping
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        return grad_norm.item()
    
    def evaluate(self) -> Dict[str, float]:
        if self.eval_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_steps = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                input_ids, labels = batch
                input_ids = input_ids.to(self.config.device)
                labels = labels.to(self.config.device)
                
                # Handle mixed precision for different devices
                if self.config.mixed_precision and self.config.device == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = self.model(input_ids, labels=labels)
                        loss = outputs['loss']
                elif self.config.mixed_precision and self.config.device == "mps":
                    with torch.autocast(device_type='mps', dtype=torch.float16):
                        outputs = self.model(input_ids, labels=labels)
                        loss = outputs['loss']
                else:
                    outputs = self.model(input_ids, labels=labels)
                    loss = outputs['loss']
                
                total_loss += loss.item()
                total_steps += 1
        
        self.model.train()
        
        avg_loss = total_loss / total_steps if total_steps > 0 else float('inf')
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
        
        return {
            'eval_loss': avg_loss,
            'eval_perplexity': perplexity
        }
    
    def save_checkpoint(self, checkpoint_dir: str):
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
        
        # Save optimizer state
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
        
        # Save scheduler state
        if self.scheduler is not None:
            torch.save(self.scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
        
        # Save training state
        training_state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'config': asdict(self.config)
        }
        
        with open(os.path.join(checkpoint_dir, "training_state.json"), "w") as f:
            json.dump(training_state, f, indent=2)
        
        self.logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str):
        # Load model state
        model_path = os.path.join(checkpoint_dir, "model.pt")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.config.device))
            self.logger.info("Model state loaded")
        
        # Load optimizer state
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.config.device))
            self.logger.info("Optimizer state loaded")
        
        # Load scheduler state
        scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
        if os.path.exists(scheduler_path) and self.scheduler is not None:
            self.scheduler.load_state_dict(torch.load(scheduler_path, map_location=self.config.device))
            self.logger.info("Scheduler state loaded")
        
        # Load training state
        training_state_path = os.path.join(checkpoint_dir, "training_state.json")
        if os.path.exists(training_state_path):
            with open(training_state_path, "r") as f:
                training_state = json.load(f)
            
            self.global_step = training_state['global_step']
            self.epoch = training_state['epoch']
            self.best_loss = training_state['best_loss']
            self.logger.info("Training state loaded")
    
    def cleanup_checkpoints(self):
        if self.config.save_total_limit <= 0:
            return
        
        checkpoint_dirs = []
        for item in os.listdir(self.config.output_dir):
            if item.startswith("checkpoint-"):
                checkpoint_dirs.append(item)
        
        checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
        
        while len(checkpoint_dirs) > self.config.save_total_limit:
            oldest = checkpoint_dirs.pop(0)
            oldest_path = os.path.join(self.config.output_dir, oldest)
            
            import shutil
            shutil.rmtree(oldest_path)
            self.logger.info(f"Removed old checkpoint: {oldest}")
    
    def train(self):
        self.logger.info("Starting training...")
        self.logger.info(f"Training config: {asdict(self.config)}")
        
        self.model.train()
        accumulation_loss = 0.0
        
        for epoch in range(self.config.max_epochs):
            self.epoch = epoch
            self.logger.info(f"Epoch {epoch + 1}/{self.config.max_epochs}")
            
            for step, batch in enumerate(self.train_dataloader):
                # Training step
                step_metrics = self.train_step(batch)
                accumulation_loss += step_metrics['loss']
                
                # Check if we should perform optimizer step
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    grad_norm = self.optimizer_step()
                    self.global_step += 1
                    
                    # Update metrics
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.metrics.update(
                        loss=accumulation_loss,
                        lr=current_lr,
                        grad_norm=grad_norm,
                        step_time=step_metrics['step_time']
                    )
                    
                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        avg_metrics = self.metrics.get_averages(self.config.logging_steps)
                        self.logger.info(
                            f"Step {self.global_step}: "
                            f"Loss={avg_metrics['avg_loss']:.4f}, "
                            f"LR={avg_metrics['current_lr']:.2e}, "
                            f"Grad Norm={avg_metrics['last_grad_norm']:.4f}, "
                            f"Step Time={avg_metrics['avg_step_time']:.3f}s"
                        )
                    
                    # Evaluation
                    if self.config.do_eval and self.global_step % self.config.eval_steps == 0:
                        eval_metrics = self.evaluate()
                        if eval_metrics:
                            self.logger.info(
                                f"Eval Step {self.global_step}: "
                                f"Loss={eval_metrics['eval_loss']:.4f}, "
                                f"Perplexity={eval_metrics['eval_perplexity']:.2f}"
                            )
                            
                            # Save best model
                            if eval_metrics['eval_loss'] < self.best_loss:
                                self.best_loss = eval_metrics['eval_loss']
                                self.save_checkpoint(os.path.join(self.config.output_dir, "best_model"))
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
                        self.save_checkpoint(checkpoint_dir)
                        self.cleanup_checkpoints()
                    
                    accumulation_loss = 0.0
                    
                    # Check if we've reached max steps
                    if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                        self.logger.info(f"Reached max steps ({self.config.max_steps})")
                        return
        
        # Save final checkpoint
        final_checkpoint_dir = os.path.join(self.config.output_dir, "final_model")
        self.save_checkpoint(final_checkpoint_dir)
        self.logger.info("Training completed!")


def create_trainer(
    model,
    train_dataset,
    eval_dataset=None,
    config: Optional[TrainingConfig] = None,
    **config_kwargs
) -> Gemma3Trainer:
    """
    Create a trainer with the given model and datasets.
    
    Args:
        model: The Gemma3 model to train
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
        config: Training configuration (if None, uses default with config_kwargs)
        **config_kwargs: Additional configuration parameters
    
    Returns:
        Configured Gemma3Trainer
    """
    if config is None:
        config = TrainingConfig(**config_kwargs)
    
    return Gemma3Trainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )


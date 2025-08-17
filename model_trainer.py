import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm


class ModelTrainer:
    """
    An optimized and reusable trainer class for PyTorch models.

    This class encapsulates the training loop, validation, checkpointing,
    and other common training functionalities.
    """
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, scheduler: optim.lr_scheduler._LRScheduler, device: str, epochs: int,
                 checkpoint_dir='checkpoints', checkpoint_name='best_model.pt', patience=3, grad_clip_value=1.0):
        """
        Initializes the ModelTrainer.

        Args:
            model (nn.Module): The PyTorch model to be trained.
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            optimizer (optim.Optimizer): The optimizer for training (e.g., Adam).
            criterion: The loss function.
            scheduler: Learning rate scheduler.
            device (torch.device): The device to train on ('cuda' or 'cpu').
            epochs (int): The total number of epochs to train for.
            checkpoint_dir (str): Directory to save model checkpoints.
            checkpoint_name (str): Filename for the model checkpoint.
            patience (int): Number of epochs to wait for improvement before early stopping.
            grad_clip_value (float): The value to clip gradients at to prevent exploding gradients.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.epochs = epochs
        self.checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        self.patience = patience
        self.grad_clip_value = grad_clip_value
        
        # Internal state
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _train_epoch(self):
        """Performs a single training epoch and returns the average loss."""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False, unit="batch")
        
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # The output shape might be (batch, seq_len, vocab_size) and target (batch, seq_len)
            # We need to flatten them for the loss function
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            # Backward pass and optimization
            loss.backward()
            
            # Gradient Clipping to prevent exploding gradients
            if self.grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        return total_loss / len(self.train_loader)

    def _validate_epoch(self):
        """Performs a single validation epoch and returns the average loss."""
        self.model.eval()
        total_loss = 0
        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False, unit="batch")
        with torch.no_grad():
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                total_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
                
        return total_loss / len(self.val_loader)

    def train(self):
        """
        The main training loop that orchestrates training, validation,
        and checkpointing over all epochs.
        """
        print(f"Starting training on device: {self.device}")
        
        for epoch in range(self.epochs):
            train_loss = self._train_epoch()
            val_loss = self._validate_epoch()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:02d}/{self.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
            
            # Step the learning rate scheduler
            self.scheduler.step()
            
            # Handle early stopping and save the best model
            if val_loss < self.best_val_loss:
                print(f"Validation loss decreased ({self.best_val_loss:.4f} --> {val_loss:.4f}). Saving model...")
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                self.save_checkpoint()
            else:
                self.epochs_no_improve += 1
                print(f"Validation loss did not improve. Counter: {self.epochs_no_improve}/{self.patience}")
                if self.epochs_no_improve >= self.patience:
                    print(f"Early stopping triggered after {self.patience} epochs without improvement.")
                    break
        
        print("Training finished.")
        # Load the best performing model weights back into the model
        self.load_checkpoint()

    def save_checkpoint(self):
        """Saves the model's state_dict to the checkpoint file."""
        torch.save(self.model.state_dict(), self.checkpoint_path)

    def load_checkpoint(self):
        """Loads the model's state_dict from the checkpoint file."""
        if os.path.exists(self.checkpoint_path):
            print(f"Loading best model weights from {self.checkpoint_path}")
            self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        else:
            print("Warning: No checkpoint found. Model weights are not loaded.")

import logging
from typing import List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    A PyTorch Dataset for text data with tokenization and sliding window context.
    
    This dataset creates overlapping sequences from tokenized text for language modeling tasks.
    Each sequence has a fixed context length and the dataset uses a sliding window approach
    with configurable stride for data augmentation.
    """
    
    def __init__(
        self,
        token_ids: List[int],
        context_length: int,
        stride: int = 1,
        pad_token_id: int = 0,
        eos_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        truncate_long_sequences: bool = True
    ):
        """
        Initialize the TextDataset.
        
        Args:
            token_ids: List of token IDs from the tokenizer
            context_length: Length of each sequence (input + target)
            stride: Step size for sliding window (default: 1 for maximum overlap)
            pad_token_id: Token ID used for padding (default: 0)
            eos_token_id: End of sequence token ID (optional)
            bos_token_id: Beginning of sequence token ID (optional)
            truncate_long_sequences: Whether to truncate sequences longer than context_length
        """
        self.token_ids = token_ids
        self.context_length = context_length
        self.stride = stride
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.truncate_long_sequences = truncate_long_sequences
        
        # Validate inputs
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        if len(token_ids) == 0:
            raise ValueError("token_ids cannot be empty")
        
        # Prepare sequences
        self.sequences = self._prepare_sequences()
        
        logger.info(f"Created dataset with {len(self.sequences)} sequences "
                   f"from {len(token_ids)} tokens (context_length={context_length}, stride={stride})")
    
    def _prepare_sequences(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Prepare input-target pairs using sliding window approach.
        
        Returns:
            List of (input_sequence, target_sequence) tuples
        """
        sequences = []
        
        # Calculate the number of sequences we can create
        if self.truncate_long_sequences:
            max_start_idx = len(self.token_ids) - self.context_length
        else:
            max_start_idx = len(self.token_ids) - 1
        
        if max_start_idx < 0:
            # If text is shorter than context_length, create a single padded sequence
            input_seq = self.token_ids + [self.pad_token_id] * (self.context_length - len(self.token_ids))
            target_seq = self.token_ids[1:] + [self.pad_token_id] * (self.context_length - len(self.token_ids) + 1)
            
            # Ensure both sequences are exactly context_length
            input_seq = input_seq[:self.context_length]
            target_seq = target_seq[:self.context_length]
            
            sequences.append((
                torch.tensor(input_seq, dtype=torch.long),
                torch.tensor(target_seq, dtype=torch.long)
            ))
        else:
            # Create sequences using sliding window
            for start_idx in range(0, max_start_idx + 1, self.stride):
                end_idx = start_idx + self.context_length
                
                # Extract input sequence (current tokens)
                input_seq = self.token_ids[start_idx:end_idx]
                
                # Extract target sequence (next tokens, shifted by 1)
                target_seq = self.token_ids[start_idx + 1:end_idx + 1]
                
                # Pad sequences if necessary
                if len(input_seq) < self.context_length:
                    input_seq = input_seq + [self.pad_token_id] * (self.context_length - len(input_seq))
                if len(target_seq) < self.context_length:
                    target_seq = target_seq + [self.pad_token_id] * (self.context_length - len(target_seq))
                
                # Truncate if longer than context_length
                input_seq = input_seq[:self.context_length]
                target_seq = target_seq[:self.context_length]
                
                sequences.append((
                    torch.tensor(input_seq, dtype=torch.long),
                    torch.tensor(target_seq, dtype=torch.long)
                ))
        
        return sequences
    
    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sequence pair by index."""
        if idx < 0 or idx >= len(self.sequences):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.sequences)}")
        return self.sequences[idx]
    
    def get_sequence_info(self) -> dict:
        """Get information about the dataset."""
        return {
            "total_sequences": len(self.sequences),
            "context_length": self.context_length,
            "stride": self.stride,
            "original_token_count": len(self.token_ids),
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "bos_token_id": self.bos_token_id
        }


def prepare_dataset(
    tokenizer,
    text: Union[str, List[str]],
    context_length: int,
    stride: int = 1,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    **dataset_kwargs
) -> DataLoader:
    """
    Prepare a DataLoader from text using a tokenizer.
    
    Args:
        tokenizer: HuggingFace tokenizer or compatible tokenizer
        text: Input text (string or list of strings)
        context_length: Length of each sequence
        stride: Step size for sliding window
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop the last incomplete batch
        **dataset_kwargs: Additional arguments passed to TextDataset
    
    Returns:
        DataLoader configured for training
    """
    # Tokenize the text
    if isinstance(text, str):
        # Single text string
        token_ids = tokenizer.encode(text, add_special_tokens=True)
    elif isinstance(text, list):
        # List of text strings - concatenate with separator
        if hasattr(tokenizer, 'sep_token_id') and tokenizer.sep_token_id is not None:
            separator = [tokenizer.sep_token_id]
        else:
            separator = []
        
        token_ids = []
        for i, t in enumerate(text):
            if i > 0:
                token_ids.extend(separator)
            token_ids.extend(tokenizer.encode(t, add_special_tokens=False))
        
        # Add special tokens at the beginning and end
        if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            token_ids = [tokenizer.bos_token_id] + token_ids
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            token_ids.append(tokenizer.eos_token_id)
    else:
        raise TypeError("text must be a string or list of strings")
    
    # Get tokenizer-specific IDs
    pad_token_id = getattr(tokenizer, 'pad_token_id', 0)
    eos_token_id = getattr(tokenizer, 'eos_token_id', None)
    bos_token_id = getattr(tokenizer, 'bos_token_id', None)
    
    # Create dataset
    dataset = TextDataset(
        token_ids=token_ids,
        context_length=context_length,
        stride=stride,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
        **dataset_kwargs
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )
    
    logger.info(f"Created DataLoader with {len(dataloader)} batches "
               f"(batch_size={batch_size}, total_sequences={len(dataset)})")
    
    return dataloader


def create_multiple_datasets(
    tokenizer,
    texts: List[str],
    context_length: int,
    stride: int = 1,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    batch_size: int = 32,
    shuffle: bool = True,
    **dataloader_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test datasets from a list of texts.
    
    Args:
        tokenizer: HuggingFace tokenizer or compatible tokenizer
        texts: List of text strings
        context_length: Length of each sequence
        stride: Step size for sliding window
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        batch_size: Batch size for DataLoaders
        shuffle: Whether to shuffle the training data
        **dataloader_kwargs: Additional arguments passed to DataLoader
    
    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader)
    """
    # Validate splits
    total_split = train_split + val_split + test_split
    if abs(total_split - 1.0) > 1e-6:
        raise ValueError(f"Train, validation, and test splits must sum to 1.0, got {total_split}")
    
    # Calculate split indices
    total_texts = len(texts)
    train_end = int(total_texts * train_split)
    val_end = int(total_texts * (train_split + val_split))
    
    # Split texts
    train_texts = texts[:train_end]
    val_texts = texts[train_end:val_end]
    test_texts = texts[val_end:]
    
    logger.info(f"Split {total_texts} texts: {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test")
    
    # Create dataloaders
    train_dataloader = prepare_dataset(
        tokenizer, train_texts, context_length, stride, batch_size, shuffle=shuffle, **dataloader_kwargs
    )
    
    val_dataloader = prepare_dataset(
        tokenizer, val_texts, context_length, stride, batch_size, shuffle=False, **dataloader_kwargs
    )
    
    test_dataloader = prepare_dataset(
        tokenizer, test_texts, context_length, stride, batch_size, shuffle=False, **dataloader_kwargs
    )
    
    return train_dataloader, val_dataloader, test_dataloader


# Example usage and testing
if __name__ == "__main__":
    # Example with a simple tokenizer-like object
    class SimpleTokenizer:
        def __init__(self):
            self.vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
            self.pad_token_id = 0
            self.unk_token_id = 1
            self.bos_token_id = 2
            self.eos_token_id = 3
        
        def encode(self, text, add_special_tokens=True):
            # Simple character-level tokenization for demo
            tokens = [ord(c) + 10 for c in text]  # Offset by 10 to avoid special tokens
            if add_special_tokens:
                tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
            return tokens
    
    # Test the dataset preparation
    tokenizer = SimpleTokenizer()
    sample_text = "Hello world! This is a test of the dataset preparation."
    
    print("Testing dataset preparation...")
    dataloader = prepare_dataset(
        tokenizer=tokenizer,
        text=sample_text,
        context_length=10,
        stride=2,
        batch_size=2,
        shuffle=False
    )
    
    print(f"Created dataloader with {len(dataloader)} batches")
    
    # Show first batch
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"Inputs shape: {inputs.shape}")
        print(f"Targets shape: {targets.shape}")
        print(f"Sample input: {inputs[0]}")
        print(f"Sample target: {targets[0]}")
        break

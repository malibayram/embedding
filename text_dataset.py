import json
import os
import pickle
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    """
    A simple dataset to handle text data with save/load functionality.
    """
    def __init__(self, text: str, tokenizer, seq_len: int, stride: int = 1, device: str = "cpu"):
        self.tokens = tokenizer.encode(text)
        self.seq_len = seq_len
        self.device = device
        self.stride = stride
        self.tokenizer_name = getattr(tokenizer, 'name_or_path', 'unknown')
        self.tokenizer = tokenizer
        
        # Debug information
        print(f"TextDataset: {len(self.tokens)} tokens, seq_len={seq_len}, stride={stride}")
        print(f"Calculated sequences: {len(self)}")
        
        if len(self) == 0:
            print(f"WARNING: Dataset will be empty!")
            print(f"Tokens: {len(self.tokens)}, Seq len: {seq_len}, Stride: {stride}")
            print(f"Formula: max(0, ({len(self.tokens)} - {seq_len}) // {stride} + 1)")
            print(f"Result: max(0, {len(self.tokens) - seq_len} // {stride} + 1)")
            print(f"Final: max(0, {(len(self.tokens) - seq_len) // stride + 1})")

    def __len__(self):
        # Calculate number of sequences with stride
        return max(0, (len(self.tokens) - self.seq_len) // self.stride + 1)

    def __getitem__(self, idx):
        # Use stride to calculate the starting position
        start_idx = idx * self.stride
        chunk = self.tokens[start_idx:start_idx + self.seq_len + 1]
        pad_token_id = self.tokenizer.pad_token_id
        # Ensure we have exactly seq_len + 1 tokens by padding if necessary
        if len(chunk) < self.seq_len + 1:
            # Pad with the last token to maintain sequence length
            padding_needed = self.seq_len + 1 - len(chunk)
            if len(chunk) > 0:
                chunk = chunk + [pad_token_id] * padding_needed
            else:
                # Handle edge case where chunk is empty
                chunk = [pad_token_id] * (self.seq_len + 1)
        
        return torch.tensor(chunk[:-1], device=self.device), torch.tensor(chunk[1:], device=self.device)
    
    def get_sequence_info(self):
        return {
            "seq_len": self.seq_len,
            "stride": self.stride,
            "num_sequences": len(self),
            "tokenizer_name": self.tokenizer_name
        }
    
    def save(self, filepath: str):
        """
        Save the dataset to disk for faster loading later.
        
        Args:
            filepath: Path where to save the dataset (without extension)
        """
        # Save tokens and metadata
        data = {
            'tokens': self.tokens,
            'seq_len': self.seq_len,
            'stride': self.stride,
            'tokenizer_name': self.tokenizer_name
        }
        
        # Save as pickle for fast loading
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(data, f)
        
        # Also save metadata as JSON for inspection
        metadata = {
            'seq_len': self.seq_len,
            'stride': self.stride,
            'num_sequences': len(self),
            'tokenizer_name': self.tokenizer_name,
            'num_tokens': len(self.tokens)
        }
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str, device: str = "cpu"):
        """
        Load a saved dataset from disk.
        
        Args:
            filepath: Path to the saved dataset (without extension)
            device: Device to load tensors on
            
        Returns:
            TextDataset instance
        """
        with open(f"{filepath}.pkl", 'rb') as f:
            data = pickle.load(f)
        
        dataset = cls.__new__(cls)
        dataset.tokens = data['tokens']
        dataset.seq_len = data['seq_len']
        dataset.stride = data['stride']
        dataset.device = device
        dataset.tokenizer_name = data['tokenizer_name']
        
        return dataset
    
    @classmethod
    def load_metadata(cls, filepath: str) -> Dict[str, Any]:
        """
        Load just the metadata of a saved dataset.
        
        Args:
            filepath: Path to the saved dataset (without extension)
            
        Returns:
            Dictionary containing dataset metadata
        """
        with open(f"{filepath}_metadata.json", 'r') as f:
            return json.load(f)


def create_dataset(text: str, tokenizer, seq_len: int, stride: int = 1, device: str = "cpu"):
    return TextDataset(text, tokenizer, seq_len, stride, device)

def create_dataloader(dataset: TextDataset, batch_size: int, shuffle: bool = True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def create_or_load_dataset(text: Optional[str], tokenizer, seq_len: int, stride: int = 1, 
                          device: str = "cpu", cache_path: Optional[str] = None):
    """
    Create a dataset, optionally loading from cache if available.
    
    Args:
        text: Text to tokenize (can be None if loading from cache)
        tokenizer: Tokenizer to use
        seq_len: Sequence length
        stride: Stride for sequences
        device: Device to use
        cache_path: Path to cache the dataset (without extension)
        
    Returns:
        TextDataset instance
    """
    if cache_path and os.path.exists(f"{cache_path}.pkl"):
        print(f"Loading dataset from cache: {cache_path}")
        return TextDataset.load(cache_path, device)
    
    if text is None:
        raise ValueError("text must be provided when not loading from cache")
    
    print(f"Creating new dataset and saving to cache: {cache_path}")
    dataset = TextDataset(text, tokenizer, seq_len, stride, device)
    # if folder does not exist, create it
    if cache_path and not os.path.exists(os.path.dirname(cache_path)):
        os.makedirs(os.path.dirname(cache_path))
    if cache_path:
        dataset.save(cache_path)
    
    return dataset

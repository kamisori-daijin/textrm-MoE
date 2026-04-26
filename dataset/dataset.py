import os
import torch
import numpy as np
from torch.utils.data import IterableDataset
from .prepare_binary_dataset import prepare_binary_data

class BinaryPackedDataset(IterableDataset):
    def __init__(self, bin_file, max_length=512):
        super().__init__()
        self.max_length = max_length
        
        if not os.path.exists(bin_file):
            raise FileNotFoundError(f"Binary file not found: {bin_file}. Run prepare_binary_data() first.")

        # Memory-map the binary file (uint16 = 2 bytes per token)
        # This points to the SSD file without loading it into RAM.
        self.data = np.memmap(bin_file, dtype=np.uint16, mode='r')
        self.num_tokens = len(self.data)

    def __len__(self):
        # Calculate total possible chunks
        return (self.num_tokens - 1) // self.max_length

    def __iter__(self):
        # High-speed sliding window over the memory-mapped file
        for i in range(0, self.num_tokens - self.max_length - 1, self.max_length):
            # Fetch chunk from SSD (OS handles pre-fetching automatically)
            chunk = self.data[i : i + self.max_length + 1]
            
            # Convert to PyTorch tensor (int64 is required for most loss functions)
            d = torch.from_numpy(chunk.astype(np.int64))
            
            # Yield (input, target)
            yield d[:-1], d[1:]

def get_binary_datasets(tokenizer,max_length=512,max_samples=300000, val_ratio=0.01,):
    """
    Load pre-processed binary datasets from the 'bin_dataset' folder.
    """
    
    val_size = int(max_samples * val_ratio)
    train_size = max_samples - val_size
    
    train_bin = prepare_binary_data(tokenizer, "train_data.bin", max_samples=train_size)
    val_bin = prepare_binary_data(tokenizer, "val_data.bin", max_samples=val_size)
    
    train_ds = BinaryPackedDataset(train_bin, max_length=max_length)
    val_ds = BinaryPackedDataset(val_bin, max_length=max_length)
    
    return train_ds, val_ds
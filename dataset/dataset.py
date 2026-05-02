import os
import numpy as np
import mlx.core as mx
from .prepare_binary_dataset import prepare_binary_data

class MLXBinaryDataLoader:
    def __init__(self, bin_file, batch_size, max_length=512, shuffle=False):
        if not os.path.exists(bin_file):
            raise FileNotFoundError(f"Binary file not found: {bin_file}.")

        
        self.data = np.memmap(bin_file, dtype=np.uint16, mode='r')
        self.max_length = max_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        
        self.num_samples = (len(self.data) - 1) // self.max_length

    def __iter__(self):
        
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)
            
        for i in range(0, self.num_samples, self.batch_size):
            batch_idx = indices[i : i + self.batch_size]
            
            input_batch = []
            target_batch = []
            
            for idx in batch_idx:
                start = idx * self.max_length
                end = start + self.max_length + 1
                
                
                chunk = self.data[start:end].astype(np.int32)
                
                input_batch.append(chunk[:-1])
                target_batch.append(chunk[1:])
                
            # Stack into a single numpy array first for stability
            input_batch = np.stack(input_batch)
            target_batch = np.stack(target_batch)
            
            yield mx.array(input_batch), mx.array(target_batch)

    def __len__(self):
        
        import math
        return math.ceil(self.num_samples / self.batch_size)


def get_binary_datasets(tokenizer, max_length=512, max_samples=300000, val_ratio=0.01, batch_size=4):
    """
    Load pre-processed binary datasets and return MLX native loaders.
    """
    val_size = int(max_samples * val_ratio)
    train_size = max_samples - val_size
    
    train_bin = prepare_binary_data(tokenizer, "train_data.bin", max_samples=train_size)
    val_bin = prepare_binary_data(tokenizer, "val_data.bin", max_samples=val_size)
    
    
    train_loader_factory = lambda: MLXBinaryDataLoader(train_bin, batch_size, max_length, shuffle=True)
    val_loader_factory = lambda: MLXBinaryDataLoader(val_bin, batch_size, max_length, shuffle=False)
    
    return train_loader_factory, val_loader_factory

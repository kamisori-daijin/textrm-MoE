import os
import mlx.core as mx
import numpy as np

from .prepare_binary_dataset import prepare_binary_data


class MLXBinaryDataLoader:
    def __init__(self, bin_file, batch_size, max_length=512, shuffle=False, max_sequences=None):
        if not os.path.exists(bin_file):
            raise FileNotFoundError(f"Binary file not found: {bin_file}.")

        # Disk streaming via virtual memory mapping to preserve Mac unified memory
        self.data = np.memmap(bin_file, dtype=np.uint16, mode="r")
        self.max_length = max_length
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Compute the absolute maximum number of sequences available in the file
        total_sequences = (len(self.data) - 1) // self.max_length

        # CRITICAL FIX: Limit the total sequence count strictly based on user-defined dynamic bounds
        if max_sequences is not None:
            self.num_samples = min(total_sequences, max_sequences)
        else:
            self.num_samples = total_sequences

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

            input_batch = np.stack(input_batch)
            target_batch = np.stack(target_batch)

            yield mx.array(input_batch), mx.array(target_batch)

    def __len__(self):
        import math
        return math.ceil(self.num_samples / self.batch_size)


def get_binary_datasets(
    tokenizer, max_length=512, max_documents=300000, max_train_sequences=50000, val_ratio=0.01, batch_size=4
):
    """
    Load pre-processed binary datasets and return bounded MLX native loaders.
    
    Args:
        max_documents: Total raw documents to download and stream into the binary files.
        max_train_sequences: The maximum number of 512-token chunks to actually train on per epoch.
    """
    val_doc_size = int(max_documents * val_ratio)
    train_doc_size = max_documents - val_doc_size

    # 1. Export a generous fallback asset pool from huggingface stream channels
    train_bin = prepare_binary_data(tokenizer, "train_data.bin", max_samples=train_doc_size)
    val_bin = prepare_binary_data(tokenizer, "val_data.bin", max_samples=val_doc_size)

    # 2. Extract a slice of sequences to prevent multi-day execution loops on local machines
    val_sequences = int(max_train_sequences * val_ratio)

    def train_loader_factory():
        return MLXBinaryDataLoader(train_bin, batch_size, max_length, shuffle=True, max_sequences=max_train_sequences)
        
    def val_loader_factory():
        return MLXBinaryDataLoader(val_bin, batch_size, max_length, shuffle=False, max_sequences=val_sequences)
    
    return train_loader_factory, val_loader_factory

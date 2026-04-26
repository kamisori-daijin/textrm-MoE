import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset, interleave_datasets

class StreamingPackedDataset(IterableDataset):
    def __init__(self, tokenizer, max_length=512, max_samples=300000, split='train', val_ratio=0.01):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = max_samples
        self.split = split
        self.val_ratio = val_ratio
        self.eos_id = tokenizer.eos_token_id
        
        
        ds_cosmo = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
        ds_fineweb = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", split="train", streaming=True)
        ds_python = load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", split="train", streaming=True)
        
        self.combined_ds = interleave_datasets(
            [ds_cosmo, ds_fineweb, ds_python], 
            probabilities=[0.4, 0.4, 0.2],
        )

    def __iter__(self):
       
        val_size = int(self.max_samples * self.val_ratio)
        train_size = self.max_samples - val_size
        
        buffer_ids = []
        yielded_count = 0
        skipped_count = 0

        for item in self.combined_ds:
            text = item.get("text") or item.get("content") or ""
            if len(text) < 100: continue

            ids = self.tokenizer.encode(text, add_special_tokens=True)
            if not ids or ids[-1] != self.eos_id:
                ids.append(self.eos_id)
            buffer_ids.extend(ids)

            while len(buffer_ids) >= self.max_length + 1:
                chunk = buffer_ids[:self.max_length + 1]
                
                
                if self.split == 'val':
                   
                    if yielded_count < val_size:
                        yield torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)
                        yielded_count += 1
                    else:
                        return 
                else:
                    
                    if skipped_count < val_size:
                        skipped_count += 1
                    elif yielded_count < train_size:
                        yield torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)
                        yielded_count += 1
                    else:
                        return 
                
                buffer_ids = buffer_ids[self.max_length:]

def get_streaming_datasets(tokenizer, max_length=512, max_samples=300000, val_ratio=0.01):
   
    train_ds = StreamingPackedDataset(tokenizer, max_length, max_samples, split='train', val_ratio=val_ratio)
    val_ds = StreamingPackedDataset(tokenizer, max_length, max_samples, split='val', val_ratio=val_ratio)
    return train_ds, val_ds
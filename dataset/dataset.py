import torch
from torch.utils.data import Dataset
from datasets import load_dataset, interleave_datasets
from tqdm import tqdm

class PackedDataset(Dataset):
    """
    A simple wrapper to hold the tokenized and packed data.
    """
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Returns input_ids and target labels
        ex = self.examples[idx]
        return ex["input_ids"], ex["labels"]

def get_packed_dataset(
    tokenizer,
    max_length=512, 
    max_samples=300000, 
    val_ratio=0.01
):
    """
    Loads, filters, and packs the SmolLM-Corpus subsets into a fixed-length dataset.
    """
    print("Loading SmolLM-Corpus subsets: Cosmopedia v2, FineWeb-Edu, and Python-Edu...")

    # Load high-quality educational subsets in streaming mode
    # 1. Cosmopedia v2: Synthetic textbooks and stories
    ds_cosmo = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
    
    # 2. FineWeb-Edu: High-score educational web pages
    ds_fineweb = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", split="train", streaming=True)
    
    # 3. Python-Edu: Cleaned educational Python code
    ds_python = load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", split="train", streaming=True)

    # Interleave datasets to balance knowledge, web reasoning, and logic
    # Using 40% Cosmo, 40% FineWeb, and 20% Python for a balanced brain
    combined_ds = interleave_datasets(
        [ds_cosmo, ds_fineweb, ds_python], 
        probabilities=[0.4, 0.4, 0.2], 
        
    )
    
    all_packed_examples = []
    buffer_ids = []
    eos_id = tokenizer.eos_token_id

    pbar = tqdm(total=max_samples, desc="Packing 300k Blocks")
    
    for item in combined_ds:
        # Extract text based on the specific schema of each subset
        text = item.get("text") or item.get("content") or ""
        if len(text) < 100:
            continue

        # Encode text with BOS (automatically added by LlamaTokenizer if configured)
        ids = tokenizer.encode(text, add_special_tokens=True)
        
        # Ensure the sequence ends with an EOS token
        if not ids or ids[-1] != eos_id:
            ids.append(eos_id)

        buffer_ids.extend(ids)

        # Packing Logic (Non-overlapping):
        # We need (max_length + 1) tokens to create an input/label pair of max_length
        while len(buffer_ids) >= max_length + 1:
            chunk = buffer_ids[:max_length + 1]
            
            # Prediction task: given tokens [0...N-1], predict [1...N]
            all_packed_examples.append({
                "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                "labels": torch.tensor(chunk[1:], dtype=torch.long)
            })
            
            # Efficient Packing: discard used tokens and move to the next block
            buffer_ids = buffer_ids[max_length:] 

            pbar.update(1)
            if len(all_packed_examples) >= max_samples:
                break
        
        if len(all_packed_examples) >= max_samples:
            break
            
    pbar.close()

    # Split into Train and Validation sets
    val_idx = int(len(all_packed_examples) * (1 - val_ratio))
    train_data = all_packed_examples[:val_idx]
    val_data = all_packed_examples[val_idx:]
    
    print(f"Data packing complete. Train: {len(train_data)}, Val: {len(val_data)}")
    return PackedDataset(train_data), PackedDataset(val_data)

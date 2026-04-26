import os
import numpy as np
from datasets import load_dataset, interleave_datasets
from tqdm import tqdm

def prepare_binary_data(tokenizer, output_filename, max_samples=300000):
    # --- Folder Setup ---
    # Create 'bin_dataset' directory if it doesn't exist
    output_dir = "bin_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine directory and filename
    output_path = os.path.join(output_dir, output_filename)
    
    # --- Load stream ---
    ds_cosmo = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
    ds_fineweb = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", split="train", streaming=True)
    ds_python = load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", split="train", streaming=True)

    combined_ds = interleave_datasets(
        [ds_cosmo, ds_fineweb, ds_python], 
        probabilities=[0.4, 0.4, 0.2],
    )

    print(f"Starting binary export to {output_path}...")
    
    # Use 'wb' to write binary data directly
    with open(output_path, "wb") as f:
        count = 0
        pbar = tqdm(total=max_samples)
        
        for item in combined_ds:
            if count >= max_samples:
                break
            
            text = item.get("text") or item.get("content") or ""
            if len(text) < 100: 
                continue
            
            # Encode to IDs
            ids = tokenizer.encode(text, add_special_tokens=True)
            if not ids or ids[-1] != tokenizer.eos_token_id:
                ids.append(tokenizer.eos_token_id)
            
            # Convert to uint16 (2 bytes per token)
            bin_ids = np.array(ids, dtype=np.uint16)
            f.write(bin_ids.tobytes())
            
            count += 1
            pbar.update(1)
            
    print(f"\nExport completed! Check your file at: {output_path}")
    return output_path

# --- How to use ---
# prepare_binary_data(tokenizer, "train_data.bin")
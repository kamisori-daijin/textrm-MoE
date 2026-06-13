import os
import random
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

def prepare_binary_data(tokenizer, output_filename: str, max_samples: int = 200000) -> str:
    # --- Folder Setup ---
    output_dir = "bin_dataset"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    # Fast-pass if tokenized binary dataset already cached
    if os.path.exists(output_path):
        print(f"Skipping: {output_path} already exists.")
        return output_path

    # --- Domain Mix Allocation (40% / 40% / 20%) ---
    count_cosmo = int(max_samples * 0.4)
    count_fineweb = int(max_samples * 0.4)
    count_python = int(max_samples * 0.2)

    raw_documents = []
    
    # --- High-Efficiency Sequential Streaming ---
    # Instead of interleaved multi-streaming (which introduces massive network latency),
    # we stream from each dataset sequentially. This maximizes TCP throughput and downloads
    # exactly the required number of valid text samples within seconds.
    
    print("🌐 Streaming Cosmopedia (40%)...")
    ds_cosmo = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
    for item in ds_cosmo:
        if len(raw_documents) >= count_cosmo: 
            break
        text = item.get("text") or item.get("content") or ""
        if len(text) >= 100:  # Prune noise/empty entries
            raw_documents.append(text)

    print("🌐 Streaming FineWeb-Edu (40%)...")
    ds_fineweb = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", split="train", streaming=True)
    start_idx = len(raw_documents)
    for item in ds_fineweb:
        if len(raw_documents) - start_idx >= count_fineweb: 
            break
        text = item.get("text") or item.get("content") or ""
        if len(text) >= 100: 
            raw_documents.append(text)

    print("🌐 Streaming Python-Edu (20%)...")
    ds_python = load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", split="train", streaming=True)
    start_idx = len(raw_documents)
    for item in ds_python:
        if len(raw_documents) - start_idx >= count_python: 
            break
        text = item.get("text") or item.get("content") or ""
        if len(text) >= 100: 
            raw_documents.append(text)

    # --- Global In-Memory Shuffle ---
    # Shuffle the aggregated text corpus locally to guarantee thorough domain blending 
    # and deterministic reproducibility before tokenization.
    print("🔀 Shuffling blended text assets...")
    random.seed(42)
    random.shuffle(raw_documents)

    # --- High-Throughput Tokenization & Binary Packing ---
    # Processing strictly local RAM strings unlocks maximum tokenizer speed.
    print(f"📦 Packing tokens into binary format: {output_path}...")
    write_buffer = []
    
    with open(output_path, "wb") as f:
        pbar = tqdm(total=len(raw_documents), desc="Packing Data", unit="docs")
        for text in raw_documents:
            ids = tokenizer.encode(text, add_special_tokens=True)
            if not ids or ids[-1] != tokenizer.eos_token_id:
                ids.append(tokenizer.eos_token_id)
                
            write_buffer.extend(ids)
            pbar.update(1)
            
            # Flush to disk when the buffer exceeds 100,000 token IDs to maximize sequential SSD write speed
            if len(write_buffer) > 100000:
                f.write(np.array(write_buffer, dtype=np.uint16).tobytes())
                write_buffer = []  # Free memory references
                
        # Flush any remaining trailing tokens
        if write_buffer:
            f.write(np.array(write_buffer, dtype=np.uint16).tobytes())
            
    print(f"\n🎉 Success! original-style stream packing completed at: {output_path}")
    return output_path



# --- How to use ---
# prepare_binary_data(tokenizer, "train_data.bin")
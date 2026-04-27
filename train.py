import mlx.core as mx
import numpy as np
from models.config import config
from dataset.dataset import get_binary_datasets
from training.instantiate import tokenizer
from training.trainer import train
from training.instantiate import model


# --- 💡 Fast DataLoader Alternative for MLX ---
def get_batches(dataset, batch_size, shuffle=False):
    """Simple generator to yield batches from dataset for MLX."""
    indices = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(indices)
        
    for i in range(0, len(dataset), batch_size):
        batch_idx = indices[i:i + batch_size]
        
       
        samples = [dataset[int(idx)] for idx in batch_idx]
        
       
        input_ids = mx.array([s[0] for s in samples])
        targets = mx.array([s[1] for s in samples])
        
        yield input_ids, targets

if __name__ == '__main__':
    # Load and pack dataset once, then split
    train_dataset, val_dataset = get_binary_datasets(
            tokenizer=tokenizer,
            max_length=config['max_seq_len'],
            max_samples=config['max_train_samples'] + config['max_val_samples'],
            val_ratio=config['max_val_samples'] / (config['max_train_samples'] + config['max_val_samples'])
    )
   
    print("Dataset loaded. Now training model...")    
    
    
    train_loader_factory = lambda: get_batches(train_dataset, config['batch_size'], shuffle=True)
    val_loader_factory = lambda: get_batches(val_dataset, config['batch_size'], shuffle=False)

    # Training
    save_path = 'best_model.safetensors'
   
    
    model = train(
        model=model,
        train_loader=train_loader_factory, 
        val_loader=val_loader_factory,
        tokenizer=tokenizer,
        epochs=config['epochs'],
        lr=config['lr'],
        warmup_steps=config['warmup_steps'],
        n_supervision_steps=config['n_supervision_steps'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        save_path=save_path,
    )
    
    print('\nTraining complete!')
    
    final_path = 'final_model.safetensors'
    model.save_weights(final_path)
    print(f'Saved final model to {final_path}')

    # Test Generation
    prompts = [
        "Explain why the sky looks blue during the day:", 
        "The following is a Python function for binary search:\ndef binary_search(arr, target):", 
        "Question: If a cube has 6 faces, how many faces do 3 cubes have in total? Answer:", 
        "A formal email to a professor requesting an extension on a deadline:", 
    ]

    print('\n=== Generated ===\n')
    for prompt in prompts:
        prompt_ids = mx.array([tokenizer.encode(prompt)])
        generated = model.generate(prompt_ids, max_new_tokens=150, temperature=0.8)
        
        
        text = tokenizer.decode(generated.tolist()[0])
        
        print(f'Prompt: "{prompt}"')
        print(f'Generated: {text}\n')
        print('-' * 50 + '\n')

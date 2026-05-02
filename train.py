import mlx.core as mx
from models.config import config
from dataset.dataset import get_binary_datasets
from training.instantiate import tokenizer
from training.trainer import train
from training.instantiate import model

if __name__ == '__main__':
    
    train_loader_factory, val_loader_factory = get_binary_datasets(
            tokenizer=tokenizer,
            max_length=config['max_seq_len'],
            max_samples=config['max_train_samples'] + config['max_val_samples'],
            val_ratio=config['max_val_samples'] / (config['max_train_samples'] + config['max_val_samples']),
            batch_size=config['batch_size']  
    )
   
    print("Dataset loaded. Now training model...")    

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
    
    # Standard weight saving in MLX
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
        
    model.eval() 
        
    for prompt in prompts:
            
        prompt_ids = mx.array([tokenizer.encode(prompt)], dtype=mx.int32)
            
            
        generated = model.generate(prompt_ids, max_new_tokens=150, temperature=0.8)
            
            
        full_text = tokenizer.decode(generated[0].tolist())
            
        print(f'Prompt: "{prompt}"')
        print(f'Generated: {full_text}\n')
        print('-' * 50 + '\n')
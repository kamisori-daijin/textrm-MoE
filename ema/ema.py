import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

class EMA:
    """Exponential Moving Average for model weights in MLX"""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        
       
        self.shadow = tree_map(lambda x: mx.array(x), model.parameters())
        self.backup = None

    def update(self):
        """
        Calculates the moving average:
        shadow = decay * shadow + (1 - decay) * current_param
        """
        current_params = self.model.parameters()
        
        def _ema_update(s, p):
            return self.decay * s + (1 - self.decay) * p
            
        self.shadow = tree_map(_ema_update, self.shadow, current_params)

    def apply_shadow(self):
        """Swaps the model's parameters with the EMA parameters."""
        
        
        self.backup = tree_map(lambda x: mx.array(x), self.model.parameters())
        
        self.model.update(self.shadow)

    def restore(self):
        """Restores the original parameters from the backup."""
        if self.backup is not None:
            self.model.update(self.backup)
            self.backup = None

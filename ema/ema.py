import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

class EMA:
    """Exponential Moving Average"""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        
       
        self.shadow = tree_map(lambda x: mx.array(x), model.parameters())
        self.backup = None

    def update(self):
        """
        Update the shadows with the current model parameters.
        mx.lerp(p, s, decay) -> p * (1 - decay) + s * decay
        """
        current_params = self.model.parameters()
        
      
        def _ema_update(s, p):
           
            return self.decay * s + (1.0 - self.decay) * p
            
        self.shadow = tree_map(_ema_update, self.shadow, current_params)

    def apply_shadow(self):
        """Replace the model weights with EMA (shadow)."""
       
        self.backup = tree_map(lambda x: mx.array(x), self.model.parameters())
       
        self.model.update(self.shadow)

    def restore(self):
        """Restore the original weights from the backup."""
        if self.backup is not None:
            self.model.update(self.backup)
            self.backup = None
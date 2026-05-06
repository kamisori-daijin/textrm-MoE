import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map


class EMA:
    """Exponential Moving Average"""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay

        # Keep track of current model state (reference, not copy)
        self.shadow = tree_map(lambda x: mx.array(x), model.parameters())
        self.backup = None
        self._using_shadow = False

    def update(self):
        """
        Update the shadows with the current model parameters.
        EMA update: shadow = decay * shadow + (1 - decay) * current
        """
        current_params = self.model.parameters()

        def _ema_update(s, p):
            return self.decay * s + (1.0 - self.decay) * p

        self.shadow = tree_map(_ema_update, self.shadow, current_params)

    def apply_shadow(self):
        """Replace the model weights with EMA (shadow) weights.

        Instead of deep copying, we just swap references.
        This is O(model size) in memory but avoids the copy overhead.
        """
        if not self._using_shadow:
            # Store reference to current weights (not a copy)
            self.backup = self.model.parameters()
            # Swap model weights to shadow weights
            self.model.update(self.shadow)
            self._using_shadow = True

    def restore(self):
        """Restore the original weights from the backup.
        Simply restores the reference back.
        """
        if self._using_shadow and self.backup is not None:
            self.model.update(self.backup)
            self.backup = None
            self._using_shadow = False

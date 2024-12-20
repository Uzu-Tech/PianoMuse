import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, context_size):
        """Positional Encoding.

        Args:
            d_model: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.

        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        position_encoding = torch.zeros(context_size, embed_dim)
        # Unsqueeze to turn positions into columns
        position = torch.arange(0, context_size, dtype=torch.float).unsqueeze(1)
        # e^-log(10_000) is the same as 10_000^-1
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10_000.0) / embed_dim)
        )
        # Get the even and odd i in each row pos
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        # Add a batch dimension so it can be broadcasted
        position_encoding = position_encoding.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("position_encoding", position_encoding, persistent=False)

    def forward(self, x):
        x = x + self.position_encoding[:, :x.size(-2)]
        return x
    
class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
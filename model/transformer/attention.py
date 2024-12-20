import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class MultiHeadAttention(nn.Module):
    def __init__(self, context_size, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.context_size = context_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_concat = torch.nn.Linear(embed_dim, embed_dim * 3)
        self.output_proj = torch.nn.Linear(embed_dim, embed_dim)
        # Create learnable relative embeddings for each head
        self.relative_embeddings = nn.Parameter(
            torch.empty(num_heads, context_size, self.head_dim)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization
        nn.init.xavier_uniform_(self.qkv_concat.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.xavier_uniform_(self.relative_embeddings)
        # No biases were used
        self.qkv_concat.bias.data.fill_(0)
        self.output_proj.bias.data.fill_(0)

    @staticmethod
    def attention(
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Optional[Tensor] = None,
        relative_embeddings: Optional[torch.nn.Parameter] = None,
    ) -> Tuple[Tensor, Tensor]:

        dimension_size = queries.size()[-1]
        # Dot Product for every query and key embedding (Each embedding is a row)
        attention_logits = torch.einsum(
            "...ik, ...jk -> ...ij", queries, keys
        ) / math.sqrt(dimension_size)

        if relative_embeddings is not None:
            seq_len = queries.size(-2)
            other_dimensions = queries.size()[:-2]
            # Calculate dot product between each Qi and relative embedding
            relative_logits = torch.einsum(
                "...ik, ...jk -> ...ij", queries, relative_embeddings
            )
            # Concatenate a zero column to pad the original tensor
            relative_logits = torch.cat(
                (
                    torch.zeros(*other_dimensions, 1, relative_logits.size(-1)),
                    relative_logits,
                ),
                dim=-2,
            )
            # Reshape QR^T to align relative positions to position j - i in tensor,
            # and remove top row of zeros
            relative_logits = torch.reshape(
                relative_logits, (*other_dimensions, seq_len + 1, seq_len)
            )[..., 1:, :]
            attention_logits += relative_logits

        # Replace logits in the mask with negative infinity so they become 0 after softmax
        if mask is not None:
            attention_logits = attention_logits.masked_fill(mask == 0, float("-inf"))

        # Softmax over all the keys for each query, so softmax over the rows of the logits
        attention_weights = torch.nn.functional.softmax(attention_logits, dim=-1)
        attention_values = torch.matmul(attention_weights, values)
        return attention_weights, attention_values

    def forward(self, x, mask=None):
        other_dimensions = x.size()[:-1]
        seq_length = x.size(-2)

        # Dynamically adjust the relative embeddings based on the sequence length
        relative_embeddings = self.relative_embeddings[..., -seq_length:, :]
        qkv_concat = self.qkv_concat(x)  # Get the Q, K, V in one go

        # Separate Q, K, V from linear output
        qkv_concat = qkv_concat.reshape(
            *other_dimensions, self.num_heads, 3 * self.head_dim
        )
        # [Batch, Head, SeqLen, Dims] Flip head and seq length dimension for attention
        qkv_concat = qkv_concat.permute(0, 2, 1, 3)
        q, k, v = qkv_concat.chunk(3, dim=-1)

        # Determine value outputs
        attention_weights, attention_values = self.attention(
            q, k, v, relative_embeddings=relative_embeddings, mask=mask
        )
        # [Batch, SeqLen, Head, Dims]
        attention_values = attention_values.permute(0, 2, 1, 3)
        # Concatenate head output
        attention_values = attention_values.reshape(*other_dimensions, self.embed_dim)
        attention_output = self.output_proj(attention_values)

        return attention_output, attention_weights

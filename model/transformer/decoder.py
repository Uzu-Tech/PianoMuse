import matplotlib_inline.backend_inline
import torch.nn as nn
from model.transformer.attention import MultiHeadAttention

matplotlib_inline.backend_inline.set_matplotlib_formats("svg", "pdf")

class MultiLayeredDecoder(nn.Module):
    def __init__(self, num_layers, **decoder_args):
        super().__init__()
        self.decoder_layers = nn.ModuleList(
            [DecoderBlock(**decoder_args) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        for layer in self.decoder_layers:
            x = layer(x, mask)
        return x

class DecoderBlock(nn.Module):
    def __init__(
        self,
        context_size,
        embed_dim,
        num_attention_heads,
        feed_forward_dim,
        dropout=0.0,
    ):
        super().__init__()

        self.context_size = context_size
        self.embed_dim = embed_dim
        self.num_attention_heads = num_attention_heads

        # Layer Norms
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.norm_2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Feed forward layer, multi-layer perception
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, embed_dim),
        )

        self.self_attention = MultiHeadAttention(
            context_size, embed_dim, num_attention_heads
        )

    def forward(self, x, mask=None):
        # Follow structure of X -> Self Attention -> Norm -> Feed Forward -> Norm
        attention_values, _ = self.self_attention.forward(x, mask)
        x += self.dropout(attention_values)
        x = self.norm_1(x)

        feed_forward_output = self.feed_forward(x)
        x += self.dropout(feed_forward_output)
        x = self.norm_2(x)
        return x

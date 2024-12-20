import lightning as pL
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model.transformer.decoder import MultiLayeredDecoder
from model.transformer.util import CosineWarmupScheduler, PositionalEncoding


class RelativeTransformerPredictor(pL.LightningModule):
    def __init__(
        self,
        vocab_size,
        context_size,
        embed_dim,
        num_heads,
        num_layers,
        learning_rate,
        warmup,
        max_iters,
        padding_idx,
        dropout=0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._create_model()

    def _create_model(self):
        self._embeddings = nn.Embedding(
            self.hparams.vocab_size,
            self.hparams.embed_dim,
            padding_idx=self.hparams.padding_idx,
        )

        self._positional_encoding = PositionalEncoding(
            self.hparams.embed_dim, self.hparams.context_size
        )

        self._multi_layered_decoder = MultiLayeredDecoder(
            self.hparams.num_layers,
            self.hparams.context_size,
            self.hparams.embed_dim,
            self.hparams.num_heads,
            feed_forward_dim=self.hparams.embed_dim * 4,
            dropout=self.hparams.dropout,
        )

        # Final feed forward network
        self._output_net = nn.Sequential(
            nn.Linear(self.hparams.embed_dim, self.hparams.embed_dim),
            nn.LayerNorm(self.hparams.embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.embed_dim, self.hparams.vocab_size),
        )

        self._loss = nn.CrossEntropyLoss(ignore_index=self.hparams.padding_idx)

    def _get_mask(self, x):
        seq_len = x.size(1)
        # Create a lower triangular mask, with two extra dimensions for batch size and num heads
        return (
            torch.tril(
                torch.ones((seq_len, seq_len), dtype=torch.float, device=x.device)
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

    def forward(self, x):
        x = self._embeddings(x)
        x += self._positional_encoding(x)
        x = self._multi_layered_decoder(x, self._get_mask(x))
        x = self._output_net(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # We don't return the lr scheduler since we need to apply it per iteration, not per epoch
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters
        )
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration

    def _get_loss(self, batch, log_mode):
        input, targets = batch
        outputs = self.forward(input)
        # Combine batch and seq len dimensions when calculating loss
        loss = self._loss(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        #
        accuracy = (outputs.argmax(dim=-1) == targets).float().mean()
        self.log(f"{log_mode}_loss", loss)
        self.log(f"{log_mode}_accuracy", accuracy)
        return loss, accuracy
    
    def training_step(self, batch, batch_idx):
        loss, _ = self._get_loss(batch, log_mode="train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        _ = self._get_loss(batch, log_mode="val")

    def test_step(self, batch, batch_idx):
        _ = self._get_loss(batch, log_mode="test")
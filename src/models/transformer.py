import math
from typing import Any
import torch
from torch import nn


from lightning import LightningModule


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super(Embeddings, self).__init__()
        self.d_model = d_model
        self.lut = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.d_model)

    def forward(self, x):
        x = self.lut(x) * (self.d_model) ** 0.5
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len) -> None:
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        self.positions = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(self.positions * div_term)
        pe[:, 1::2] = torch.cos(self.positions * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return x


class PatchEncoding(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = nn.Flatten(1, 2)

    def forward(self, images):
        B, C, H, W = images.shape

        assert (
            H % self.patch_size == 0
        ), f"Image size ({H}X{W}) should be divisible by Patch size: {self.patch_size}"

        num_patches = H // self.patch_size

        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        patches = patches.contiguous().view(B, num_patches, num_patches, -1)
        vectors = self.flatten(patches)
        return vectors


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1) -> None:
        super().__init__()
        self.fcn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        x = self.fcn(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True
        )

        self.ln1 = nn.LayerNorm(normalized_shape=d_model)
        self.ln2 = nn.LayerNorm(normalized_shape=d_model)

        self.feedforward = PositionwiseFeedForward(
            d_model=d_model, d_ff=d_ff, dropout=dropout
        )

    def forward(self, x):
        x1 = self.ln1(x)
        x = x + self.mha(query=x1, key=x1, value=x1)[0]

        x1 = self.ln2(x)
        x = x + self.feedforward(x1)
        return x


class Transformer(LightningModule):
    def __init__(
        self,
        arch,
        d_model,
        max_len,
        num_heads,
        num_layers,
        num_classes,
        d_ff,
        dropout,
        lr,
        **kwargs,
    ):
        super().__init__()
        self.arch = arch
        self.lr = lr
        self.weight_decay = kwargs.get("weight_decay")
        self.betas = kwargs.get("betas")

        if self.arch == "vit":
            patch_size = kwargs.get("patch_size")
            input_channels = kwargs.get("input_channels")

            self.class_embedding = nn.Parameter(
                data=torch.randn(1, 1, d_model), requires_grad=True
            )
            self.patch_embedding = PatchEncoding(patch_size=patch_size)

            self.linear_mapper = nn.Linear(
                input_channels * patch_size * patch_size, d_model
            )
        else:
            self.embedding = Embeddings(d_model=d_model, vocab_size=num_classes)

        self.pe = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.embedding_dropout = nn.Dropout(dropout)

        self.transformer_layers = nn.Sequential(
            *[
                TransformerLayer(
                    d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.lm_head = nn.Linear(d_model, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        if self.arch == "vit":
            batch_size = x.shape[0]
            class_tokens = self.class_embedding.expand(batch_size, -1, -1)
            x = self.patch_embedding(x)
            x = self.linear_mapper(x)
            x = torch.cat((class_tokens, x), dim=1)

        else:
            x = self.embedding(x)

        x = self.pe(x)
        x = self.embedding_dropout(x)
        x = self.transformer_layers(x)
        x = self.lm_head(x[:, 0] if self.arch == "vit" else x)
        x = self.softmax(x)
        return x

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )
        return optimizer

    def loss_fn(self, pred, target):
        return nn.CrossEntropyLoss()(pred, target)

    def training_step(self, batch, batch_idx):
        xb, yb = batch
        y_pred = self.forward(xb)

        if self.arch != "vit":
            output_v = y_pred.view(-1, y_pred.shape[-1])
            target_v = yb.view(-1, 1).squeeze()
        else:
            output_v = y_pred
            target_v = yb

        loss = self.loss_fn(output_v, target_v)

        self.log(
            name="train_loss",
            value=loss.item(),
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        xb, yb = batch
        y_pred = self.forward(xb)

        if self.arch != "vit":
            output_v = y_pred.view(-1, y_pred.shape[-1])
            target_v = yb.view(-1, 1).squeeze()
        else:
            output_v = y_pred
            target_v = yb

        loss = self.loss_fn(output_v, target_v)

        self.log(
            name="val_loss",
            value=loss.item(),
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )

        return loss

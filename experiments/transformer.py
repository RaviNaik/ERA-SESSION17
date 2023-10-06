from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import math
import lightning as L


# Positional Embedding
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        pe.requires_grad = False
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]  # x.size(1) = seq_len


class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """

    # 2. Initialize the class with appropriate variables
    def __init__(
        self, in_channels: int = 3, patch_size: int = 16, embedding_dim: int = 768
    ):
        super().__init__()
        self.patch_size = patch_size

        # 3. Create a layer to turn an image into patches
        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

        # 4. Create a layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(
            start_dim=2,  # only flatten the feature map dimensions into a single vector
            end_dim=3,
        )

    # 5. Define the forward method
    def forward(self, x):
        # Create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert (
            image_resolution % self.patch_size == 0
        ), f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"

        # Perform the forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        # 6. Make sure the output shape has the right order
        return x_flattened.permute(
            0, 2, 1
        )  # adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]


class Transformer(L.LightningModule):
    def __init__(
        self,
        arch,
        n_layers,
        n_heads,
        embedding_dim,
        dim_feedforward,
        n_embeddings,
        seq_len,
        ignore_index,  # dataset.IGNORE_IDX
        rvocab,  # dataset.rvocab
        dropout=0.1,
        lr=1e-4,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        **kwargs,
    ) -> None:
        super().__init__()
        self.arch = arch
        self.weight_decay = weight_decay
        self.lr = lr
        self.betas = betas

        n_classes = n_embeddings

        if self.arch == "vit":
            img_size = kwargs.get("img_size", 224)
            patch_size = kwargs.get("patch_size", 16)
            embedding_dropout = kwargs.get("embedding_dropout", 0.1)
            in_channels = kwargs.get("in_channels", 0.1)

            assert (
                img_size % patch_size == 0
            ), f"Image size must be divisible by patch size, image size: {img_size}, patch size: {patch_size}."

            self.num_patches = (img_size * img_size) // patch_size**2

            # 5. Create learnable class embedding (needs to go at front of sequence of patch embeddings)
            self.class_embedding = nn.Parameter(
                data=torch.randn(1, 1, embedding_dim), requires_grad=True
            )

            # 6. Create learnable position embedding
            self.position_embedding = nn.Parameter(
                data=torch.randn(1, self.num_patches + 1, embedding_dim),
                requires_grad=True,
            )

            # 7. Create embedding dropout value
            self.embedding_dropout = nn.Dropout(p=embedding_dropout)

            # 8. Create patch embedding layer
            self.patch_embedding = PatchEmbedding(
                in_channels=in_channels,
                patch_size=patch_size,
                embedding_dim=embedding_dim,
            )

        else:
            self.embeddings = nn.Embedding(n_embeddings, embedding_dim)
            self.pe = PositionalEmbedding(embedding_dim, seq_len)

        if self.arch == "gpt":
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=embedding_dim, nhead=n_heads, dropout=dropout
            )
            self.transformer_decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer, num_layers=n_layers
            )

        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=n_layers
            )

        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=n_classes),
        )

    def forward(self, x):
        if self.arch == "gpt":
            x = self.embeddings(x)
            x = x + self.pe(x)
            x = self.transformer_decoder(x)
            x = self.classifier(x)
        else:
            if self.arch == "bert":
                x = self.embeddings(x)
                x = x + self.pe(x)
                x = self.transformer_encoder(x)
                x = self.classifier(x)

            else:
                batch_size = x.shape[0]

                # 13. Create class token embedding and expand it to match the batch size (equation 1)
                class_token = self.class_embedding.expand(
                    batch_size, -1, -1
                )  # "-1" means to infer the dimension (try this line on its own)

                # 14. Create patch embedding (equation 1)
                x = self.patch_embedding(x)

                # 15. Concat class embedding and patch embedding (equation 1)
                x = torch.cat((class_token, x), dim=1)

                # 16. Add position embedding to patch embedding (equation 1)
                x = self.position_embedding + x

                # 17. Run embedding dropout (Appendix B.1)
                x = self.embedding_dropout(x)

                # 18. Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3)
                x = self.transformer_encoder(x)

                # 19. Put 0 index logit through classifier (equation 4)
                x = self.classifier(x[:, 0])  # run on each sample in a batch at 0 index

        return x

    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )
        return optimizer

    def loss_fn(self, logits, targets):
        B, T, C = logits.shape
        logits = torch.reshape(logits, (B * T, C))
        targets = torch.reshape(targets, (B * T,))
        loss = F.cross_entropy(logits, targets)
        return loss

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        xb, yb = batch
        output = self.forward(xb)
        loss = self.loss_fn(output, yb)

        self.log(
            name="train_loss",
            value=np.round(loss.item(), 2),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return loss

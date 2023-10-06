import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import math
import lightning as L


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, out_dim, dropout=0.1):
        super().__init__()

        #        self.q_linear = nn.Linear(out_dim, out_dim)
        #        self.k_linear = nn.Linear(out_dim, out_dim)
        #        self.v_linear = nn.Linear(out_dim, out_dim)
        self.linear = nn.Linear(out_dim, out_dim * 3)

        self.n_heads = n_heads
        self.out_dim = out_dim
        self.out_dim_per_head = out_dim // n_heads
        self.out = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, t):
        return t.reshape(t.shape[0], -1, self.n_heads, self.out_dim_per_head)

    def attention(self, q, k, v, mask=None, dropout=None):
        scores = q.matmul(k.transpose(-2, -1))
        scores /= math.sqrt(q.shape[-1])

        # mask
        scores = scores if mask is None else scores.masked_fill(mask == 0, -1e3)

        scores = F.softmax(scores, dim=-1)
        scores = dropout(scores) if dropout is not None else scores
        output = scores.matmul(v)
        return output

    def forward(self, x, y=None, mask=None):
        # in decoder, y comes from encoder. In encoder, y=x
        y = x if y is None else y

        qkv = self.linear(x)  # BS * SEQ_LEN * (3*EMBED_SIZE_L)
        q = qkv[:, :, : self.out_dim]  # BS * SEQ_LEN * EMBED_SIZE_L
        k = qkv[:, :, self.out_dim : self.out_dim * 2]  # BS * SEQ_LEN * EMBED_SIZE_L
        v = qkv[:, :, self.out_dim * 2 :]  # BS * SEQ_LEN * EMBED_SIZE_L

        # break into n_heads
        q, k, v = [
            self.split_heads(t) for t in (q, k, v)
        ]  # BS * SEQ_LEN * HEAD * EMBED_SIZE_P_HEAD
        q, k, v = [
            t.transpose(1, 2) for t in (q, k, v)
        ]  # BS * HEAD * SEQ_LEN * EMBED_SIZE_P_HEAD

        # n_heads => attention => merge the heads => mix information
        scores = self.attention(
            q, k, v, mask, self.dropout
        )  # BS * HEAD * SEQ_LEN * EMBED_SIZE_P_HEAD
        scores = (
            scores.transpose(1, 2).contiguous().view(scores.shape[0], -1, self.out_dim)
        )  # BS * SEQ_LEN * EMBED_SIZE_L
        out = self.out(scores)  # BS * SEQ_LEN * EMBED_SIZE

        return out


class FeedForward(nn.Module):
    def __init__(self, inp_dim, inner_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(inp_dim, inner_dim)
        self.linear2 = nn.Linear(inner_dim, inp_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # inp => inner => relu => dropout => inner => inp
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, n_heads, inner_transformer_size, inner_ff_size, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(n_heads, inner_transformer_size, dropout)
        self.ff = FeedForward(inner_transformer_size, inner_ff_size, dropout)
        self.norm1 = nn.LayerNorm(inner_transformer_size)
        self.norm2 = nn.LayerNorm(inner_transformer_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.mha(x2, mask=mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.ff(x2))
        return x


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


class Bert(L.LightningModule):
    def __init__(
        self,
        n_code,
        n_heads,
        embed_size,
        inner_ff_size,
        n_embeddings,
        seq_len,
        ignore_index,  # dataset.IGNORE_IDX
        rvocab,  # dataset.rvocab
        dropout=0.1,
        lr=1e-4,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        print_each=10,
    ):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.ignore_index = ignore_index
        self.rvocab = rvocab
        self.print_each = print_each
        # model input
        self.embeddings = nn.Embedding(n_embeddings, embed_size)
        self.pe = PositionalEmbedding(embed_size, seq_len)

        # backbone
        encoders = []
        for i in range(n_code):
            encoders += [EncoderLayer(n_heads, embed_size, inner_ff_size, dropout)]
        self.encoders = nn.ModuleList(encoders)

        # language model
        self.norm = nn.LayerNorm(embed_size)
        self.linear = nn.Linear(embed_size, n_embeddings, bias=False)
        self.it = 1

    def forward(self, x):
        x = self.embeddings(x)
        x = x + self.pe(x)
        for encoder in self.encoders:
            x = encoder(x)
        x = self.norm(x)
        x = self.linear(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        masked_input = batch["input"]
        masked_target = batch["target"]

        output = self(masked_input)
        # compute the cross entropy loss
        output_v = output.view(-1, output.shape[-1])
        target_v = masked_target.view(-1, 1).squeeze()

        loss_model = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        loss = loss_model(output_v, target_v)

        if self.it % self.print_each == 0:
            print(
                "it:",
                self.it,
                " | loss",
                np.round(loss.item(), 2),
                " | Δw:",
                round(self.embeddings.weight.grad.abs().sum().item(), 3),
            )

        self.log(
            name="loss",
            value=np.round(loss.item(), 2),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.log(
            name="Δw",
            value=round(self.embeddings.weight.grad.abs().sum().item(), 3),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.it += 1
        return loss

    def on_train_end(self) -> None:
        print("saving embeddings...")
        N = 3000
        np.savetxt(
            "values.tsv",
            np.round(self.embeddings.weight.detach().cpu().numpy()[0:N], 2),
            delimiter="\t",
            fmt="%1.2f",
        )
        s = [self.rvocab[i] for i in range(N)]
        open("names.tsv", "w+").write("\n".join(s))

from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer  # pip install transformers
import lightning as L


class TextDataset(Dataset):
    def __init__(self, raw_data, seq_len):
        self.data = raw_data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, index):
        # ix = torch.randint(len(self.data) - self.seq_len, (1,))
        x = self.data[index : index + self.seq_len]
        y = self.data[index + 1 : index + self.seq_len + 1]
        return x, y


class GPTDataModule(L.LightningDataModule):
    def __init__(self, path_do_data="data/english.txt", batch_size=32, seq_len=64):
        self.batch_size = batch_size
        self.seq_len = seq_len
        data_raw = open(path_do_data, encoding="utf-8").read()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        # tokenize the input text
        tokens = self.tokenizer.tokenize(data_raw)
        # convert the tokens to their corresponding ids
        token_indices = self.tokenizer.convert_tokens_to_ids(tokens)
        self.data = torch.tensor(token_indices, dtype=torch.long)

    def setup(self, stage=None):
        n = int(0.9 * len(self.data))
        self.train_dataset = TextDataset(self.data[:n], self.seq_len)
        self.val_dataset = TextDataset(self.data[n:], self.seq_len)

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

    def decode_tokens(self, tokens):
        enc_sec = tokens.tolist()
        # decode the indices to a string
        text = self.tokenizer.decode(enc_sec)
        return text

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )

import os
import re
import torch
import random
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import lightning as L


class SentencesDataset(Dataset):
    # Init dataset
    def __init__(self, sentences, vocab, seq_len):
        dataset = self

        dataset.sentences = sentences
        dataset.vocab = vocab + ["<ignore>", "<oov>", "<mask>"]
        dataset.vocab = {e: i for i, e in enumerate(dataset.vocab)}
        dataset.rvocab = {v: k for k, v in dataset.vocab.items()}
        dataset.seq_len = seq_len

        # special tags
        dataset.IGNORE_IDX = dataset.vocab[
            "<ignore>"
        ]  # replacement tag for tokens to ignore
        dataset.OUT_OF_VOCAB_IDX = dataset.vocab[
            "<oov>"
        ]  # replacement tag for unknown words
        dataset.MASK_IDX = dataset.vocab[
            "<mask>"
        ]  # replacement tag for the masked word prediction task

    # fetch data
    def __getitem__(self, index, p_random_mask=0.15):
        dataset = self

        # while we don't have enough word to fill the sentence for a batch
        s = []
        while len(s) < dataset.seq_len:
            s.extend(dataset.get_sentence_idx(index % len(dataset)))
            index += 1

        # ensure that the sequence is of length seq_len
        s = s[: dataset.seq_len]
        [
            s.append(dataset.IGNORE_IDX) for i in range(dataset.seq_len - len(s))
        ]  # PAD ok

        # apply random mask
        s = [
            (dataset.MASK_IDX, w)
            if random.random() < p_random_mask
            else (w, dataset.IGNORE_IDX)
            for w in s
        ]

        return (
            torch.Tensor([w[0] for w in s]).long(),
            torch.Tensor([w[1] for w in s]).long(),
        )

    # return length
    def __len__(self):
        return len(self.sentences)

    # get words id
    def get_sentence_idx(self, index):
        dataset = self
        s = dataset.sentences[index]
        s = [
            dataset.vocab[w] if w in dataset.vocab else dataset.OUT_OF_VOCAB_IDX
            for w in s
        ]
        return s


class BertDataModule(L.LightningDataModule):
    def __init__(
        self,
        seq_len=20,
        n_vocab=40000,
        trainpth="training.txt",
        vocabpth="vocab.txt",
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        batch_size=1024,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.batch_size = batch_size

        sentences = open(trainpth).read().lower().split("\n")
        special_chars = ",?;.:/*!+-()[]{}\"'&"
        sentences = [
            re.sub(f"[{re.escape(special_chars)}]", " \g<0> ", s).split(" ")
            for s in sentences
        ]
        self.sentences = [[w for w in s if len(w)] for s in sentences]

        if not os.path.exists(vocabpth):
            words = [w for s in sentences for w in s]
            vocab = Counter(words).most_common(
                n_vocab
            )  # keep the N most frequent words
            vocab = [w[0] for w in vocab]
            open(vocabpth, "w+").write("\n".join(vocab))
        else:
            vocab = open(vocabpth).read().split("\n")

        self.vocab = vocab

    def setup(self, stage=None):
        self.train_dataset = SentencesDataset(self.sentences, self.vocab, self.seq_len)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

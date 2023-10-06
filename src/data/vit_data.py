import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torchvision import datasets
from torch.utils.data import DataLoader


class ViTDataModule(L.LightningDataModule):
    def __init__(self, train_dir, test_dir, transform, batch_size, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_data = datasets.ImageFolder(train_dir, transform=transform)
        self.test_data = datasets.ImageFolder(test_dir, transform=transform)
        self.class_names = self.train_data.classes

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

from datasets import load_dataset
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch.utils.data import random_split, DataLoader


class RLHFDataModule(LightningDataModule):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.save_hyperparameters(config)
        print(config)
        self._load_dataset()

    def _load_dataset(self):
        raw_dataset = load_dataset(self.hparams.path, split="train")
        train_size = int(self.hparams.split.train * len(raw_dataset))
        validation_size = int(self.hparams.split.validation * len(raw_dataset))
        test_size = len(raw_dataset) - train_size - validation_size
        train_dataset, val_dataset, test_dataset = random_split(
            raw_dataset,
            [train_size, validation_size, test_size]
        )
        self.train_dataset = self._generate_prompt(train_dataset)
        self.val_dataset = self._generate_prompt(val_dataset)
        self.test_dataset = self._generate_prompt(test_dataset)

    def _generate_prompt(self, dataset):
        ret = []
        for example in dataset:
            if example["input"]:
                ret.append(
                    "Below is an instruction that describes a task, paired with an input that provides further context. "
                    "Write a response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
                )
            else:
                ret.append(
                    "Below is an instruction that describes a task. "
                    "Write a response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{example['instruction']}\n\n### Response:"
                )
        return ret

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size * 2,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=False,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size * 2,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=False,
            drop_last=False,
        )

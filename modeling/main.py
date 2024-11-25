from pathlib import Path

import yaml
import typer
from addict import Addict
from torch import manual_seed
from lightning import Trainer

from model import LLMModel
from data import RLHFDataModule


manual_seed(seed=64)


def main(config_path: str):

    with Path(config_path).open("r") as config_fp:
        config: dict = Addict(
            yaml.load(
                stream=config_fp.read(),
                Loader=yaml.FullLoader,
            )
        )
    train_module = LLMModel(config=config["train_module"])
    data_module = RLHFDataModule(config=config["data_module"])

    trainer = Trainer(
        fast_dev_run=True,
        **config["trainer"]["config"]
    )
    trainer.fit(
        model=train_module,
        datamodule=data_module,
    )
    trainer.test(
        model=train_module,
        datamodule=data_module,
    )
    

if __name__ == "__main__":
    typer.run(main)

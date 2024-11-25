from torch import nn
from torch import Tensor
from lightning import LightningModule
from transformers import AutoModel, AutoTokenizer

from utils import import_module


class LLMModel(LightningModule):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.save_hyperparameters(config)

        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()
        self.loss_fn = import_module(self.hparams.loss_function.module) \
            (ignore_index=self.tokenizer.pad_token_id)

    def _load_model(self) -> AutoModel:
        model = import_module(self.hparams.model.module) \
            .from_pretrained(self.hparams.weight)
        return model

    def _load_tokenizer(self) -> AutoTokenizer:
        tokenizer = import_module(self.hparams.tokenizer.module) \
            .from_pretrained(self.hparams.weight, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def forward(self, tokens: dict) -> dict:
        return self.model(**tokens)

    def configure_optimizers(self):
        optim_module = import_module(self.hparams.optimizer.module)
        optimizer = optim_module(
            self.model.parameters(), **self.hparams.optimizer.params
        )
        return {"optimizer": optimizer}

    def training_step(self, batch: tuple[str, str], *args, **kwargs):
        x, y = batch
        x_tokens: dict[str, Tensor] = self.tokenizer(
            texts=x,
            return_type="pt",
            padding=True,
            truncation=True,
            max_length=self.hparams.max_token_length,
        )
        y_tokens: dict[str, Tensor] = self.tokenizer(
            texts=y,
            return_type="pt",
            padding=True,
            truncation=True,
            return_attention_mask=False,
            max_length=self.hparams.max_token_length,
        )["input_ids"]
        y_hat = self.model(x_tokens)
        loss = self.loss_fn(y_tokens, y_hat)
        self.log("loss/train_loss", loss)

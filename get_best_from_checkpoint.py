from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from datamodule import Datamodule
from params import (
    RANDOM_SEED,
    LocationConfig,
    TrainingConfig,
    WandbConfig,
)
from cnn4 import CNN4


def get_all_checkpoints():
    checkpoints_dir = Path(LocationConfig.checkpoints_dir)
    list_of_checkpoints = list(checkpoints_dir.glob("*.ckpt"))

    return list_of_checkpoints


def init_CNN4() -> CNN4:
    cnn4 = CNN4(lr=TrainingConfig.lr)

    return cnn4


def save_model_from_last_checkpoint_as_state_dict() -> None:
    list_of_checkpoints = get_all_checkpoints()
    latest_checkpoint_path = max(list_of_checkpoints, key=lambda p: p.stat().st_ctime)

    lightning_model = init_CNN4()
    lightning_model.load_from_checkpoint(latest_checkpoint_path)
    lightning_model.eval()
    lightning_model = lightning_model.cpu()

    best_model_path = Path(LocationConfig.best_model)
    torch.save(lightning_model.state_dict(), best_model_path)

    print("Saved the latest model at:", best_model_path)

save_model_from_last_checkpoint_as_state_dict()

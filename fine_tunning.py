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
    list_of_checkpoints = get_all_checkpoints()
    latest_checkpoint_path = max(list_of_checkpoints, key=lambda p: p.stat().st_ctime)

    cnn4 = CNN4(lr=TrainingConfig.lr)
    state_dict = torch.load(LocationConfig.best_model)
    cnn4.load_state_dict(state_dict)
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


def run_train(dm: Datamodule, model: CNN4):

    chkp_dir = Path(LocationConfig.checkpoints_dir)
    modelCheckpoint = ModelCheckpoint(
        dirpath=chkp_dir,
        save_top_k=1,
        verbose=True,
        monitor="val_loss_epoch",
        mode="min",
    )

    earlyStopping = EarlyStopping(
        monitor="val_loss_epoch",
        patience=TrainingConfig.patience,
    )

    wandb_logger = WandbLogger(
        project=WandbConfig.project_name,
        save_dir=WandbConfig.save_dir,
        name=WandbConfig.run_name,
        entity=WandbConfig.entity,
    )

    trainer = Trainer(
        max_epochs=TrainingConfig.epochs,
        gpus=TrainingConfig.gpus,
        deterministic=TrainingConfig.deterministic,
        accumulate_grad_batches=TrainingConfig.accumulate_grad_batches,
        callbacks=[earlyStopping, modelCheckpoint],
        logger=wandb_logger,
    )

    # Train model
    trainer.fit(model, dm)

    # Save model
    save_model_from_last_checkpoint_as_state_dict()


def init_output_dirs() -> None:
    Path(LocationConfig.checkpoints_dir).mkdir(exist_ok=True, parents=True)
    Path(LocationConfig.best_model).parent.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    init_output_dirs()
    pl.seed_everything(RANDOM_SEED)

    train_data_path = Path(LocationConfig.train_data)
    test_data_path = Path(LocationConfig.test_data)
    dm = Datamodule(
        batch_size=TrainingConfig.batch_size,
        train_dir=train_data_path,
        val_dir=test_data_path,
    )
    model = init_CNN4()

    run_train(dm, model)

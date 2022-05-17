from pathlib import Path

import pytorch_lightning as pl
import torch
import wandb
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


def init_CNN4(lr, batch_norm, negative_slope, dropout, batch_size) -> CNN4:
    cnn4 = CNN4(
        lr=lr,
        batch_norm=batch_norm,
        negative_slope=negative_slope,
        dropout = dropout,
        batch_size = batch_size
        )
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


def run_train(dm: Datamodule, model: CNN4, run_name: str):

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
        name=run_name,
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
    # save_model_from_last_checkpoint_as_state_dict()
    
    wandb.finish()

    
def sweep_iteration():
    
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
    
    # set up W&B logger
    
    wandb.init()    # required to have access to `wandb.config`
    run_name = str(TrainingConfig.negative_slope) + '_'
    run_name += str(TrainingConfig.dropout)
    wandb_logger = WandbLogger(
        project=WandbConfig.project_name,
        save_dir=WandbConfig.save_dir,
        name=run_name,
        entity=WandbConfig.entity,
    )
    
    train_data_path = Path(LocationConfig.new_data + 'train/')
    test_data_path = Path(LocationConfig.new_data + 'test/')
    dm = Datamodule(
        batch_size=wandb.config.batch_size,
        train_dir=train_data_path,
        val_dir=test_data_path,
    )
    
    # setup model - note how we refer to sweep parameters with wandb.config
    model = init_CNN4(
        batch_norm=TrainingConfig.batch_norm, 
        lr=wandb.config.lr, 
        negative_slope=TrainingConfig.negative_slope, 
        dropout=TrainingConfig.dropout,
        batch_size = wandb.config.batch_size
    )

    # setup Trainer
    trainer = Trainer(
        max_epochs=TrainingConfig.epochs,
        gpus=TrainingConfig.gpus,
        deterministic=TrainingConfig.deterministic,
        accumulate_grad_batches=TrainingConfig.accumulate_grad_batches,
        callbacks=[earlyStopping, modelCheckpoint],
        logger=wandb_logger,
    )

    # train
    trainer.fit(model, dm)
    
    
def init_output_dirs() -> None:
    Path(LocationConfig.checkpoints_dir).mkdir(exist_ok=True, parents=True)
    Path(LocationConfig.best_model).parent.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    init_output_dirs()
    pl.seed_everything(RANDOM_SEED)

    train_data_path = Path(LocationConfig.new_data + 'train/')
    test_data_path = Path(LocationConfig.new_data + 'test/')
    dm = Datamodule(
        batch_size=TrainingConfig.batch_size,
        train_dir=train_data_path,
        val_dir=test_data_path,
    )
    
    sweep_config = {
      "method": "grid",   # Random search
      "metric": {           # We want to maximize val_acc
          "name": "val_loss_epoch",
          "goal": "minimize"
      },
      "parameters": {
#             "batch_norm": {"values": [False]}, # False
#             "dropout": {"values": [0.0, 0.1]}, # 0.0
#             "negative_slope": {"values": [0.0, 0.01, 0.02, 0.05, 0.1]}, # 0.0
            "lr": {"values": [5e-6, 1e-5, 2.5e-5, 5e-5, 1e-4]}, # 1e-5
            "batch_size": {"values": [4, 8]},
        }
    }
    
    sweep_id = wandb.sweep(sweep_config, project=WandbConfig.project_name)
    wandb.agent(sweep_id, function=sweep_iteration)
import json
import logging
from pathlib import Path

import hydra
import random
import numpy as np
import pytorch_lightning as pl
import torch
import os
import torchaudio
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from clarity.dataset.cec3_dataset import CEC3Dataset
from clarity.engine.losses import SNRLoss
from clarity.engine.system import System
#from clarity.enhancer.dnn.mc_conv_tasnet import ConvTasNet
from mc_conv_tasnet import ConvTasNet

logger = logging.getLogger(__name__)


class DenModule(System):
    def common_step(self, batch, batch_nb, train=True):
        mix, target, auxiliary, _  = batch
        enhanced = self.model(mix)
        loss = self.loss_func(enhanced, target)
        return loss

def train_den(cfg):
    exp_dir = Path(cfg.path.exp) / "denoising_module"
    if (exp_dir / "best_model.pth").exists():
        logger.info("Enhancement module exist")
        return

    train_set = CEC3Dataset(**cfg.train_dataset)
    train_loader = DataLoader(dataset=train_set, **cfg.train_loader)
    dev_set = CEC3Dataset(**cfg.dev_dataset)
    dev_loader = DataLoader(dataset=dev_set, **cfg.dev_loader)
    den_model = ConvTasNet(**cfg.mc_conv_tasnet)
    optimizer = torch.optim.Adam(
        params=den_model.parameters(), **cfg.den_trainer.optimizer
    )
    loss_func = SNRLoss()

    den_module = DenModule(
        model=den_model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=dev_loader,
        config=cfg,
    )

    # callbacks
    callbacks = []
    checkpoint_dir = exp_dir / "checkpoints/"
    checkpoint = ModelCheckpoint(
        str(checkpoint_dir), monitor="val_loss", mode="min", save_top_k=5, verbose=True
    )
    callbacks.append(checkpoint)

    # set device
    gpus = -1 if torch.cuda.is_available() else "auto"

    trainer = pl.Trainer(
        max_epochs=cfg.den_trainer.epochs,
        callbacks=callbacks,
        default_root_dir=str(exp_dir),
        accelerator="auto",
        devices=gpus,
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=cfg.den_trainer.gradient_clip_val,
    )
    #print("Before Fit")
    #print("############")
    trainer.fit(den_module)
    #print("After Fit")
    #print("************")

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with (exp_dir / "best_k_models.json").open("w", encoding="utf-8") as fp:
        json.dump(best_k, fp, indent=0)
    state_dict = torch.load(checkpoint.best_model_path)
    den_module.load_state_dict(state_dict=state_dict["state_dict"])
    den_module.cpu()
    torch.save(den_module.model.state_dict(), str(exp_dir / "best_model.pth"))

@hydra.main(config_path=".", config_name="config_den_lr_5", version_base="1.1")
def run(cfg: DictConfig) -> None:
    logger.info("Begin training ear enhancement module.")
    train_den(cfg)

# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    run()

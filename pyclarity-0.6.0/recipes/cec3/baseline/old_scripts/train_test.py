import json
import logging
import random
from pathlib import Path

import os
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torchaudio
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from amplifier import Amplifier
from loss import DHASPILevelLoss

from clarity.dataset.cec1_dataset import CEC1Dataset
from clarity.dataset.cec1_dataset import CEC1DatasetTrain
from clarity.engine.losses import SNRLoss, STOILevelLoss
from clarity.engine.system import System
from clarity.enhancer.dnn.mc_conv_tasnet import ConvTasNet
from clarity.enhancer.dsp.filter import AudiometricFIR
from clarity.predictor.torch_msbg import MSBGHearingModel
from torch_haspi import interpolate_HL

logger = logging.getLogger(__name__)


class DenModule(System):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ear_idx = None
        self.down_sample = None

    def common_step(self, batch, batch_nb, train=True):
        if self.down_sample is None:
            raise RuntimeError("Hearing model not loaded")
        proc, ref = batch
        ref = ref[:, self.ear_idx, :]
        if self.config.downsample_factor != 1:
            proc = self.down_sample(proc)
            ref = self.down_sample(ref)
        enhanced = self.model(proc).squeeze(1)
        # print("****************************************************************_________*********")
        # print(f"Reference shape: {enhanced.shape}, Estimated shape: {ref.shape}")
        loss = self.loss_func(enhanced, ref)
        return loss
    
    # def common_step(self, batch, batch_nb, train=True):
    #     mix, target = batch
    #     enhanced = self.model(mix)
    #     loss = self.loss_func(enhanced, target)
    #     return loss


class AmpModule(System):
    def sample_audiogram(self):
        audiogram = random.choice(self.audiograms)
        haspi_audiogram = interpolate_HL(audiogram)
        return audiogram, haspi_audiogram

    def common_step(self, batch, batch_nb, train=True):
        #mix, target, _ = batch
        mix, target = batch
        hl, hl_hapsi = self.sample_audiogram()
        hl = torch.tensor(hl, dtype=torch.float32).unsqueeze(0).to(mix.device)
        hl_hapsi = torch.tensor(hl_hapsi, dtype=torch.float32).to(mix.device)

        # denoised = self.den_model(mix)
        # denoised = denoised.view(denoised.shape[0] * denoised.shape[1], 1, -1)
        # enhanced = self.model(hl, denoised).squeeze()
        enhanced = self.model(
            hl, target.view(target.shape[0] * target.shape[1], 1, -1)
        ).squeeze()
        enhanced = torch.tanh(enhanced)  # soft_clip
        target = target.view(target.shape[0] * target.shape[1], -1)
        enhanced = self.down_sample(enhanced)
        target = self.down_sample(target)
        loss, level_loss, dhaspi_loss = self.loss_func(target, enhanced, hl_hapsi)

        self.logger.experiment.add_scalars(
            "loss",
            {"Loss": loss, "Level_loss": level_loss, "DHASPI_loss": dhaspi_loss},
            self.global_step,
        )

        return loss

#def train_den(cfg, ear):
#    exp_dir = Path(cfg.path.exp) / f"{ear}_den"
def train_den(cfg):
    exp_dir = Path(cfg.path.exp) / "denoising_module"
    if (exp_dir / "best_model.pth").exists():
        logger.info("Enhancement module exist")
        return

    train_set = CEC1DatasetTrain(**cfg.train_dataset)
    train_loader = DataLoader(dataset=train_set, **cfg.train_loader)
    dev_set = CEC1Dataset(**cfg.dev_dataset)
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
    # den_module.ear_idx = 0 if ear == "left" else 1
    # if cfg.downsample_factor != 1:
    #     den_module.down_sample = torchaudio.transforms.Resample(
    #         orig_freq=cfg.sample_rate,
    #         new_freq=cfg.sample_rate // cfg.downsample_factor,
    #         resampling_method="sinc_interp_hann",
    #     )

    # callbacks
    callbacks = []
    checkpoint_dir = exp_dir / "checkpoints/"
    checkpoint = ModelCheckpoint(
        str(checkpoint_dir), monitor="val_loss", mode="min", save_top_k=5, verbose=True
    )
    callbacks.append(checkpoint)

    # set device
    gpus = -1 if torch.cuda.is_available() else None
    # if torch.cuda.is_available():
    #     devices = [0]
    #     accelerator = 'gpu'
    # else:
    #     devices = 2
    #     accelerator = 'cpu'
    # devices=2
    # accelerator="cpu"

    trainer = pl.Trainer(
        max_epochs=cfg.den_trainer.epochs,
        callbacks=callbacks,
        default_root_dir=str(exp_dir),
        # devices=devices,
        # accelerator=accelerator,
        # gpus=gpus,
        accelerator="auto",
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=cfg.den_trainer.gradient_clip_val,
    )
    trainer.fit(den_module)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with (exp_dir / "best_k_models.json").open("w", encoding="utf-8") as fp:
        json.dump(best_k, fp, indent=0)
    state_dict = torch.load(checkpoint.best_model_path)
    den_module.load_state_dict(state_dict=state_dict["state_dict"])
    den_module.cpu()
    torch.save(den_module.model.state_dict(), str(exp_dir / "best_model.pth"))


# def train_amp(cfg, ear):
#     exp_dir = Path(cfg.path.exp) / f"{ear}_amp"
def train_amp(cfg):
    exp_dir = Path(cfg.path.exp) / "amp_module"
    Path.mkdir(exp_dir, parents=True, exist_ok=True)
    if (exp_dir / "best_model.pth").exists():
        logger.info("Amplification module exist")
        return

    train_set = CEC1DatasetTrain(**cfg.train_dataset)
    train_loader = DataLoader(dataset=train_set, **cfg.train_loader)
    dev_set = CEC1Dataset(**cfg.dev_dataset)
    dev_loader = DataLoader(dataset=dev_set, **cfg.dev_loader)

    # load denoising module
    # den_model = ConvTasNet(**cfg.mc_conv_tasnet)
    # den_model_path = exp_dir / ".." / "denoising_module/best_model.pth"
    # den_model.load_state_dict(torch.load(den_model_path))

    # amplification module
    # amp_model = AudiometricFIR(**cfg.fir)
    amp_model = Amplifier(**cfg.amplifier)
    optimizer = torch.optim.Adam(
        params=amp_model.parameters(), **cfg.amp_trainer.optimizer
    )
    # loss_func = STOILevelLoss(**cfg.amp_trainer.stoilevel_loss)
    loss_func = DHASPILevelLoss(**cfg.amp_trainer.dhaspi_loss)

    amp_module = AmpModule(
        model=amp_model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=dev_loader,
        config=cfg,
    )

    # get all audiograms and convert them to HASPI audiometric scale
    listeners = json.load(
        open(
            os.path.join(
                cfg.path.cec2_root,
                "clarity_CEC2_data/clarity_data/metadata/listeners.json",
            )
        )
    )
    audiograms = []
    # HASPI audiometric
    aud_haspi = [250, 500, 1000, 2000, 4000, 6000]
    for listener in listeners:
        aud_cfs = listeners[listener]["audiogram_cfs"]
        hl_l = listeners[listener]["audiogram_levels_l"]
        hl_r = listeners[listener]["audiogram_levels_r"]
        aud_l = [hl_l[i] for i in range(len(aud_cfs)) if aud_cfs[i] in aud_haspi]
        aud_r = [hl_r[i] for i in range(len(aud_cfs)) if aud_cfs[i] in aud_haspi]
        audiograms.append(aud_l)
        audiograms.append(aud_r)

    # add attributes to den_module
    amp_module.audiograms = audiograms
    # amp_module.den_model = den_model
    amp_module.down_sample = torchaudio.transforms.Resample(
        orig_freq=cfg.sr, new_freq=cfg.amp_trainer.dhaspi_loss.sr
    )

    # callbacks
    callbacks = []
    checkpoint_dir = exp_dir / "checkpoints/"
    checkpoint = ModelCheckpoint(
        str(checkpoint_dir), monitor="val_loss", mode="min", save_top_k=5, verbose=True
    )
    callbacks.append(checkpoint)

    # set device
    gpus = -1 if torch.cuda.is_available() else None

    trainer = pl.Trainer(
        max_epochs=cfg.amp_trainer.epochs,
        callbacks=callbacks,
        default_root_dir=exp_dir,
        # gpus=gpus,
        accelerator="auto",
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=cfg.amp_trainer.gradient_clip_val,
        # num_sanity_val_steps=cfg.amp_trainer.num_sanity_val_steps,
    )
    trainer.fit(amp_module)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with (exp_dir / "best_k_models.json").open("w", encoding="utf-8") as fp:
        json.dump(best_k, fp, indent=0)
    state_dict = torch.load(checkpoint.best_model_path)
    amp_module.load_state_dict(state_dict=state_dict["state_dict"])
    amp_module.cpu()
    torch.save(amp_module.model.state_dict(), str(exp_dir / "best_model.pth"))

@hydra.main(config_path=".", config_name="config", version_base="1.1")
def run(cfg: DictConfig) -> None:
    logger.info("Begin training ear enhancement module.")
    train_den(cfg)
    #logger.info("Begin training left ear enhancement module.")
    #train_den(cfg, ear="left")
    #logger.info("Begin training right ear enhancement module.")
    #train_den(cfg, ear="right")
    # logger.info("Begin training left ear amplification module.")
    # train_amp(cfg, ear="left")
    # logger.info("Begin training right ear amplification module.")
    # train_amp(cfg, ear="right")
    logger.info("Begin training ear amplification module.")
    train_amp(cfg)

# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    run()

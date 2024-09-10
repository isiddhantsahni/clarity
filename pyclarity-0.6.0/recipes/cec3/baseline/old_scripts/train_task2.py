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

from clarity.dataset.cec1_dataset import CEC1Dataset, CEC3Dataset
from clarity.dataset.cec1_dataset import CEC1DatasetTrain
from clarity.engine.losses import SNRLoss, STOILevelLoss
from clarity.engine.system import System
from clarity.enhancer.dnn.mc_conv_tasnet import ConvTasNet
from clarity.enhancer.dsp.filter import AudiometricFIR
from clarity.predictor.torch_msbg import MSBGHearingModel

logger = logging.getLogger(__name__)


class DenModule(System):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ear_idx = None
        self.down_sample = None

    #def common_step(self, batch, batch_nb, train=True):
        #if self.down_sample is None:
        #    raise RuntimeError("Hearing model not loaded")
        #proc, ref = batch
        #ref = ref[:, self.ear_idx, :]
        #if self.config.downsample_factor != 1:
        #    proc = self.down_sample(proc)
        #    ref = self.down_sample(ref)
        #enhanced = self.model(proc)#.squeeze(1)
        # print("****************************************************************_________*********")
        # print(f"Reference shape: {enhanced.shape}, Estimated shape: {ref.shape}")
        #loss = self.loss_func(enhanced, ref)
        #return loss
    
    def common_step(self, batch, batch_nb, train=True):
        mix, target, _ = batch
        enhanced = self.model(mix)
        loss = self.loss_func(enhanced, target)
        return loss


class AmpModule(System):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hl_ear = None
        self.nh_ear = None
        self.down_sample = None
        self.up_sample = None
        self.ear_idx = None
        self.den_model = None
        self.cfg = self.config
        # self.audiograms = None
        # Load all audiograms once during initialization
        self.audiograms = {}
        with open(self.cfg.listener.metafile, encoding="utf-8") as fp:
            listeners_file = json.load(fp)
            for listener_id, data in listeners_file.items():
                self.audiograms[listener_id] = {
                    "cfs": data["audiogram_cfs"],
                    "left": data["audiogram_levels_l"],
                    "right": data["audiogram_levels_r"]
                }
    
    def select_random_listener(self):
        return random.choice(list(self.audiograms.keys()))
    
    def set_audiogram(self, listener_id, ear):
        # Select the audiogram data for the given listener and ear
        audiogram_cfs = self.audiograms[listener_id]["cfs"]
        audiogram = self.audiograms[listener_id][ear]

        # Create the normal hearing and hearing loss models
        self.nh_ear = MSBGHearingModel(
            audiogram=np.zeros_like(audiogram), audiometric=audiogram_cfs, sr=self.cfg.sample_rate
        )
        self.hl_ear = MSBGHearingModel(
            audiogram=audiogram, audiometric=audiogram_cfs, sr=self.cfg.sample_rate
        )


    def common_step(self, batch, batch_nb, train=True):
        #if (
        #    self.hl_ear is None
        #    or self.nh_ear is None
        #    or self.down_sample is None
        #    or self.up_sample is None
        #    or self.den_model is None
        #):
        #    raise RuntimeError("Hearing model not loaded")
        proc, ref, _ = batch
        ref = ref[:, self.ear_idx, :]

        # Randomly select a listener ID and set audiogram for the current batch
        # listener_id = random.choice(list(self.audiograms.keys()))
        listener_id = self.select_random_listener()
        ear_str = 'left' if self.ear_idx == 0 else 'right'
        self.set_audiogram(listener_id, ear_str)
        
        #logger.info("##################")
        #logger.info("After setting Audiogram")

        if self.config.downsample_factor != 1:
            proc = self.down_sample(proc)
            ref = self.down_sample(ref)
        
        #if proc.shape[1] == 2:  # Example adjustment if you have 2 channels but need 1
        #    logger.info("********************************************")
            #    proc = proc[:, 0:1, :]
        #logger.info("###############################################")
        #logger.info(proc.shape)

        den_var = self.den_model(proc)
        # logger.info("@@@@@@@@@@@@@@")
        # logger.info(den_var.shape)
        den_var = den_var[:, self.ear_idx, :]
        # logger.info("___________________")
        # logger.info(den_var.shape)

        varvar = self.model(den_var)
        # logger.info("@@@@@@@@@@@@@@")
        # logger.info(varvar.shape)
        enhanced = varvar.squeeze(1)
        #enhanced = self.model(self.den_model(proc)).squeeze(1)
        if self.config.downsample_factor != 1:
            enhanced = torch.clamp(self.up_sample(enhanced), -1, 1)
            ref = torch.clamp(self.up_sample(ref), -1, 1)
        # logger.info("__________@@@@@______")
        # logger.info(enhanced.shape)
        #hl = torch.tensor(hl, dtype=torch.float32).unsqueeze(0).to(proc.device)
        #enhanced = self.model(
        #    hl, target.view(target.shape[0] * target.shape[1], 1, -1)
        #).squeeze()

        #if ref.shape[1] == 2:  # Example adjustment if you have 2 channels but need 1
            #logger.info("********************************************")
            #ref = ref[:, 0:1, :].squeeze(1)
            #logger.info(ref.shape)

        #enhanced = torch.tanh(enhanced)  # soft_clip
        #target = target.view(target.shape[0] * target.shape[1], -1)
        #enhanced = self.down_sample(enhanced)
        #target = self.down_sample(target)
        # logger.info(f"Shape of enhanced: {enhanced.shape}")
        # logger.info(f"Shape of ref: {ref.shape}")
        
        # logger.info("################")
        # logger.info(ref.shape)
        # if ref.shape[1] == 2:  # Example adjustment if you have 2 channels but need 1
        #     logger.info("********************************************")
        #     ref = ref[:, 0:1, :].squeeze(1)
        #     logger.info(ref.shape)

        sim_ref = self.nh_ear(ref)
        sim_enhanced = self.hl_ear(enhanced)
        loss = self.loss_func(sim_enhanced, sim_ref)
        # loss = self.loss_func(target, enhanced)
        #logger.info("Completed All things")
        #logger.info("******************")
        return loss


#def train_den(cfg, ear):
#    exp_dir = Path(cfg.path.exp) / f"{ear}_den"
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
    #den_module.ear_idx = 0 if ear == "left" else 1
    #if cfg.downsample_factor != 1:
    #    den_module.down_sample = torchaudio.transforms.Resample(
    #        orig_freq=cfg.sample_rate,
    #        new_freq=cfg.sample_rate // cfg.downsample_factor,
    #        resampling_method="sinc_interp_hann",
    #    )

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


def train_amp(cfg, ear):
    exp_dir = Path(cfg.path.exp) / f"{ear}_amp"
#def train_amp(cfg):
#    exp_dir = Path(cfg.path.exp) / "amp_module"
    Path.mkdir(exp_dir, parents=True, exist_ok=True)
    if (exp_dir / "best_model.pth").exists():
        logger.info("Amplification module exist")
        return

    train_set = CEC3Dataset(**cfg.train_dataset)
    train_loader = DataLoader(dataset=train_set, **cfg.train_loader)
    dev_set = CEC3Dataset(**cfg.dev_dataset)
    dev_loader = DataLoader(dataset=dev_set, **cfg.dev_loader)

    # load denoising module
    den_model = ConvTasNet(**cfg.mc_conv_tasnet)
    den_model_path = exp_dir / ".." / "denoising_module/best_model.pth"
    den_model.load_state_dict(torch.load(den_model_path))

    # amplification module
    amp_model = AudiometricFIR(**cfg.fir)
    optimizer = torch.optim.Adam(
        params=amp_model.parameters(), **cfg.amp_trainer.optimizer
    )
    loss_func = STOILevelLoss(**cfg.amp_trainer.stoilevel_loss)

    amp_module = AmpModule(
        model=amp_model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=dev_loader,
        config=cfg,
    )
    amp_module.ear_idx = 0 if ear == "left" else 1
    amp_module.den_model = den_model
    if cfg.downsample_factor != 1:
        amp_module.down_sample = torchaudio.transforms.Resample(
            orig_freq=cfg.sample_rate,
            new_freq=cfg.sample_rate // cfg.downsample_factor,
            resampling_method="sinc_interp_hann",
        )
        amp_module.up_sample = torchaudio.transforms.Resample(
            orig_freq=cfg.sample_rate // cfg.downsample_factor,
            new_freq=cfg.sample_rate,
            resampling_method="sinc_interp_hann",
        )

    # audiograms = []
    #amp_module.nh_ear = []
    #amp_module.hl_ear = []
    # build normal hearing and hearing loss ears
    # with open(cfg.listener.metafile, encoding="utf-8") as fp:
    #     listeners_file = json.load(fp)
    #     audiogram_cfs = listeners_file[cfg.listener.id]["audiogram_cfs"]
    #     audiogram_lvl_l = listeners_file[cfg.listener.id]["audiogram_levels_l"]
    #     audiogram_lvl_r = listeners_file[cfg.listener.id]["audiogram_levels_r"]
    # audiogram = audiogram_lvl_l if ear == "left" else audiogram_lvl_r

    # get all audiograms and convert them to HASPI audiometric scale
    # listeners = json.load(open(cfg.listener.metafile))

    # Do this to make more listener audiograms
    # audiograms = []
    # # HASPI audiometric
    # aud_haspi = [250, 500, 1000, 2000, 4000, 6000]
    # for listener in listeners:
    #     audiogram_cfs = listeners[listener]["audiogram_cfs"]
    #     audiogram_lvl_l = listeners[listener]["audiogram_levels_l"]
    #     audiogram_lvl_r = listeners[listener]["audiogram_levels_r"]
    #     if ear == "left":
    #         audiograms.append(audiogram_lvl_l)
    #     else:
    #         audiograms.append(audiogram_lvl_r)
    #     aud_l = [hl_l[i] for i in range(len(aud_cfs)) if aud_cfs[i] in aud_haspi]
    #     aud_r = [hl_r[i] for i in range(len(aud_cfs)) if aud_cfs[i] in aud_haspi]
    #     audiograms.append(aud_l)
    #     audiograms.append(aud_r)

    # amp_module.audiograms = audiograms
    
    # amp_module.nh_ear = MSBGHearingModel(
    #     audiogram=np.zeros_like(audiogram), audiometric=audiogram_cfs, sr=cfg.sample_rate
    # )
    # amp_module.hl_ear = MSBGHearingModel(
    #     audiogram=audiogram, audiometric=audiogram_cfs, sr=cfg.sample_rate
    # )
    
    #audiograms.append(audiogram_lvl_l)
    #audiograms.append(audiogram_lvl_r)

    #for audiogram in audiograms:
    #    nh_ear_model = MSBGHearingModel(
    #        audiogram=np.zeros_like(audiogram), audiometric=audiogram_cfs, sr=cfg.sample_rate
    #    )
    #    hl_ear_model = MSBGHearingModel(
    #        audiogram=audiogram, audiometric=audiogram_cfs, sr=cfg.sample_rate
    #    )
    #    amp_module.nh_ear.append(nh_ear_model)
    #    amp_module.hl_ear.append(hl_ear_model)
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
        num_sanity_val_steps=cfg.amp_trainer.num_sanity_val_steps,
    )
    trainer.fit(amp_module)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with (exp_dir / "best_k_models.json").open("w", encoding="utf-8") as fp:
        json.dump(best_k, fp, indent=0)
    state_dict = torch.load(checkpoint.best_model_path)
    amp_module.load_state_dict(state_dict=state_dict["state_dict"])
    amp_module.cpu()
    torch.save(amp_module.model.state_dict(), str(exp_dir / "best_model.pth"))


@hydra.main(config_path=".", config_name="config_task2", version_base="1.1")
def run(cfg: DictConfig) -> None:
    logger.info("Begin training ear enhancement module.")
    train_den(cfg)
    #logger.info("Begin training left ear enhancement module.")
    #train_den(cfg, ear="left")
    #logger.info("Begin training right ear enhancement module.")
    #train_den(cfg, ear="right")
    logger.info("Begin training left ear amplification module.")
    train_amp(cfg, ear="left")
    logger.info("Begin training right ear amplification module.")
    train_amp(cfg, ear="right")
    #logger.info("Begin training ear amplification module.")
    #train_amp(cfg)

# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    run()

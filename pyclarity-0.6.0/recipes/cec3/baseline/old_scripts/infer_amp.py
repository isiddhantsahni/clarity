import os
from pathlib import Path
import json
import hashlib
import logging
import numpy as np
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from soundfile import read, write

import torch
from clarity.enhancer.dsp.filter import AudiometricFIR
from clarity.utils.audiogram import Audiogram, Listener

logger = logging.getLogger(__name__)

@hydra.main(config_path=".", config_name="config_task2")
def run(cfg: DictConfig) -> None:
    output_folder = os.path.join(cfg.path.exp, "amplified_signals")
    os.makedirs(output_folder, exist_ok=True)
    # denoised signal folder
    denoised_folder = os.path.join(cfg.path.exp, "denoised_signals")

    with open(cfg.path.scenes_listeners_file, encoding="utf-8") as fp:
        scenes_listeners = json.load(fp)

    device = "cuda" if torch.cuda.is_available() else None
    # Amplification model left
    amp_model_left_path = os.path.join(os.path.join(cfg.path.exp, "left_amp"), "best_model.pth",)
    amp_model_left = AudiometricFIR(**cfg.fir)
    amp_model_left.load_state_dict(torch.load(amp_model_left_path, map_location=device))
    amp_model_left.eval()

    # Amplification model right
    amp_model_right_path = os.path.join(os.path.join(cfg.path.exp, "right_amp"),"best_model.pth",)
    amp_model_right = AudiometricFIR(**cfg.fir)
    amp_model_right.load_state_dict(torch.load(amp_model_right_path, map_location=device))
    amp_model_right.eval()

    for scene in tqdm(scenes_listeners):
        for listener in scenes_listeners[scene]:
            logger.info(f"Running SI calculation: scene {scene}, listener {listener}")

            # read denoised signals
            wav, sr = read(os.path.join(denoised_folder, scene + "_den.wav"))
            wav_left_torch = torch.tensor(wav[:, 0], dtype=torch.float32).view(1, 1, -1).to(device)
            wav_right_torch = torch.tensor(wav[:, 1], dtype=torch.float32).view(1, 1, -1).to(device)

            amp_left = torch.tanh(amp_model_left(wav_left_torch)).squeeze(1)
            # amp_left = torch.clamp(amp_left, -1, 1)
            amp_left = amp_left.cpu().detach().numpy()[0]
            amp_right = torch.tanh(amp_model_right(wav_right_torch)).squeeze(1)
            # amp_right = torch.clamp(amp_right, -1, 1)
            amp_right = amp_right.cpu().detach().numpy()[0]

            amplified = np.stack([amp_left, amp_right], axis=0).T
            write(
                os.path.join(output_folder, scene + "_" + listener + "_HA-output.wav"),
                amplified,
                sr,
            )

if __name__ == "__main__":
    run()


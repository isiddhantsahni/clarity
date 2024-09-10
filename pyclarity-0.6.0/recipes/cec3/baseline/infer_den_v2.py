import os
from pathlib import Path

import hydra
import numpy as np
import torch
import torchaudio
from omegaconf import DictConfig
from soundfile import write
from torch.utils.data import DataLoader
from tqdm import tqdm

from clarity.dataset.cec3_dataset import CEC3Dataset
# from clarity.enhancer.dnn.mc_conv_tasnet import ConvTasNet
from mc_conv_tasnet import ConvTasNet


@hydra.main(config_path=".", config_name="config_den")
def run(cfg: DictConfig) -> None:
    output_folder = os.path.join(cfg.path.exp, "denoised_signals")
    os.makedirs(output_folder, exist_ok=True)

    # dev dataloader
    test_set = CEC3Dataset(**cfg.dev_dataset)
    test_loader = DataLoader(dataset=test_set, **cfg.dev_loader)

    # load denoising model
    device = "cuda" if torch.cuda.is_available() else None
    den_model = ConvTasNet(**cfg.mc_conv_tasnet)
    den_model_path = os.path.join(
        os.path.join(cfg.path.exp, "denoising_module"),
        "best_model.pth",
    )
    den_model.load_state_dict(torch.load(den_model_path, map_location=device))
    den_model = torch.nn.parallel.DataParallel(den_model.to(device))
    den_model.eval()

    # infer
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="testing"):
            #if cfg.test_dataset.testing:
            #    mix, auxiliary, wav_len, scene = batch
            #else:
            mix, target, auxiliary, scene = batch
            mix = mix.to(device)
            target = target.to(device)
            auxiliary = auxiliary.to(device)
            enhanced = den_model(mix)
            enhanced = enhanced[0].cpu().detach().numpy().T
            write(os.path.join(output_folder, scene[0] + "_den.wav"), enhanced, cfg.sample_rate)

if __name__ == "__main__":
    run()

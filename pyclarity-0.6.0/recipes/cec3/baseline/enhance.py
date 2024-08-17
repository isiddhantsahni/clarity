"""Run enhancement. """

import json
import logging
import pathlib

import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.io import wavfile
from tqdm import tqdm
import torchaudio
import torch
from pathlib import Path

from clarity.enhancer.compressor import Compressor
from clarity.enhancer.nalr import NALR
from clarity.utils.audiogram import Audiogram, Listener
from recipes.icassp_2023.baseline.evaluate import make_scene_listener_list
from clarity.enhancer.dnn.mc_conv_tasnet import ConvTasNet
from clarity.enhancer.dsp.filter import AudiometricFIR

logger = logging.getLogger(__name__)


def amplify_signal(signal, audiogram: Audiogram, enhancer, compressor):
    """Amplify signal for a given audiogram"""
    nalr_fir, _ = enhancer.build(audiogram)
    out = enhancer.apply(nalr_fir, signal)
    out, _, _ = compressor.process(out)
    return out


@hydra.main(config_path=".", config_name="config")
def enhance(cfg: DictConfig) -> None:
    # """Run the dummy enhancement."""
    """Running the convtasnet enhancement"""

    enhanced_folder = pathlib.Path(cfg.path.exp) / "enhanced_signals"
    enhanced_folder.mkdir(parents=True, exist_ok=True)

    with open(cfg.path.scenes_listeners_file, encoding="utf-8") as fp:
        scenes_listeners = json.load(fp)

    listener_dict = Listener.load_listener_dict(cfg.path.listeners_file)
    # enhancer = NALR(**cfg.nalr)
    # compressor = Compressor(**cfg.compressor)
    amplified_folder = pathlib.Path(cfg.path.exp) / "amplified_signals"
    amplified_folder.mkdir(parents=True, exist_ok=True)

    # Make list of all scene listener pairs that will be run
    scene_listener_pairs = make_scene_listener_list(
        scenes_listeners, cfg.evaluate.small_test
    )

    down_sample = up_sample = None
    if cfg.downsample_factor != 1:
        down_sample = torchaudio.transforms.Resample(
            orig_freq=cfg.sample_rate,
            new_freq=cfg.sample_rate // cfg.downsample_factor,
            resampling_method="sinc_interp_hann",
        )
        up_sample = torchaudio.transforms.Resample(
            orig_freq=cfg.sample_rate // cfg.downsample_factor,
            new_freq=cfg.sample_rate,
            resampling_method="sinc_interp_hann",
        )

    device = "cuda" if torch.cuda.is_available() else None

    with torch.no_grad():
        for scene, listener_id in tqdm(scene_listener_pairs):
            sample_rate, signal_ch1 = wavfile.read(
                pathlib.Path(cfg.path.scenes_folder) / f"{scene}_mix_CH1.wav"
            )

            _, signal_ch2 = wavfile.read(
                pathlib.Path(cfg.path.scenes_folder) / f"{scene}_mix_CH2.wav"
            )

            _, signal_ch3 = wavfile.read(
                pathlib.Path(cfg.path.scenes_folder) / f"{scene}_mix_CH3.wav"
            )

            # Convert to 32-bit floating point scaled between -1 and 1
            signal_ch1 = (signal_ch1 / 32768.0).astype(np.float32)
            signal_ch2 = (signal_ch2 / 32768.0).astype(np.float32)
            signal_ch3 = (signal_ch3 / 32768.0).astype(np.float32)

            signal = (signal_ch1 + signal_ch2 + signal_ch3) / 3

            # pylint: disable=unused-variable
            # listener = listener_dict[listener_id]  # noqa: F841

            out = []
            for ear in ["left", "right"]:
                torch.cuda.empty_cache()
                # load denoising module
                den_model = ConvTasNet(**cfg.mc_conv_tasnet)
                den_model_path = Path(cfg.path.exp) / "denoising_module/best_model.pth"
                # den_model_path = exp_folder / f"{ear}_den/best_model.pth"

                den_model.load_state_dict(
                    torch.load(den_model_path, map_location=device)
                )
                _den_model = torch.nn.parallel.DataParallel(den_model.to(device))
                _den_model.eval()

                # load amplification module
                amp_model = AudiometricFIR(**cfg.fir)
                amp_model_path = Path(cfg.path.exp) / f"{ear}_amp/best_model.pth"
                # amp_model_path = exp_folder / f"{ear}_amp/best_model.pth"

                amp_model.load_state_dict(
                    torch.load(amp_model_path, map_location=device)
                )
                _amp_model = torch.nn.parallel.DataParallel(amp_model.to(device))
                _amp_model.eval()

                signal_tensor = torch.from_numpy(signal)
                signal_tensor = signal_tensor.to(device)
                print("################@@@@@")
                print(signal_tensor.shape)
                proc = signal_tensor
                #if down_sample is not None:
                #    proc = down_sample(signal_tensor)
                #proc = proc.squeeze(1)
                #proc = proc.transpose(0, 1).unsqueeze(0)
                print("################")
                print(proc.shape)
                #if proc.shape[1] == 2:  # Example adjustment if you have 2 channels but need 1
                #    proc = proc[:, 0:1]
                print("################")
                print(proc.shape)    
                den_var = den_model(proc)
                if den_var.shape[1] == 2:
                    den_var = den_var[:, 0:1, :]

                varvar = amp_model(den_var)
                enhanced = varvar.squeeze(1)

                # enhanced = amp_model(den_model(proc)).squeeze(1)
                if up_sample is not None:
                    enhanced = up_sample(enhanced)
                enhanced = torch.clamp(enhanced, -1, 1)
                out.append(enhanced.detach().cpu().numpy()[0])

            out = np.stack(out, axis=0).transpose()

            # wavfile.write(
            #     enhanced_folder / f"{scene}_{listener_id}_enhanced.wav", sample_rate, signal
            # )

            # Apply the baseline NALR amplification

            # out_l = amplify_signal(
            #     signal[:, 0], listener.audiogram_left, enhancer, compressor
            # )
            # out_r = amplify_signal(
            #     signal[:, 1], listener.audiogram_right, enhancer, compressor
            # )
            # amplified = np.stack([out_l, out_r], axis=1)

            # if cfg.soft_clip:
            #     amplified = np.tanh(amplified)

            wavfile.write(
                amplified_folder / f"{scene}_{listener_id}_HA-output.wav",
                sample_rate,
                # out.astype(np.float32),
                out,
            )


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    enhance()


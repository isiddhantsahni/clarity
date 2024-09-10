import os
import csv
import json
import hashlib
import logging
import numpy as np
import hydra
import pathlib
from omegaconf import DictConfig
from tqdm import tqdm
from soundfile import read
from scipy.io import wavfile
from clarity.enhancer.nalr import NALR
from clarity.utils.audiogram import Audiogram, Listener
from clarity.enhancer.compressor import Compressor
from recipes.icassp_2023.baseline.evaluate import make_scene_listener_list

logger = logging.getLogger(__name__)

def amplify_signal(signal, audiogram: Audiogram, enhancer, compressor):
    """Amplify signal for a given audiogram"""
    nalr_fir, _ = enhancer.build(audiogram)
    out = enhancer.apply(nalr_fir, signal)
    out, _, _ = compressor.process(out)
    return out

@hydra.main(config_path=".", config_name="config_den_arch")
def run(cfg: DictConfig) -> None:
    enhanced_folder = os.path.join(cfg.path.exp, "denoised_signals")

    #output_folder = os.path.join(cfg.path.exp, "amplified_signals")
    #os.makedirs(output_folder, exist_ok=True)
  
    output_folder = pathlib.Path(cfg.path.exp) / "amplified_signals"
    output_folder.mkdir(parents=True, exist_ok=True)

    with open(cfg.path.scenes_listeners_file, encoding="utf-8") as fp:
        scenes_listeners = json.load(fp)
    
    listener_dict = Listener.load_listener_dict(cfg.path.listeners_file)

    enhancer = NALR(**cfg.nalr)
    compressor = Compressor(**cfg.compressor)

    # Make list of all scene listener pairs that will be run
    scene_listener_pairs = make_scene_listener_list(
        scenes_listeners, cfg.evaluate.small_test
    )
    
    for scene, listener_id in tqdm(scene_listener_pairs):
        logger.info(f"Running SI calculation: scene {scene}, listener {listener_id}")

        listener = listener_dict[listener_id]  # noqa: F841

        proc, sr = read(os.path.join(enhanced_folder, f"{scene}_den.wav"))

        # Apply the baseline NALR amplification

        out_l = amplify_signal(
            proc[:, 0], listener.audiogram_left, enhancer, compressor
        )
        out_r = amplify_signal(
            proc[:, 1], listener.audiogram_right, enhancer, compressor
        )

        amplified = np.stack([out_l, out_r], axis=1)

        if cfg.soft_clip:
            amplified = np.tanh(amplified)

        wavfile.write(
            output_folder / f"{scene}_{listener_id}_HA-output.wav",
            sr,
            amplified.astype(np.float32),
        )

if __name__ == "__main__":
    run()


"""Evaluate the enhanced signals using a combination of HASPI and HASQI metrics"""

import csv
import hashlib
import json
import logging
import pathlib

import hydra
import numpy as np
from numpy import ndarray
from omegaconf import DictConfig
from scipy.io import wavfile
from tqdm import tqdm

from clarity.evaluator.haspi import haspi_v2_be
from clarity.utils.audiogram import Listener

logger = logging.getLogger(__name__)


def set_scene_seed(scene):
    """Set a seed that is unique for the given scene"""
    scene_encoded = hashlib.md5(scene.encode("utf-8")).hexdigest()
    scene_md5 = int(scene_encoded, 16) % (10**8)
    np.random.seed(scene_md5)


def compute_metric(
    metric, signal: ndarray, ref: ndarray, listener: Listener, sample_rate: float
):
    """Compute HASPI or HASQI metric"""
    score = metric(
        reference_left=ref[:, 0],
        reference_right=ref[:, 1],
        processed_left=signal[:, 0],
        processed_right=signal[:, 1],
        sample_rate=sample_rate,
        listener=listener,
    )
    return score


class ResultsFile:
    """Class to write results to a CSV file"""

    def __init__(self, file_name):
        self.file_name = file_name

    def write_header(self):
        with open(self.file_name, "w", encoding="utf-8") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            csv_writer.writerow(["scene", "listener", "haspi"])

    def add_result(self, scene: str, listener: str, haspi: float):
        """Add a result to the CSV file"""

        logger.info(f"The HASPI score is {haspi})")

        with open(self.file_name, "a", encoding="utf-8") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            csv_writer.writerow([scene, listener, str(haspi)])


def make_scene_listener_list(scenes_listeners, small_test=False):
    """Make the list of scene-listener pairing to process"""
    scene_listener_pairs = [
        (scene, listener)
        for scene in scenes_listeners
        for listener in scenes_listeners[scene]
    ]

    # Can define a standard 'small_test' with just 1/15 of the data
    if small_test:
        scene_listener_pairs = scene_listener_pairs[::15]

    return scene_listener_pairs


@hydra.main(config_path=".", config_name="config_den_arch")
def run_calculate_si(cfg: DictConfig) -> None:
    """Evaluate the enhanced signals using a combination of HASPI and HASQI metrics"""

    # Load listener data
    with open(cfg.path.scenes_listeners_file, encoding="utf-8") as fp:
        scenes_listeners = json.load(fp)

    listeners_dict = Listener.load_listener_dict(cfg.path.listeners_file)

    amplified_folder = pathlib.Path(cfg.path.exp) / "amplified_signals"
    scenes_folder = pathlib.Path(cfg.path.scenes_folder)
    amplified_folder.mkdir(parents=True, exist_ok=True)

    # Make list of all scene listener pairs that will be run
    scene_listener_pairs = make_scene_listener_list(
        scenes_listeners, cfg.evaluate.small_test
    )

    # Define a range of scenes to evaluate
    i_first = cfg.evaluate.first_scene
    n_scenes = cfg.evaluate.n_scenes
    if n_scenes == 0:
        n_scenes = len(scene_listener_pairs) - i_first
    i_last = min(i_first + n_scenes, len(scene_listener_pairs))

    # Each range is stored in a separate file
    # make scores directory is not already existing
    scores_dir = pathlib.Path(cfg.path.exp) / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    results_file = ResultsFile(scores_dir / f"scores.{i_first}_{i_last}.csv")
    results_file.write_header()

    for scene, listener_id in tqdm(scene_listener_pairs[i_first:i_last]):
        logger.info(f"Running evaluation: scene {scene}, listener {listener_id}")

        if cfg.evaluate.set_random_seed:
            set_scene_seed(scene)

        # Read signals
        sr_signal, signal = wavfile.read(
            amplified_folder / f"{scene}_{listener_id}_HA-output.wav",
        )
        _, reference = wavfile.read(scenes_folder / f"{scene}_reference.wav")

        # if reference is mono then make stereo version
        if len(reference.shape) == 1:
            reference = np.stack([reference, reference], axis=1)

        reference = reference / 32768.0

        # Evaluate the HA-output signals
        listener = listeners_dict[listener_id]

        haspi_score = compute_metric(
            haspi_v2_be, signal, reference, listener, sr_signal
        )

        results_file.add_result(scene, listener_id, haspi_score)


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    run_calculate_si()

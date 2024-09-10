import os
from glob import glob
from tqdm import tqdm
import librosa
from soundfile import read, write

import hydra
from omegaconf import DictConfig

TARGET_FS = 24000


@hydra.main(config_path=".", config_name="config_den")
def run_resample(cfg):
    print("Doing train/scenes")
    # resample train/scenes
    orig_tr_scene_folder = os.path.join(
        cfg.path.cec3_task1, "clarity_data/train/scenes/"
    )
    target_tr_scene_folder = orig_tr_scene_folder.replace("scenes", "scenes_24k")
    os.makedirs(target_tr_scene_folder, exist_ok=True)
    tr_scene_wav_lst = sorted(glob(os.path.join(orig_tr_scene_folder, "*.wav")))
    for wavfile in tqdm(tr_scene_wav_lst):
        wav, fs = read(wavfile)
        resampled_wav = librosa.resample(wav.T, orig_sr=fs, target_sr=TARGET_FS)
        write(
            wavfile.replace(orig_tr_scene_folder, target_tr_scene_folder),
            resampled_wav.T,
            TARGET_FS,
        )

    print("Doing dev/scenes")
    # resample dev/scenes
    orig_dev_scene_folder = os.path.join(
        cfg.path.cec3_task1, "clarity_data/dev/scenes/"
    )
    target_dev_scene_folder = orig_dev_scene_folder.replace("scenes", "scenes_24k")
    os.makedirs(target_dev_scene_folder, exist_ok=True)
    dev_scene_wav_lst = sorted(glob(os.path.join(orig_dev_scene_folder, "*.wav")))
    for wavfile in tqdm(dev_scene_wav_lst):
        wav, fs = read(wavfile)
        resampled_wav = librosa.resample(wav.T, orig_sr=fs, target_sr=TARGET_FS)
        write(
            wavfile.replace(orig_dev_scene_folder, target_dev_scene_folder),
            resampled_wav.T,
            TARGET_FS,
        )

    print("Doing train/targets")
    # resample train/targets
    orig_tr_target_folder = os.path.join(
        cfg.path.cec3_task1, "clarity_data/train/targets/"
    )
    target_tr_target_folder = orig_tr_target_folder.replace("targets", "targets_24k")
    os.makedirs(target_tr_target_folder, exist_ok=True)
    tr_target_wav_lst = sorted(glob(os.path.join(orig_tr_target_folder, "*.wav")))
    for wavfile in tqdm(tr_target_wav_lst):
        wav, fs = read(wavfile)
        resampled_wav = librosa.resample(wav.T, orig_sr=fs, target_sr=TARGET_FS)
        write(
            wavfile.replace(orig_tr_target_folder, target_tr_target_folder),
            resampled_wav.T,
            TARGET_FS,
        )

    # resample dev/targets
    # orig_dev_target_folder = os.path.join(
    #     cfg.path.cec2_root, "clarity_CEC2_data/clarity_data/dev/targets/"
    # )
    # target_dev_target_folder = orig_dev_target_folder.replace("targets", "targets_24k")
    # os.makedirs(target_dev_target_folder, exist_ok=True)
    # dev_target_wav_lst = sorted(glob(os.path.join(orig_dev_target_folder, "*.wav")))
    # for wavfile in tqdm(dev_target_wav_lst):
    #     wav, fs = read(wavfile)
    #     resampled_wav = librosa.resample(wav.T, orig_sr=fs, target_sr=TARGET_FS)
    #     write(
    #         wavfile.replace(orig_dev_target_folder, target_dev_target_folder),
    #         resampled_wav.T,
    #         TARGET_FS,
    #     )

    print("Doing dev/speaker_adapt")
    # resample dev/speaker_adapt
    orig_dev_speaker_adapt_folder = os.path.join(
        cfg.path.cec3_task1, "clarity_data/dev/speaker_adapt/"
    )
    target_dev_speaker_adapt_folder = orig_dev_speaker_adapt_folder.replace(
        "speaker_adapt", "speaker_adapt_24k"
    )
    os.makedirs(target_dev_speaker_adapt_folder, exist_ok=True)
    dev_speaker_adapt_wav_lst = sorted(
        glob(os.path.join(orig_dev_speaker_adapt_folder, "*.wav"))
    )
    for wavfile in tqdm(dev_speaker_adapt_wav_lst):
        wav, fs = read(wavfile)
        resampled_wav = librosa.resample(wav.T, orig_sr=fs, target_sr=TARGET_FS)
        write(
            wavfile.replace(
                orig_dev_speaker_adapt_folder, target_dev_speaker_adapt_folder
            ),
            resampled_wav.T,
            TARGET_FS,
        )


if __name__ == "__main__":
    run_resample()


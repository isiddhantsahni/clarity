import json
import os
import hydra
import random
import numpy as np
from glob import glob

import torch
from soundfile import read


class CEC3Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scenes_folder,
        scenes_file,
        auxiliary_folder,
        sample_rate,
        wav_sample_len=None,
        auxiliary_sample_len=2,
        num_channels=6,
        norm=False,
        testing=False,
        dev=False,
    ):
        self.scenes_folder = scenes_folder
        self.auxiliary_folder = auxiliary_folder  # for spk embedding extraction
        self.sr = sample_rate
        self.wav_sample_len = wav_sample_len
        self.auxiliary_sample_len = auxiliary_sample_len
        self.num_channels = num_channels
        self.norm = norm
        self.testing = testing
        self.dev = dev

        self.scene_list = []
        with open(scenes_file, "r") as f:
            self.scene_dict = json.load(f)
            if not testing:
                for i in range(len(self.scene_dict)):
                    self.scene_list.append(self.scene_dict[i]["scene"])
            else:
                for i in range(len(self.scene_dict)):
                    self.scene_list.append(self.scene_dict[i]["scene"])
            f.close()

        auxiliary_list = sorted(glob(os.path.join(self.auxiliary_folder, "*.wav")))
        self.auxiliary_dict = {}
        for auxiliary_file in auxiliary_list:
            spk = os.path.basename(auxiliary_file).split("_")[0]
            spk_candidates = self.auxiliary_dict.get(spk, [])
            spk_candidates.append(os.path.basename(auxiliary_file))
            self.auxiliary_dict[spk] = spk_candidates

        if self.num_channels == 6:
            self.mix_suffix = ["_mix_CH1.wav", "_mix_CH2.wav", "_mix_CH3.wav"]
            if not self.dev:
                self.target_suffix = "_target_CH1.wav"
            else:
                self.target_suffix = "_reference.wav"
        else:
            raise NotImplementedError

    def read_wavfile(self, path):
        wav, wav_sr = read(path)
        assert wav_sr == self.sr
        return wav.transpose()

    def wav_sample(self, x, y):
        wav_len = x.shape[1]
        sample_len = self.wav_sample_len * self.sr
        if wav_len > sample_len:
            start = np.random.randint(wav_len - sample_len)
            end = start + sample_len
            x = x[:, start:end]
            y = y[:, start:end]
            return x, y
        elif wav_len < sample_len:
            x = np.append(
                x, np.zeros([x.shape[1], sample_len - wav_len], dtype=np.float32)
            )
            y = np.append(
                y, np.zeros([x.shape[1], sample_len - wav_len], dtype=np.float32)
            )
            return x, y
        else:
            return x, y

    def auxiliary_sample(self, wav):
        wav_len = wav.shape[0]
        sample_len = self.auxiliary_sample_len * self.sr
        if wav_len > sample_len:
            start = np.random.randint(wav_len - sample_len)
            end = start + sample_len
            return wav[start:end]
        elif wav_len < sample_len:
            wav = np.append(wav, np.zeros(sample_len - wav_len, dtype=np.float32))
            return wav
        else:
            return wav

    def __getitem__(self, item):
        scene_path = os.path.join(self.scenes_folder, self.scene_list[item])
        target_spk = self.scene_dict[item]["target"]["name"].split("_")[0]
        auxiliary_path = os.path.join(
            self.auxiliary_folder, random.choice(self.auxiliary_dict[target_spk])
        )
        mix = []
        for suffix in self.mix_suffix:
            mix.append(self.read_wavfile(scene_path + suffix))
        mix = np.concatenate(mix, axis=0)

        if not self.testing:
            target = self.read_wavfile(scene_path + self.target_suffix)

        auxiliary = self.read_wavfile(auxiliary_path)

        if self.wav_sample_len is not None:
            mix, target = self.wav_sample(mix, target)

        if self.auxiliary_sample_len is not None:
            auxiliary = self.auxiliary_sample(auxiliary)

        if self.norm:
            mix_max = np.max(np.abs(mix))
            mix = mix / mix_max
            target = target / mix_max

        if not self.testing:
            #print("****************")
            #print("Returning")
            return (
                torch.tensor(mix, dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32),
                torch.tensor(auxiliary, dtype=torch.float32),
                self.scene_list[item],
            )
        else:
            return (
                torch.tensor(mix, dtype=torch.float32),
                torch.tensor(auxiliary, dtype=torch.float32),
                torch.tensor(mix.shape[-1]),
                self.scene_list[item],
            )

    def __len__(self):
        return len(self.scene_list)


class CEC3DenoisedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scenes_folder,
        denoised_folder,
        scenes_file,
        sr,
        wav_sample_len=None,
        testing=False,
    ):
        self.scenes_folder = scenes_folder
        self.denoised_folder = denoised_folder
        self.sr = sr
        self.wav_sample_len = wav_sample_len
        self.testing = testing

        self.scene_list = []
        with open(scenes_file, "r") as f:
            self.scene_dict = json.load(f)
            if not testing:
                for i in range(len(self.scene_dict)):
                    self.scene_list.append(self.scene_dict[i]["scene"])
            else:
                for i in range(len(self.scene_dict)):
                    self.scene_list.append(self.scene_dict[i]["scene"])
            f.close()

    def read_wavfile(self, path):
        wav, wav_sr = read(path)
        assert wav_sr == self.sr
        return wav.transpose()

    def wav_sample(self, x, y):
        wav_len = x.shape[1]
        sample_len = self.wav_sample_len * self.sr
        if wav_len > sample_len:
            start = np.random.randint(wav_len - sample_len)
            end = start + sample_len
            x = x[:, start:end]
            y = y[:, start:end]
            return x, y
        elif wav_len < sample_len:
            x = np.append(
                x, np.zeros([x.shape[1], sample_len - wav_len], dtype=np.float32)
            )
            y = np.append(
                y, np.zeros([x.shape[1], sample_len - wav_len], dtype=np.float32)
            )
            return x, y
        else:
            return x, y

    def __getitem__(self, item):
        scene_path = os.path.join(self.scenes_folder, self.scene_list[item])
        denoised = self.read_wavfile(
            os.path.join(self.denoised_folder, self.scene_list[item] + "_den.wav")
        )

        if not self.testing:
            target = self.read_wavfile(scene_path + "_target_CH1.wav")

        if self.wav_sample_len is not None:
            denoised, target = self.wav_sample(denoised, target)

        if not self.testing:
            return (
                torch.tensor(denoised, dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32),
                self.scene_list[item],
            )
        else:
            return (
                torch.tensor(denoised, dtype=torch.float32),
                torch.tensor(denoised.shape[-1]),
                self.scene_list[item],
            )

    def __len__(self):
        return len(self.scene_list)


@hydra.main(config_path="../../", config_name="config_den")
def run(cfg):
    from torch.utils.data import DataLoader

    train_set = CEC3Dataset(**cfg.dev_dataset)
    train_loader = DataLoader(dataset=train_set, **cfg.dev_loader)

    for step, item in enumerate(train_loader):
        print(step)

if __name__ == "__main__":
    run()


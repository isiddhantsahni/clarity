import os
import torch
import torchaudio
from torch_haspi import DHASPIModel, DHASPI_NN


class DHASPILevelLoss(torch.nn.Module):
    def __init__(
        self,
        sr=24000,
        alpha=1e-4,
        dhaspi_kernel_size=513,
        dhaspi_level1=100,
        block_size=0.4,
        overlap=0.7,
        gamma_a=-70,
    ):
        super(DHASPILevelLoss, self).__init__()
        self.alpha = alpha
        self.dhaspi = DHASPIModel(kernel_size=dhaspi_kernel_size, Level1=dhaspi_level1)
        self.dhaspi_nn = DHASPI_NN()
        # load dhaspi_nn parameters, need to find a better way to do this
        dhaspi_nn_params = torch.load(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "dhaspi/DHASPI_NN.pkl"
            ),
            map_location="cpu",
        )
        self.dhaspi_nn.load_state_dict(dhaspi_nn_params)

        # rms measurement
        self.frame_size = int(block_size * sr)
        self.frame_shift = int(block_size * sr * (1 - overlap))
        self.unfold = torch.nn.Unfold(
            (1, self.frame_size), stride=(1, self.frame_shift)
        )
        self.gamma_a = gamma_a

    def measure_loudness(self, signal, eps=1e-8):
        x_unfold = self.unfold(signal.unsqueeze(1).unsqueeze(2))

        z = (
            torch.sum(x_unfold**2, dim=1) / self.frame_size
        )  # mean square for each frame
        el = -0.691 + 10 * torch.log10(z + eps)

        idx_a = torch.where(el > self.gamma_a, 1, 0)
        z_ave_gated_a = torch.sum(z * idx_a, dim=1, keepdim=True) / (
            torch.sum(idx_a, dim=1, keepdim=True) + eps
        )
        gamma_r = -0.691 + 10 * torch.log10(z_ave_gated_a + eps) - 10

        idx_r = torch.where(el > gamma_r, 1, 0)
        idx_a_r = idx_a * idx_r
        z_ave_gated_a_r = torch.sum(z * idx_a_r, dim=1, keepdim=True) / (
            torch.sum(idx_a_r, dim=1, keepdim=True) + eps
        )
        lufs = -0.691 + 10 * torch.log10(z_ave_gated_a_r + eps)  # loudness
        return lufs

    def forward(self, x, y, HL):
        # x: ref, y: proc
        feat, x_env, y_env = self.dhaspi(x, y, HL)
        DhaspiLoss = -self.dhaspi_nn(feat).mean()

        loudness_x = self.measure_loudness(
            x_env.view(x_env.shape[0] * x_env.shape[1], -1)
        )
        loudness_y = self.measure_loudness(
            y_env.view(x_env.shape[0] * x_env.shape[1], -1)
        )
        LevelLoss = (
            self.alpha
            * torch.maximum(loudness_y - loudness_x, torch.zeros_like(loudness_x)).sum()
        )
        return LevelLoss + DhaspiLoss, LevelLoss, DhaspiLoss


class CSMLoss(torch.nn.Module):
    def __init__(
        self,
        n_fft=512,
        win_length=512,
        hop_length=256,
        window_fn=torch.hann_window,
        power=None,
        est_form="wav",
    ):
        super(CSMLoss, self).__init__()
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window_fn=window_fn,
            power=power,
        )
        assert est_form in ["wav", "spec"]
        self.est_form = est_form  # est can be wav or spec, if wav, shape should be [bs, channel, wav_len],
        # if spec, shape should be [bs, channel, frame_len, 2]

    def forward(self, est, src):
        self.stft = self.stft.to(src.device)

        spec_src = self.stft(src)
        ri_spec_src = torch.view_as_real(
            spec_src
        )  # [bs, channel, nfft // 2 + 1, frames, 2]
        mag_spec_src = torch.abs(spec_src)  # [bs, channel, nfft // 2 + 1, frames]

        if self.est_form == "wav":
            spec_est = self.stft(est)
            ri_spec_est = torch.view_as_real(spec_est)
            mag_spec_est = torch.abs(spec_est)
            ri_loss = torch.abs(ri_spec_src - ri_spec_est).mean(dim=(0, 1, 3))
            mag_loss = torch.abs(mag_spec_src - mag_spec_est).mean(dim=(0, 1, 3))
            loss = (ri_loss.sum(dim=-1) + mag_loss).sum()
            return loss
        elif self.est_form == "spec":
            ri_loss = torch.abs(ri_spec_src - est).mean(dim=(0, 1, 3))
            mag_spec_est = torch.abs(torch.view_as_complex(est))
            mag_loss = torch.abs(mag_spec_src - mag_spec_est).mean(dim=(0, 1, 3))
            loss = (ri_loss.sum(dim=-1) + mag_loss).sum()
            return loss
        else:
            raise NotImplementedError


if __name__ == "__main__":
    from scipy.io.wavfile import read
    from clarity.evaluator.haspi.eb import Resamp24kHz
    from dhaspi.torch_haspi import interpolate_HL

    # dhaspi = DHASPIModel()
    # dhaspi.cuda()

    sr, wav = read("dhaspi/S06001_target_CH1.wav")
    wav = wav / 32768.0
    wav = Resamp24kHz(wav[:, 0], sr)[0]
    audiogram = [50, 50, 50, 50, 50, 50]
    wav_ref_torch = (
        (torch.tensor(wav, dtype=torch.float32).view(1, -1).repeat([2, 1]))
        .unsqueeze(0)
        .cuda()
    )
    wav_proc_torch = (
        (torch.tensor(wav, dtype=torch.float32).view(1, -1).repeat([2, 1]))
        .unsqueeze(0)
        .cuda()
    ) * 2

    # audiogram_torch = torch.tensor(
    #     interpolate_HL(audiogram), dtype=torch.float32
    # ).cuda()
    # loss = DHASPILevelLoss(alpha=1).cuda()
    # print(loss(wav_ref_torch, torch.tanh(wav_proc_torch), audiogram_torch))

    csmloss = CSMLoss()
    out = csmloss(wav_proc_torch, wav_ref_torch)
    print(out)


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def interpolate_aud(aud, nfir, sr):
    aud_fv = np.append(np.append(0, aud), sr // 2)  # Audiometric frequency vector
    linear_fv = (
        np.linspace(0, nfir, nfir + 1) / nfir * sr // 2
    )  # linear frequency vector
    interval_freq = np.zeros([len(linear_fv), 2])
    interval_idx = np.zeros([len(linear_fv), 2], dtype=int)
    for i, linear_fv_i in enumerate(linear_fv):
        for j in range(len(aud_fv) - 1):
            if aud_fv[j] <= linear_fv_i < aud_fv[j + 1]:
                interval_freq[i, 0] = aud_fv[j]
                interval_freq[i, 1] = aud_fv[j + 1]
                interval_idx[i, 0] = j
                interval_idx[i, 1] = j + 1
    interval_freq[-1, 0] = aud_fv[-2]
    interval_freq[-1, 1] = aud_fv[-1]
    interval_idx[-1, 0] = len(aud_fv) - 2
    interval_idx[-1, 1] = len(aud_fv) - 1
    x2_minus_x1 = interval_freq[:, 1] - interval_freq[:, 0]
    x_minus_x1 = linear_fv - interval_freq[:, 0]
    return x2_minus_x1, x_minus_x1, interval_idx


class Amplifier(nn.Module):
    def __init__(
        self, sr=24000, nfir=240, aud=None, n_layer=5, n_hidden=512, dropout=0.1
    ):
        super(Amplifier, self).__init__()

        self.window_size = nfir + 1
        self.padding = nfir // 2
        if aud is None:
            aud = np.array([250, 500, 1000, 2000, 4000, 6000])

        # audiometric interpolation parameters
        x2_minus_x1, x_minus_x1, self.interval_idx = interpolate_aud(
            aud, nfir=nfir, sr=sr
        )
        self.x2_minus_x1 = nn.Parameter(
            torch.tensor(x2_minus_x1, dtype=torch.float32), requires_grad=False
        )
        self.x_minus_x1 = nn.Parameter(
            torch.tensor(x_minus_x1, dtype=torch.float32), requires_grad=False
        )

        # network
        network = [nn.Linear(len(aud), n_hidden), nn.ReLU(), nn.Dropout(dropout)]
        for i in range(n_layer - 1):
            network.extend(
                [nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Dropout(dropout)]
            )
        network.append(nn.Linear(n_hidden, len(aud)))
        self.network = nn.Sequential(*network)

    def construct_fir(self, HL):
        amp = self.network(HL / 100.0) * 100.0
        amp = torch.pow(10, torch.abs(amp) / 20.0)
        amp = torch.cat((torch.cat((amp[:, :1], amp), dim=1), amp[:, -1:]), dim=1)
        y = amp[:, self.interval_idx]
        y2_minus_y1 = y[:, :, 1] - y[:, :, 0]
        y1 = y[:, :, 0]
        gain = y2_minus_y1 * self.x_minus_x1 / self.x2_minus_x1 + y1

        # firwin
        phase = torch.zeros_like(gain)
        gain = gain.unsqueeze(-1)
        phase = phase.unsqueeze(-1)
        magnitudes = torch.view_as_complex(torch.cat([gain, phase], dim=-1))
        impulse_response = torch.fft.irfft(magnitudes, dim=-1)

        window_size = self.window_size + 1
        window = torch.hann_window(window_size).to(gain.device)
        ir_size = int(impulse_response.shape[-1])
        half_idx = (window_size + 1) // 2
        padding = ir_size - window_size
        window = torch.cat(
            [
                window[half_idx:],
                torch.zeros([padding]).to(gain.device),
                window[:half_idx],
            ],
            dim=0,
        )
        impulse_response = impulse_response * window
        first_half_start = ir_size - (half_idx - 1)
        second_half_end = half_idx
        fir_filter = torch.cat(
            [
                impulse_response[:, first_half_start:],
                impulse_response[:, :second_half_end],
            ],
            dim=-1,
        )

        return fir_filter.unsqueeze(1)

    def forward(self, HL, wav):
        # input shape, HL: [N, n_HL], wav: [bs, wav_len]
        fir_filter = self.construct_fir(HL)  # [n, fir_len]
        out = F.conv1d(wav, fir_filter, padding=self.padding, bias=None)
        return out


if __name__ == "__main__":
    from scipy.io.wavfile import read
    from clarity.evaluator.haspi.eb import Resamp24kHz

    sr, wav = read("dhaspi/S06001_target_CH1.wav")
    wav = wav / 32768.0
    wav = Resamp24kHz(wav[:, 0], sr)[0]
    audiogram = [10, 20, 30, 40, 50, 60]
    wav_torch = (
        torch.tensor(wav, dtype=torch.float32).view(1, 1, -1).repeat([2, 1, 1]).cuda()
    )
    hl_torch = torch.tensor(audiogram, dtype=torch.float32).view(1, -1).cuda()
    print(wav_torch.shape, hl_torch.shape)
    amp = Amplifier().cuda()

    amp.eval()
    out = amp(hl_torch, wav_torch)[0]

    # fir = out.squeeze().cpu().detach().numpy()
    # import matplotlib.pyplot as plt
    # from scipy import signal
    #
    # # plt.plot(fir)
    # w, h = signal.freqz(fir, fs=24000)
    # plt.plot(w, 20 * np.log10(abs(h)), "b")
    # plt.show()


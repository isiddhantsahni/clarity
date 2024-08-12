import math
import numpy as np
import scipy
import scipy.signal

import torch
from torch import nn
import torch.nn.functional as F

from clarity.evaluator.haspi.eb import center_frequency, loss_parameters

def interpolate_HL(HL, nchan=32, aud=None):
    """
    Function to interpolate hearing losses
    Args:
        HL: hearing losses
        nchan: number of Gammatone filters
        aud: Audiometric frequencies in Hz

    Returns:
        interpolated hearing loss paramters
    """
    if aud is None:
        aud = np.array([250, 500, 1000, 2000, 4000, 6000])
    cfreq = center_frequency(nchan=nchan)
    # Interpolation to give the loss at the gammatone center frequencies. Use linear interpolation in dB. The
    # interpolation assumes that cfreq[0] < aud[0] and cfreq[-1] > aud[-1]
    fv = np.append(
        np.append(cfreq[0], aud), cfreq[-1]
    )  # Frequency vector for the interpolation
    HLv = np.append(np.append(HL[0], HL), HL[-1])  # HL vector for the interpolation
    interpf = scipy.interpolate.interp1d(fv, HLv)
    loss = interpf(cfreq)
    loss = np.clip(loss, 0, None)  # Make sure no negative losses
    return loss


class DHASPIModel(nn.Module):
    def __init__(
        self,
        nchan=32,
        audiometric_freq=None,
        kernel_size=513,
        sample_rate=24000,
        order=4,
        Level1=100,
        eps=1e-16,
    ):
        """
        Differentiable approximation to HASPI, no alignment, resample, HL interpolation
        Args:
            nchan: number of gammatone filters
            audiometric_freq: audiometric frequencies
        """
        super(DHASPIModel, self).__init__()
        self.nchan = nchan
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.order = order
        self.padding = (self.kernel_size - 1) // 2
        self.level1 = Level1
        self.eps = eps

        # Basic parameters
        cfreq = center_frequency(nchan)
        if audiometric_freq is None:
            audiometric_freq = [250, 500, 1000, 2000, 4000, 6000]
        earQ = 9.26449
        minBW = 24.7
        ERB = minBW + (cfreq / earQ)
        self.erb = nn.Parameter(
            torch.tensor(ERB, dtype=torch.float32), requires_grad=False
        )
        self.cfreq = nn.Parameter(
            torch.tensor(cfreq.copy(), dtype=torch.float32),
            requires_grad=False,
        )

        # Parameters for the reference normal hearing LossParameters
        HLx = nn.Parameter(torch.zeros(nchan, dtype=torch.float32), requires_grad=False)
        attnOHCx, BWminx, lowkneex, CRx, attnIHCx = self.LossParameters(HLx)
        self.attnOHCx = nn.Parameter(attnOHCx, requires_grad=False)
        self.BWminx = nn.Parameter(BWminx, requires_grad=False)
        self.lowkneex = nn.Parameter(lowkneex, requires_grad=False)
        self.CRx = nn.Parameter(CRx, requires_grad=False)
        self.attnIHCx = nn.Parameter(attnIHCx, requires_grad=False)

        # Parameters for MiddleEar
        bLP, aLP = scipy.signal.butter(1, 5000 / (0.5 * sample_rate))
        middleEar_lp_fir = scipy.signal.lfilter(bLP, aLP, [1] + [0] * (kernel_size - 1))
        bHP, aHP = scipy.signal.butter(2, 350 / (0.5 * sample_rate), "high")
        middleEar_hp_fir = scipy.signal.lfilter(bHP, aHP, [1] + [0] * (kernel_size - 1))

        self.middleEar_lp_fir = nn.Parameter(
            torch.tensor(middleEar_lp_fir, dtype=torch.float32).view(1, 1, -1),
            requires_grad=False,
        )
        self.middleEar_hp_fir = nn.Parameter(
            torch.tensor(middleEar_hp_fir, dtype=torch.float32).view(1, 1, -1),
            requires_grad=False,
        )
        self.middleEar_padding = kernel_size // 2

        # Parameters for gammatone filterbank
        n_lin = torch.linspace(0, kernel_size - 1, kernel_size)
        window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / kernel_size)
        self.n_ = nn.Parameter(
            torch.arange(0, kernel_size, dtype=torch.float32).view(1, -1) / sample_rate,
            requires_grad=False,
        )
        self.window_ = nn.Parameter(window_, requires_grad=False)
        # self.n_ = nn.Parameter(n_, requires_grad=False)

        center_hz = nn.Parameter(
            torch.tensor(cfreq.copy(), dtype=torch.float32).view(-1, 1),
            requires_grad=False,
        )
        f_times_t = torch.matmul(center_hz, self.n_)
        carrier_cos = torch.cos(2 * math.pi * f_times_t)
        carrier_sin = torch.sin(2 * math.pi * f_times_t)
        self.carrier_cos = nn.Parameter(carrier_cos, requires_grad=False)
        self.carrier_sin = nn.Parameter(carrier_sin, requires_grad=False)

        # Parameters for the control filter bank
        HLmax = [100, 100, 100, 100, 100, 100]
        # HLmax = [10, 10, 10, 10, 10, 10]
        _, BW1, _, _, _ = loss_parameters(HLmax, cfreq)
        self.bw1 = nn.Parameter(
            torch.tensor(BW1.copy(), dtype=torch.float32), requires_grad=False
        )
        gt_control_cos, gt_control_sin = self.construct_gammatone(
            self.bw1, self.carrier_cos, self.carrier_sin
        )
        self.gt_control_cos = nn.Parameter(gt_control_cos, requires_grad=False)
        self.gt_control_sin = nn.Parameter(gt_control_sin, requires_grad=False)

        # Parameters for EnvCompress
        bLP, aLP = scipy.signal.butter(1, 800 / (0.5 * sample_rate))
        compr_lp_fir = scipy.signal.lfilter(bLP, aLP, [1] + [0] * (kernel_size - 1))
        self.compr_lp_fir = nn.Parameter(
            torch.tensor(compr_lp_fir, dtype=torch.float32)
            .view(1, 1, -1)
            .repeat([self.nchan, 1, 1]),
            requires_grad=False,
        )
        self.compr_padding = kernel_size // 2

        # EnvFilt: Parameters for envelope subsampling
        fLP = 320
        fsub = math.floor(8 * fLP)
        self.subsample_space = math.floor(self.sample_rate / fsub)
        # Compute the lowpass filter length in samples to give -3 dB at fcut Hz
        tfilt = 1000 * (1 / fLP)  # filter length in ms
        tfilt = 0.7 * tfilt  # Empirical adjustment to the filter length
        nfilt = round(0.001 * tfilt * sample_rate)  # Filter length in samples
        nhalf = math.floor(nfilt / 2)
        nfilt = int(2 * nhalf)  # Force an even filter length
        # Design the FIR LP filter using a von Hann window to ensure that there are no negative envelope values
        # The MATLAB code uses the hanning() function, which returns the Hann window without the first and last zero-weighted window samples,
        # unlike np.hann and scipy.signal.windows.hann; the code below replicates this behaviour
        w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, nfilt / 2 + 1) / (nfilt + 1)))
        benv = np.concatenate((w, np.flip(w)))
        benv = np.expand_dims(benv / np.sum(benv), 0)
        self.benv = nn.Parameter(
            torch.tensor(np.ascontiguousarray(benv[::-1, ::-1]), dtype=torch.float32)
            .unsqueeze(0)
            .repeat([self.nchan, 1, 1]),
            requires_grad=False,
        )
        self.envfilt_padding = nhalf

        # CepCoef: Compute the ceptstral coefficients as a function of subsampled time
        nbasis = 6  # Use 6 basis functions
        self.thr = 2.5  # Silence threshold in dB SL
        dither = 0.1  # Dither in dB RMS to add to envelope signals
        # Mel cepstrum basis functions
        freq = np.arange(0, nbasis)
        k = np.arange(0, self.nchan)
        cepm = np.zeros((self.nchan, nbasis))
        for nb in range(nbasis):
            basis = np.cos(freq[nb] * np.pi * k / (self.nchan - 1))
            cepm[:, nb] = basis / np.sqrt(np.sum(basis**2))
        self.cepm = nn.Parameter(
            torch.tensor(cepm, dtype=torch.float32), requires_grad=False
        )
        self.nbasis = nbasis

        # ModFilt: Cepstral coeffifiencts filtered at each modulation rate
        # Band center frequencies [2, 6, 10, 16, 25, 40, 64, 100, 160, 256] Hz
        # Band edges [0, 4, 8, 12.5, 20.5, 30.5, 52.4, 78.1, 128, 200, 328] Hz
        # Modulation filter band cf and edges, 10 bands
        # Band spacing uses the factor of 1.6 used by Dau
        cf = np.array(
            [2, 6, 10, 16, 25, 40, 64, 100, 160, 256]
        )  # Band center frequencies
        nmod = len(cf)
        edge = np.zeros(nmod + 1)
        edge[0:3] = [0, 4, 8]
        for k in range(3, nmod + 1):
            edge[k] = (cf[k - 1] ** 2) / edge[k - 1]
        # Allowable filters based on envelope subsampling rate
        fNyq = 0.5 * fsub
        nmod = len(edge) - 1
        cf = cf[:nmod]
        # Assign FIR filter lengths. Setting t0=0.2 gives a filter Q of about 1.25 to match Ewert et al. (2002),
        # and t0=0.33 corresponds to Q=2 (Dau et al 1997a). Moritz et al. (2015) used t0=0.29. General relation Q=6.25*t0,
        # compromise with t0=0.24 which gives Q=1.5
        t0 = 0.24  # Filter length in seconds for the lowest modulation frequency band
        t = np.zeros(nmod)
        t[0] = t0
        t[1] = t0
        t[2:nmod] = t0 * cf[2] / cf[2:nmod]  # Constant-Q filters above 10 Hz
        nfir = 2 * np.floor(t * fsub / 2)  # Force even filter lengths in samples
        nhalf = nfir / 2
        # Design the family of lowpass windows
        b = []  # Filter coefficients, one filter impulse response per list element
        for k in range(nmod):
            bk = np.hanning(nfir[k] + 1)
            bk = bk / np.sum(bk)
            if len(bk) < max(nfir) + 1:
                bk = np.pad(bk, int(max(nfir) + 1 - len(bk)) // 2)
            b.append(bk)
        b = np.array(b)
        self.nmod = nmod
        self.modfilters = nn.Parameter(
            torch.tensor(b, dtype=torch.float32).repeat([nbasis, 1]).unsqueeze(1),
            requires_grad=False,
        )
        # Pre-compute the cosine and sine arrays
        modfilt_cf = cf.copy()
        modfilt_cf[0] = 0
        self.modfilt_cf = nn.Parameter(
            torch.tensor(modfilt_cf / fNyq, dtype=torch.float32), requires_grad=False
        )

    def LossParameters(self, HL):
        """
        Generating Loss parameters, see HASPI.eb LossParameters
        Args:
            HL: interpolated HL parameters
        Returns:
            attnOHC, BW, lowknee, CR, attnIHC
        """
        # Compression ratio changes linearly with ERB rate from 1.25:1 in the 80-Hz frequency band to 3.5:1 in the
        # 8-kHz frequency band
        CR = (1.25 + 2.25 * torch.arange(len(self.cfreq)) / (len(self.cfreq) - 1)).to(
            self.cfreq.device
        )

        # Maximum OHC sensitivity loss depends on the compression ratio.
        # The compression I/O curves assume linear below 30 and above 100 dB SPL in normal ears.
        maxOHC = 70 * (1 - (1 / CR))  # HC loss that results in 1:1 compression
        thrOHC = 1.25 * maxOHC  # Loss threshold for adjusting the OHC parameters

        # Apportion the loss in dB to the outer and inner hair cells based on the data of Moore et al (1999), JASA 106,
        # 2761-2778.
        # Reduce the CR towards 1:1 in proportion to the OHC loss.
        attnOHC = 0.8 * HL
        attnOHC_tmp = 0.8 * thrOHC
        attnOHC = torch.where(HL > thrOHC, attnOHC_tmp, attnOHC)

        attnIHC = 0.2 * HL
        attnIHC_tmp = 0.2 * thrOHC + (HL - thrOHC)
        attnIHC = torch.where(HL > thrOHC, attnIHC_tmp, attnIHC)

        # Adjust the OHC bandwidth in proportion to the OHC loss
        BW = 1 + (attnOHC / 50.0) + 2.0 * (attnOHC / 50.0) ** 6

        # Compute the compression lower kneepoint and compression ratio
        lowknee = attnOHC + 30
        upamp = 30 + 70 / CR  # Output level for an input of 100 dB SPL

        CR = (100 - lowknee) / (upamp + attnOHC - lowknee)  # OHC loss Compression ratio

        return attnOHC, BW, lowknee, CR, attnIHC

    def MiddleEar(self, x):
        """
        Function to simulate middle ear filters, including a butter HP filter at 350Hz, and a butter LP filter at 500Hz.
        """
        x = F.conv1d(x, self.middleEar_hp_fir, padding=self.middleEar_padding)
        x = F.conv1d(x, self.middleEar_lp_fir, padding=self.middleEar_padding)
        return x

    def construct_gammatone(self, bw, carrier_cos, carrier_sin, n_repeat=1):
        """
        Construct Gammatone FIR filters
        """
        bfreq = bw * self.erb.repeat(n_repeat) * 1.019
        band_hz = bfreq.view(-1, 1)
        b_times_t = torch.matmul(band_hz, self.n_)
        kernel = torch.pow(self.n_, self.order - 1) * torch.exp(
            -2 * math.pi * b_times_t
        )
        gt_fir_cos = kernel * carrier_cos
        gt_fir_sin = kernel * carrier_sin

        # gain = gt_fir_cos.max(dim=1)[0].unsqueeze(1)
        gain = torch.abs(torch.fft.fft(gt_fir_cos, dim=-1)).max(dim=-1)[0].view(-1, 1)
        gt_fir_cos = (gt_fir_cos / gain).view(
            self.nchan * n_repeat, 1, self.kernel_size
        )
        gt_fir_sin = (gt_fir_sin / gain).view(
            self.nchan * n_repeat, 1, self.kernel_size
        )
        return gt_fir_cos, gt_fir_sin

    def gammatone_conv(self, x, gt_cos, gt_sin):
        """
        Retrieve envelopes with Gammatone filterbank
        """
        x = x.repeat([1, self.nchan, 1])
        ureal = F.conv1d(x, gt_cos, padding=self.padding, bias=None, groups=self.nchan)
        uimag = F.conv1d(x, gt_sin, padding=self.padding, bias=None, groups=self.nchan)
        env = torch.sqrt(ureal * ureal + uimag * uimag + self.eps)
        return env

    def BWadjust(self, control, bwmin, bwmax, Level1):
        """
        Function to compute the increase in auditory filter bandwidth in response
        to high signal levels.
        """
        cRMS = torch.sqrt(torch.mean(control**2, dim=2) + self.eps)
        cdB = 20 * torch.log10(cRMS) + Level1
        BW = bwmin + ((cdB - 50) / 50) * (bwmax - bwmin)
        return torch.clamp(BW, min=bwmin, max=bwmax)

    def EnvCompress(self, env, control, attnOHC, thrLow, CR, level1):
        """
        Function to compute the cochlear compression in one auditory filter
        band. The gain is linear below the lower threshold, compressive with
        a compression ratio of CR:1 between the lower and upper thresholds,
        and reverts to linear above the upper threshold. The compressor
        assumes that auditory thresold is 0 dB SPL.
        """
        attnOHC = attnOHC.view(1, -1, 1)
        thrLow = thrLow.view(1, -1, 1)
        CR = CR.view(1, -1, 1)
        bs = env.shape[0]
        thrHigh = torch.ones_like(env) * 100.0
        logenv = 20 * torch.log10(torch.clamp(control, self.eps)) + level1
        logenv = torch.max(torch.min(logenv, thrHigh), thrLow.repeat([bs, 1, 1]))
        gain = -attnOHC - (logenv - thrLow) * (1 - (1 / CR))
        gain = torch.pow(10.0, gain / 20)
        gain = F.conv1d(
            gain,
            self.compr_lp_fir,
            padding=self.compr_padding,
            bias=None,
            groups=self.nchan,
        )
        return gain * env

    def EnvAlign(self, x, y):
        """
        Envelope alignment. Not implemented.
        """

    def EnvSL(self, env, attnIHC, level1):
        """
        Function to convert the compressed envelope returned by
        cochlea_envcomp to dB SL.
        """
        y = level1 - attnIHC.view(1, -1, 1) + 20 * torch.log10(env + self.eps)
        return torch.clamp(y, 0)

    def EnvFilt(self, x, y):
        """
        Function to lowpass filter and subsample the envelope in dB SL produced by the model of the auditory periphery.
        :param x: env in dB SL for the ref signal in each auditory band
        :param y: env in dB SL for the degraded signal in each auditory band
        """
        xenv = F.conv1d(
            x, self.benv, padding=self.envfilt_padding, bias=None, groups=self.nchan
        )[:, :, 1:]
        yenv = F.conv1d(
            y, self.benv, padding=self.envfilt_padding, bias=None, groups=self.nchan
        )[:, :, 1:]

        # Subsample the LP filtered envelopes
        index = torch.arange(0, xenv.shape[-1], self.subsample_space).to(x.device)
        xLP = xenv[:, :, index]
        yLP = yenv[:, :, index]
        return xLP, yLP

    def CepCoef(self, x, y):
        """
        Function to compute the cepstral correlation coefficients between the reference signal and the distorted signal
        log envelopes. The silence portions of the signals are removed prior to the calculation based on the envelope of
        the reference signal. For each time sample, the log spectrum in dB SL is fitted with a set of half-cosine basis
        functions. The cepstral coefficients then form the input to the cepstral correlation calculation.

        :param x: subsampled reference signal envelope in dB SL in each band
        :param y: subsampled distorted output signal envelope
        """
        # Find the reference segments that lie sufficiently above the quiescent rate
        xsum = torch.sum(10 ** (x / 20), dim=1) / self.nchan
        xsum = 20 * torch.log10(xsum + self.eps)

        # suppress the silent frames, different from the original implementation, as slicing can be difficult for batch training
        key_frames = torch.where(xsum > self.thr, 1, 0)
        xdB = x * key_frames.unsqueeze(1)
        ydB = y * key_frames.unsqueeze(1)

        xcep = torch.matmul(xdB.transpose(1, 2), self.cepm).transpose(1, 2)
        ycep = torch.matmul(ydB.transpose(1, 2), self.cepm).transpose(1, 2)
        xcep = xcep - xcep.mean(dim=2, keepdim=True)
        ycep = ycep - ycep.mean(dim=2, keepdim=True)
        return xcep, ycep

    def ModFilt(self, x, y):
        """
        Function to apply an FIR modulation filterbank to the reference envelope signals contained in matrix x and the
        processed signal envelope signals in matrix y. Each column (excluding batch size) in x and y is a separate
        filter band or cepstral coefficient basis function. The modulation filters use a lowpass filter for the lowest
        modulation rate, and complex demodulation followed by a lowpass filter for the remaining bands. The onset and
        offset transients are removed from the FIR convolutions to temporally align the modulation filter outputs.

        :param x: tensor containing the subsampled reference envelope values. The second order is a different frequency
        band or cepstral basis function arranged from low to high.
        :param y: tensor containing the subsampled processed envelope values
        """
        batch_size = x.shape[0]
        n = torch.arange(1, x.shape[-1] + 1, dtype=torch.float32).to(x.device)
        cos_mod = math.sqrt(2) * torch.cos(
            math.pi * n.unsqueeze(0) * self.modfilt_cf.unsqueeze(1)
        )
        cos_mod[0, :] = cos_mod[0, :] / math.sqrt(2)
        cos_mod = cos_mod.repeat([x.shape[1], 1])

        sin_mod = math.sqrt(2) * torch.sin(
            math.pi * n.unsqueeze(0) * self.modfilt_cf.unsqueeze(1)
        )
        sin_mod[0, :] = sin_mod[0, :] / math.sqrt(2)
        sin_mod = sin_mod.repeat([x.shape[1], 1])

        x_repeat = torch.repeat_interleave(x, self.modfilt_cf.shape[0], 1)
        y_repeat = torch.repeat_interleave(y, self.modfilt_cf.shape[0], 1)

        xmod_real = F.conv1d(
            x_repeat * cos_mod,
            self.modfilters,
            padding=self.modfilters.shape[-1] // 2,
            bias=None,
            groups=self.modfilters.shape[0],
        )
        xmod_imag = F.conv1d(
            x_repeat * sin_mod,
            self.modfilters,
            padding=self.modfilters.shape[-1] // 2,
            bias=None,
            groups=self.modfilters.shape[0],
        )
        xmod = xmod_real * cos_mod + xmod_imag * sin_mod

        ymod_real = F.conv1d(
            y_repeat * cos_mod,
            self.modfilters,
            padding=self.modfilters.shape[-1] // 2,
            bias=None,
            groups=self.modfilters.shape[0],
        )
        ymod_imag = F.conv1d(
            y_repeat * sin_mod,
            self.modfilters,
            padding=self.modfilters.shape[-1] // 2,
            bias=None,
            groups=self.modfilters.shape[0],
        )
        ymod = ymod_real * cos_mod + ymod_imag * sin_mod
        return xmod.view(batch_size, self.nbasis, self.nmod, -1), ymod.view(
            batch_size, self.nbasis, self.nmod, -1
        )

    def ModCorr(self, x, y):
        """
        Function to compute the cross-correlations between the input signal time-frequency envelope and the distortion
        time-frequency envelope. The cepstral coefficients or envelopes in each frequency band have been passed through
        the modulation filterbank using function ModFilt.

        :param x: tensor containing the reference signal output of the modulation filterbank. The size is [batch_size,
        nchan, nmodfilter, length]
        :param y: subsampled distorted output signal envelope
        """
        xnorm = x - x.mean(dim=-1, keepdim=True)
        xsum = torch.sum(xnorm**2, dim=-1)
        ynorm = y - y.mean(dim=-1, keepdim=True)
        ysum = torch.sum(ynorm**2, dim=-1)
        CM = torch.abs(torch.sum(xnorm * ynorm, dim=-1)) / torch.sqrt(
            xsum * ysum + self.eps
        )
        return torch.mean(CM[:, 1:, :], dim=1, keepdim=False)

    def forward(self, x, y, HL):
        """
        Args:
            x: reference signal, shape [batch_size, length]
            y: processed signal, shape [batch_size, length]
            HL: interpolated HL
        """
        batch_size = y.shape[0]
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        attnOHCy, BWminy, lowkneey, CRy, attnIHCy = self.LossParameters(HL)

        # Middle ear processing
        x_mid = self.MiddleEar(x)
        y_mid = self.MiddleEar(y)

        # control signals
        x_control = self.gammatone_conv(x_mid, self.gt_control_cos, self.gt_control_sin)
        y_control = self.gammatone_conv(y_mid, self.gt_control_cos, self.gt_control_sin)

        # band width adjustment
        BWx = self.BWadjust(
            x_control, self.BWminx, self.bw1, self.level1
        )  # BW shape [bs, nchan]
        BWy = self.BWadjust(y_control, BWminy, self.bw1, self.level1)

        # construct gammatone filterbanks
        gt_fir_cos_x, gt_fir_sin_x = self.construct_gammatone(
            BWx.view(batch_size * self.nchan),
            self.carrier_cos.repeat([batch_size, 1]),
            self.carrier_sin.repeat([batch_size, 1]),
            batch_size,
        )
        gt_fir_cos_y, gt_fir_sin_y = self.construct_gammatone(
            BWy.view(batch_size * self.nchan),
            self.carrier_cos.repeat([batch_size, 1]),
            self.carrier_sin.repeat([batch_size, 1]),
            batch_size,
        )

        # Get the envelopes
        # NEED A CLEVERER WAY TO DO THIS!
        x_env = []
        for i in range(batch_size):
            x_env.append(
                self.gammatone_conv(
                    x_mid[i].unsqueeze(0),
                    gt_fir_cos_x[self.nchan * i : self.nchan * (i + 1)],
                    gt_fir_sin_x[self.nchan * i : self.nchan * (i + 1)],
                )
            )
        x_env = torch.cat(x_env, dim=0)
        y_env = []
        for i in range(batch_size):
            y_env.append(
                self.gammatone_conv(
                    y_mid[i].unsqueeze(0),
                    gt_fir_cos_y[self.nchan * i : self.nchan * (i + 1)],
                    gt_fir_sin_y[self.nchan * i : self.nchan * (i + 1)],
                )
            )
        y_env = torch.cat(y_env, dim=0)

        # Compress envelopes
        x_compr = self.EnvCompress(
            x_env, x_control, self.attnOHCx, self.lowkneex, self.CRx, self.level1
        )
        y_compr = self.EnvCompress(
            y_env, y_control, attnOHCy, lowkneey, CRy, self.level1
        )

        # Convert to dB
        x_dB = self.EnvSL(x_compr, self.attnIHCx, self.level1)
        y_dB = self.EnvSL(y_compr, attnIHCy, self.level1)

        # Feature extraction
        # EnvFilt
        x_LP, y_LP = self.EnvFilt(x_dB, y_dB)

        # CepCoef
        x_cep, y_cep = self.CepCoef(x_LP, y_LP)

        # ModFilt
        x_mod, y_mod = self.ModFilt(x_cep, y_cep)

        # ModCorr
        aveCM = self.ModCorr(x_mod, y_mod)

        return aveCM, x_dB, y_dB


class DHASPI_NN(nn.Module):
    def __init__(self):
        super(DHASPI_NN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, feat):
        return self.network(feat)


if __name__ == "__main__":
    from scipy.io.wavfile import read
    from clarity.evaluator.haspi.eb import Resamp24kHz, InputAlign
    from clarity.evaluator.haspi import haspi_v2
    import time

    dhaspi = DHASPIModel()
    dhaspi.cuda()

    sr, wav = read("S06001_target_CH1.wav")
    wav = wav / 32768.0
    wav = Resamp24kHz(wav[:, 0], sr)[0]
    audiogram = [50, 50, 50, 50, 50, 50]
    wav_ref_torch = (
        torch.tensor(wav, dtype=torch.float32).view(1, -1).repeat([1, 1])
    ).cuda()
    wav_proc_torch = (
        torch.tensor(wav, dtype=torch.float32).view(1, -1).repeat([1, 1])
    ).cuda()
    audiogram_torch = torch.tensor(
        interpolate_HL(audiogram), dtype=torch.float32
    ).cuda()

    time1 = time.time()
    aveCM = haspi_v2(wav, 24000, wav, 24000, audiogram, Level1=100)
    time2 = time.time()
    daveCM = dhaspi(wav_ref_torch, wav_proc_torch, audiogram_torch)
    time3 = time.time()

    print(aveCM, daveCM)
    print(f"dhasp time: {time3 - time2}s,  haspi time: {time2 - time1}")


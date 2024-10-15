#!/usr/bin/env python3
"""
Created on 17:36, Nov. 7th, 2023

@author: Norbert Zheng
"""
import numpy as np
import scipy as sp
from functools import partial
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))
    from superlet import superlet_sp as _superlet
else:
    from .superlet import superlet_sp as _superlet

__all__ = [
    "stft",
    "superlet",
]

"""
wav2spec funcs
"""
# def stft func
def stft(X, sfreq, norm_type="zscore", n_sfreqs=40, nperseg=400, noverlap=350, return_onesided=True):
    """
    Use `STFT` to convert raw time series into spectrum.

    Args:
        X: (*, seq_len) - The raw time series.
        sfreq: float - The sampling frequency of the `X`.
        norm_type: str - The type of data normalization.
        n_sfreqs: int - The threshold of # of sampling frequency components.
        nperseg: int - The length of each segment.
        noverlap: int - The number of points to overlap between segments.
        return_onesided: bool - The flag that indicates whether return a one-sided spectrum for real data.

    Returns:
        spec: (*, len(f), len(t)) - The transformed spectrum.
    """
    # Compute the Short Time Fourier Transform (STFT). STFTs can be used as a way of quantifying the change
    # of a nonstationary signal's frequency and phase content over time. The arguments are as follows:
    #  1) x: array_like - Time series of measurement values.
    #  2) fs: float - Sampling frequency of `x` time series, defaults to `1.0`.
    #  3) window: str or tuple or array_like - Desired window to use. If `window` is a string or tuple, it is
    #     passed to `get_window` to generate the window values, which are DFT-even by default. See `get_window`
    #     for a list of windows and required parameters. If `window` is array_like it will be used directly
    #     as the window and its length must be nperseg. Defaults to a Hann window.
    #  4) nperseg: int - The length of each segment. Defaults to 256.
    #  5) noverlap: int - The number of points to overlap between segments. If `None`, `noverlap = nperseg // 2`.
    #     Defaults to `None`. When specified, the COLA constraint must be met (see Notes below).
    #  6) nfft: int - The length of the FFT used, if a zero padded FFT is desired.
    #     If `None`, the FFT length is `nperseg`. Defaults to `None`.
    #  7) detrend: str or function or `False` - Specifies how to detrend each segment. If `detrend` is a string,
    #     it is passed as the `type` argument to the `detrend` function. If it is a function, it takes a segment
    #     and returns a detrended segment. If `detrend` is `False`, no detrending is done. Defaults to `False`.
    #  8) return_onesided: bool - If `True`, return a one-sided spectrum for real data. If `False` return a two-sided
    #     spectrum. Defaults to `True`, but for complex data, a two-sided spectrum is always returned.
    #  9) boundary: str or None - Specifies whether the input signal is extended at both ends, and how to generate
    #     the new values, in order to center the first windowed segment on the first input point. This has the benefit
    #     of enabling reconstruction of the first input point when the employed window function starts at zero. Valid
    #     options are `["even", "odd", "constant", "zeros", None]`. Defaults to "zeros", for zero padding extension.
    #     I.e. `[1, 2, 3, 4]` is extended to `[0, 1, 2, 3, 4, 0]` for `nperseg=3`.
    # 10) padded: bool - Specifies whether the input signal is zero-padded at the end to make the signal fit exactly
    #     into an integer number of window segments, so that all of the signal is included in the output. Defaults to `True`.
    #     Padding occurs after boundary extension, if `boundary` is not `None`, and `padded` is `True`, as is the default.
    # 11) axis: int - Axis along which the STFT is computed; the default is over the last axis (i.e., `axis=-1`).
    # 12) scaling: {"spectrum", "psd"} - The default 'spectrum' scaling allows each frequency line of `Zxx` to
    #     be interpreted as a magnitude spectrum. The 'psd' option scales each line to a power spectral density
    #     - it allows to calculate the signal's energy by numerically integrating over `abs(Zxx)**2`.
    # f - ((nperseg // 2) + 1,), min 0, max sfreq.
    # t - (ceil(seq_len / (nperseg - noverlap)) + 1), min 0, max (seq_len + x).
    # Zxx - (*, len(f), len(t))
    f, t, Zxx = sp.signal.stft(
        # Modified `sp.signal.stft` parameters.
        X, fs=sfreq, nperseg=nperseg, noverlap=noverlap, return_onesided=return_onesided,
        # Default `sp.signal.stft` parameters.
        window="hann", detrend=False, boundary="zeros", padded=True, axis=-1, scaling="spectrum"
    )
    # Truncate `f` & `Zxx` according to `n_sfreqs`.
    if return_onesided:
        # f - (n_sfreqs,); Zxx - (*, n_sfreqs, len(t))
        f = f[:n_sfreqs]; Zxx = Zxx[...,:n_sfreqs,:]
    # Get the absolute amplitude of `Zxx`.
    Zxx = np.abs(Zxx)
    # Normalize `Zxx` according to `norm_type`.
    if norm_type == "zscore":
        Zxx = zscore(Zxx, axis=-1)
    elif norm_type == "db":
        Zxx = np.log(Zxx)
    else:
        raise ValueError("ERROR: Get unknown normalization type {} in brainbert.wav2spec.")
    # Make sure `Zxx` satisfies certain constraints.
    assert not (np.std(Zxx) == 0.).any()
    assert not np.isnan(Zxx).any()
    # Return the final `spec`.
    return Zxx

# def superlet func
def superlet(X, sfreq, cycle_order=(2, 12), cycle_base=3, foi=None, decim=1):
    """
    Use `superlet` to convert raw time series into spectrum.

    Args:
        X: (*, seq_len) - The raw time series.
        sfreq: float - The sampling frequency of the `X`.
        cycle_order: tuple - The order of the superlet set, containing [order_min, order_max]. `order_min`
            controls the minimal number of cycles within a SL together with `cycle_base` parameter:
            `cycle_min = cycle_base * order_min`. `order_max` controls the minimal number of cycles
            within a SL together with the `cycle_base` parameter: `cycle_max = cycle_base * order_min`.
            Note that for admissability reasons `cycle_min` should be at least 3!
        cycle_base: int - The number of cycles of the base Morlet wavelet. If set to lower
            than 3 increase `order_min` as to never have less than 3 cycles in a wavelet!
        foi: np.array - The frequencies of interest used to generate `scales`.

    Returns:
        spec: (*, len(f), len(t)) - The transformed spectrum.
    """
    # Initialize `foi`, if is `None`.
    # foi - (n_freqs,)
    foi = np.linspace(5, 200, 50) if foi is None else foi
    # Calculate `scales` according to `foi`.
    # scales - (n_freqs,)
    scales = (1. / foi) / (2. * np.pi)
    # Compute superlet spectrum.
    # spec - (*, n_freqs, seq_len)
    spec = _superlet(data=X, sfreq=sfreq, scales=scales, cycle_order=cycle_order, cycle_base=cycle_base, use_adaptive=True)
    # Get the absolute amplitude of `spec`.
    spec = np.abs(spec)
    # Get downsampled `spec` according to `decim`.
    # spec - (*, n_freqs, seq_len // decim)
    spec = spec[...,::decim]
    # Z-score normalize `spec`.
    spec = sp.stats.zscore(spec, axis=-1)
    # Make sure `spec` satisfies certain constraints.
    assert not (np.std(spec) == 0.).any()
    assert not np.isnan(spec).any()
    # Return the final `spec`.
    return spec

"""
tool funcs
"""
# def zscore func
def zscore(X, axis=-1):
    """
    Perform z-score normalization.

    Args:
        X: (n_samples, *, seq_len) - The original data.
        axis: int - The axis to execute z-score normalization.

    Returns:
        Z: (n_samples, *, seq_len) - The z-score transformed data.
    """
    return (X - np.mean(X, axis=axis, keepdims=True)) / np.std(X, axis=axis, keepdims=True)

if __name__ == "__main__":
    import time

    # Initialize macros.
    n_samples = 32; sfreq = 2048; seq_len = sfreq * 5

    # Initialize raw time series.
    # X - (n_samples, seq_len)
    X = np.random.uniform(low=0., high=1., size=(n_samples, seq_len)).astype(np.float32)
    # Forward stft function.
    time_start = time.time()
    Z = stft(X, sfreq, norm_type="zscore", n_sfreqs=40, nperseg=400, noverlap=350, return_onesided=True)
    print("INFO: The total time of stft for {:d} samples is {:.2f}s.".format(n_samples, time.time()-time_start))
    # Forward superlet function.
    time_start = time.time()
    Z = superlet(X, sfreq, cycle_order=(2, 12), cycle_base=3, foi=None, decim=1)
    print("INFO: The total time of superlet for {:d} samples is {:.2f}s.".format(n_samples, time.time()-time_start))


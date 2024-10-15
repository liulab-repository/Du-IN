#!/usr/bin/env python3
"""
Created on 21:00, Nov. 8th, 2023

@author: Norbert Zheng
"""
import numpy as np
from functools import partial
from scipy.signal import fftconvolve
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))

__all__ = [
    "superlet",
]

"""
core funcs
"""
# def superlet func
def superlet(data, sfreq, scales, cycle_order, cycle_base=3, use_adaptive=True):
    """
    Time-frequency analysis with superlets. Based on [1]. Implementation by Gregor Mönke: github.com/tensionhead.

    Performs Superlet Transform (SLT) according to Moca et al. [1]. Both `multiplicative SLT` and
    `fractional adaptive SLT` are available. The former is recommended for a narrow frequency band
    of interest, whereas the latter is better suited for the analysis of a broad range of frequencies.

    A superlet (SL) is a set of Morlet wavelets with increasing number  of cycles within the Gaussian envelope.
    Hence the bandwith is constrained more and more with more cycles yielding a sharper frequency resolution.
    Complementary the low cycle numbers will give a high time resolution. The SLT then is the geometric mean
    of the set of individual wavelet transforms, combining both wide and narrow-bandwidth wavelets into
    a super-resolution estimate.

    Args:
        data: (*, seq_len) - The raw time series. The last dimension is interpreted as the time axis.
        sfreq: float - The sample rate of `data` in Hz.
        scales: (n_freqs,) - The set of scales to use in wavelet transform. We should note that for
            the SL Morlet the relationship between scale and frequency simply is `s(f) = 1/(2*pi*f)`.
            Need to be ordered from high to low for `adaptive=True`.
        cycle_order: tuple - The order of the superlet set, containing [order_min, order_max]. `order_min`
            controls the minimal number of cycles within a SL together with `cycle_base` parameter:
            `cycle_min = cycle_base * order_min`. `order_max` controls the minimal number of cycles
            within a SL together with the `cycle_base` parameter: `cycle_max = cycle_base * order_min`.
            Note that for admissability reasons `cycle_min` should be at least 3!
        cycle_base: int - The number of cycles of the base Morlet wavelet. If set to lower
            than 3 increase `order_min` as to never have less than 3 cycles in a wavelet!
        use_adaptive: bool - The flag that indicates whether use adaptive superlet.

    Returns:
        spec: (*, n_freqs, seq_len) - The complex time-frequency representation of the input data.

    Notes:
        [1] Moca, Vasile V., et al. "Time-frequency super-resolution with superlets."
            Nature communications 12.1 (2021): 1-18.
    """
    # Multiplicative SLT.
    if not use_adaptive:
        spec = MSLT(data=data, sfreq=sfreq, scales=scales, cycle_order=cycle_order, cycle_base=cycle_base)
    # Fractional Adaptive SLT.
    else:
        spec = FASLT(data=data, sfreq=sfreq, scales=scales, cycle_order=cycle_order, cycle_base=cycle_base)
    # Return the final `spec`.
    return spec

# def MSLT func
def MSLT(data, sfreq, scales, cycle_order, cycle_base=3):
    """
    Multiplicative SL transform.

    Args:
        data: (*, seq_len) - The raw time series. The last dimension is interpreted as the time axis.
        sfreq: float - The sample rate of `data` in Hz.
        scales: (n_freqs,) - The set of scales to use in wavelet transform. We should note that for
            the SL Morlet the relationship between scale and frequency simply is `s(f) = 1/(2*pi*f)`.
            Need to be ordered from high to low for `adaptive=True`.
        cycle_order: tuple - The order of the superlet set, containing [order_min, order_max]. `order_min`
            controls the minimal number of cycles within a SL together with `cycle_base` parameter:
            `cycle_min = cycle_base * order_min`. `order_max` controls the minimal number of cycles
            within a SL together with the `cycle_base` parameter: `cycle_max = cycle_base * order_min`.
            Note that for admissability reasons `cycle_min` should be at least 3!
        cycle_base : int - The number of cycles of the base Morlet wavelet. If set to lower
            than 3 increase `order_min` as to never have less than 3 cycles in a wavelet!

    Returns:
        spec: (*, n_freqs, seq_len) - The complex time-frequency representation of the input data.
    """
    # Initialize `dt` from `sfreq`.
    dt = 1 / sfreq
    # Create the complete multiplicative set spanning, from `order_min` to `order_max`.
    # n_cycles - (n_freqs,)
    n_cycles = cycle_base * np.arange(cycle_order[0], cycle_order[1] + 1)
    # Initialize `spec` according to the lowest order, then update `spec` according to SL with higher orders.
    # spec - (*, seq_len, n_freqs)
    spec = np.prod([np.power(
        cwtSL(data=data, dt=dt, scales=scales, n_cycles=n_cycles_i), 1. / len(n_cycles)
    ) for n_cycles_i in n_cycles], axis=0)
    # Transpose `spec` to get general formulation.
    # spec - (*, n_freqs, seq_len)
    spec = np.moveaxis(spec, -1, -2)
    # Return the final `spec`.
    return spec

# def FASLT func
def FASLT(data, sfreq, scales, cycle_order, cycle_base=3):
    """
    Fractional adaptive SL transform. For non-integer orders fractional SLTs are calculated in the interval
    [order, order+1) via: `R(o_f) = R_1 * R_2 * ... * R_i * R_i+1 ** alpha` with `o_f = o_i + alpha`.

    Args:
        data: (*, seq_len) - The raw time series. The last dimension is interpreted as the time axis.
        sfreq: float - The sample rate of `data` in Hz.
        scales: (n_freqs,) - The set of scales to use in wavelet transform. We should note that for
            the SL Morlet the relationship between scale and frequency simply is `s(f) = 1/(2*pi*f)`.
            Need to be ordered from high to low for `adaptive=True`.
        cycle_order: tuple - The order of the superlet set, containing [order_min, order_max]. `order_min`
            controls the minimal number of cycles within a SL together with `cycle_base` parameter:
            `cycle_min = cycle_base * order_min`. `order_max` controls the minimal number of cycles
            within a SL together with the `cycle_base` parameter: `cycle_max = cycle_base * order_min`.
            Note that for admissability reasons `cycle_min` should be at least 3!
        cycle_base : int - The number of cycles of the base Morlet wavelet. If set to lower
            than 3 increase `order_min` as to never have less than 3 cycles in a wavelet!

    Returns:
        spec: (*, n_freqs, seq_len) - The complex time-frequency representation of the input data.
    """
    # Initialize `dt` from `sfreq`.
    dt = 1 / sfreq
    # Initialize frequencies of interest from the scales for the SL Morlet.
    # fois - (n_freqs,); orders - (n_freqs,)
    fois = 1 / (2 * np.pi * scales); foi_min, foi_max = fois[0], fois[-1]; assert foi_min < foi_max
    orders = cycle_order[0] + ((cycle_order[1] - cycle_order[0]) * (fois - foi_min) / (foi_max - foi_min))
    # Every scale needs a different exponent for the geometric mean.
    # exps - (n_freqs,)
    exps = 1 / (orders - cycle_order[0] + 1)
    # Each frequency/scale will have its own multiplicative SL,
    # which overlap -> higher orders have all the lower orders.
    # Calculate the fractions.
    alphas = orders % np.int32(np.floor(orders)); orders = np.int32(np.floor(orders))
    # Create the complete superlet set from all enclosed integer orders.
    # n_cycles - (n_freqs,)
    n_cycles = cycle_base * np.unique(orders)
    # Find which frequencies/scales use the same integer orders SL.
    order_jumps = np.where(np.diff(orders) > 0)[0]
    # Initialize `spec` according to the lowest order.
    # The lowest order is needed for all scales/frequencies.
    # Geometric normalization according to scale dependent order.
    # spec - (*, seq_len, n_freqs)
    spec = np.power(cwtSL(data=data, dt=dt, scales=scales, n_cycles=n_cycles[0]), exps)
    # Update `spec` according to SL with higher orders. We go to the next scale and order
    # in any case, but for `order_max==1` for which `order_jumps` is empty.
    last_jump = 1
    for i, jump in enumerate(order_jumps):
        # Get relevant scales for the next order, then get next (order + 1) spec.
        # spec_nxt - (*, seq_len, n_freqs)
        spec_nxt = cwtSL(data=data, dt=dt, scales=scales[last_jump:], n_cycles=n_cycles[i+1])
        # Which fractions for the current spec_nxt in the interval [order, order+1).
        scale_idxs = slice(last_jump, jump + 1)
        # Multiply non-fractional `spec_nxt` for all current scales/frequencies.
        spec[...,scale_idxs] *= np.power(spec_nxt[...,:(jump - last_jump + 1)], alphas[scale_idxs] * exps[scale_idxs])
        # Multiply non-fractional `spec_nxt` for all remaining scales/frequencies.
        spec[...,(jump + 1):] *= np.power(spec_nxt[...,(jump - last_jump + 1):], exps[(jump + 1):])
        # Go to the next [order, order+1) interval.
        last_jump = jump + 1
    # Transpose `spec` to get general formulation.
    # spec - (*, n_freqs, seq_len)
    spec = np.moveaxis(spec, -1, -2)
    # Return the final `spec`.
    return spec

"""
tool funcs
"""
# def _cwtSL func
def cwtSL(data, dt, scales, n_cycles):
    """
    The continuous Wavelet transform specifically for Morlets with the Superlet formulation of Moca et al. 2021.

    Morlet support gets adjusted by number of cycles. Normalisation is with `1 / (scale * 4pi)`. This way, the norm
    of the spectrum (modulus) at the corresponding harmonic frequency is the harmonic signal's amplitude.

    Args:
        data: (*, seq_len) - The raw time series.
        dt: float - The sampling time unit (second), i.e., `1 / sfreq`.
        scales: (n_freqs,) - The scaling factors, i.e, the inverse of frequencys of interest, `1 / (2 * np.pi * foi)`.
        n_cycles: int - The number of cycles of the wavelet.

    Returns:
        spec: (*, seq_len, n_freqs) - The spectrum of scaling factors.
    """
    return np.stack([_cwtSL(data=data, dt=dt, s=scale_i, c=n_cycles) for scale_i in scales], axis=-1)

# def _cwtSL func
def _cwtSL(data, dt, s, c):
    """
    The Continuous Wavelet Transform (CWT) specifically for Morlets with the Superlet formulation of Moca et al. 2021.

    Args:
        data: (*, seq_len) - The raw time series.
        dt: float - The sampling time unit (second), i.e., `1 / sfreq`.
        s: float - The scaling factor, which corresponds to `1 / (2\pi f)`.
        c: int - The number of cycles of the wavelet.

    Returns:
        spec: (*, seq_len) - The spectrum of the specified scale factor `s`.
    """
    # Get the superlet support according to `s` & `c` & `dt`.
    # t - (kernel_size,)
    t = _get_superlet_support(s=s, c=c, dt=dt)
    # Get the corresponding wavelet data.
    # wavelet - (kernel_size,)
    wavelet = (dt ** 0.5 / (4 * np.pi)) * morletSL(t=t, s=s, c=c, k_sd=5)
    # Convolve `data` with `wavelet` to get `spec`.
    # spec - (*, seq_len)
    spec = fftconvolve(data, np.broadcast_to(wavelet, shape=data.shape[:-1]+wavelet.shape), mode="same", axes=-1)
    # Return the final `spec`.
    return spec

# def morletSL func
def morletSL(t, s=1., c=3, k_sd=5):
    """
    The modified Morlet formulation according to Moca et al.[1], which shifts the admissability criterion
    from the central frequency to the number of cycles `c` within the Gaussian envelope which has
    a constant standard deviation of `k_sd`. The corresponding equations are as follows:
    >>> \psi_{f,c}(t)=\frac{1}{B_{c}\sqrt{2\pi}}e^{-\frac{t^{2}}{2B_{c}^{2}}e^{j2\pi ft}
    >>> B_{c}=\frac{c}{k_{sd}f}
    We should note that `1 / (2\pi f)` equals the scale factor `s`.

    Args:
        t: float - Time. If `s` is not specified, this can be used as the non-dimensional time `t/s`.
        s: float - Scaling factor, which corresponds to `1 / (2\pi f)`. Default is `1`.
        c: int - The number of cycles of the wavelet.
        k_sd: int - The spanning SDs of the Gaussian envelop.

    Returns:
        psi: complex - The morlet transformed result.

    Notes:
        [1] Moca, Vasile V., et al. "Time-frequency super-resolution with superlets."
            Nature communications 12.1 (2021): 1-18.
    """
    # Initialize the non-dimensional time `ts`.
    ts = t / s
    # Get scaled time spread parameter, also includes scale normalisation!
    B_c = k_sd / (s * c * (2 * np.pi) ** 1.5)
    # Calculate the final `output`.
    output = B_c * np.exp(1j * ts)
    output *= np.exp(-0.5 * (k_sd * ts / (2 * np.pi * c)) ** 2)
    # Return the final `output`.
    return output

# def _get_superlet_support func
def _get_superlet_support(s, c, dt):
    """
    Get the support for the convolution in superlet. We should note that the effective support
    for the convolution here is not only `scale`-dependent but also `cycle`-dependent.

    Args:
        s: float - The scaling factor, which corresponds to `1 / (2\pi f)`.
        c: int - The number of cycles of the wavelet.
        dt: float - The sampling time unit (second), i.e., `1 / sfreq`.

    Returns:
        t: (kernel_size,) - The corresponding convolution time steps, centered at 0.
    """
    # Initialize the number of points needed to capture wavelet.
    M = 10. * s * c / dt
    # The times to use, centred at zero.
    t = np.arange((-M + 1.) / 2., (M + 1.) / 2.) * dt
    # Return the final `t`.
    return t

"""
demo funcs
"""
# def gen_superlet_data func
def gen_superlet_data(freqs=[20, 40, 60], n_cycles=11, sfreq=1000, eps=0.):
    """
    Harmonic superposition of multiple few-cycle oscillations akin to the example of Figure 3 in Moca et al. 2021 NatComm.
    """
    # Initialize `signal` as empty list.
    signal = []
    for freq_i in freqs:
        # 10 cycles of f1.
        tvec = np.arange(n_cycles / freq_i, step=(1. / sfreq))
        harmonic = np.cos(2 * np.pi * freq_i * tvec)
        f_neighbor = np.cos(2 * np.pi * (freq_i + 10) * tvec)
        packet = harmonic + f_neighbor
        # 2 cycles time neighbor.
        delta_t = np.zeros(int(2 / freq_i * sfreq))
        # 5 cycles break.
        pad = np.zeros(int(5 / freq_i * sfreq))
        signal.extend([pad, packet, delta_t, harmonic])
    signal.append(pad)
    # Stack the packets together with some padding.
    signal = np.concatenate(signal)
    # Additive white noise.
    if eps > 0.: signal = np.random.randn(len(signal)) * eps + signal
    # Return the final `signal`.
    return signal

if __name__ == "__main__":
    import os, time
    import matplotlib.pyplot as plt

    # Initialize macros.
    path_img = os.path.join(os.getcwd(), "__image__")
    if not os.path.exists(path_img): os.makedirs(path_img)

    # Initialize sampling frequency & amplitude.
    n_samples = 16; sfreq = 1000; amplitude = 10
    # Initialize `signal` according to 20Hz, 40Hz and 60Hz.
    signal = amplitude * gen_superlet_data(sfreq=sfreq, eps=0.)
    # Get frequencies of interest in Hz.
    foi = np.linspace(1, 100, 50); scales = 1 / (2. * np.pi * foi)
    # Record the start time of superlet.
    time_start = time.time()
    # Get the corresponding spectrum.
    spec = superlet(np.stack([signal for _ in range(n_samples)], axis=0), sfreq=sfreq,
        scales=scales, cycle_order=(1, 30), cycle_base=5, use_adaptive=False)[0,...]
    # Record the total time of superlet.
    time_stop = time.time()
    print("INFO: The total time for {:d} samples is {:.2f}s.".format(n_samples, time_stop - time_start))
    # Get amplitude scalogram.
    ampls = np.abs(spec)
    # Plot the spectrum figure.
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios":[1,3]}, figsize=(6, 6))
    ax1.plot(np.arange(signal.size) / sfreq, signal, c="cornflowerblue"); ax1.set_ylabel("signal (a.u.)")
    extent = [0, len(signal) / sfreq, foi[0], foi[-1]]
    im = ax2.imshow(ampls, cmap="magma", aspect="auto", extent=extent, origin="lower")
    plt.colorbar(im,ax = ax2, orientation="horizontal", shrink=0.7, pad=0.2, label="amplitude (a.u.)")
    ax2.plot([0, len(signal) / sfreq], [20, 20], "--", c="0.5")
    ax2.plot([0, len(signal) / sfreq], [40, 40], "--", c="0.5")
    ax2.plot([0, len(signal) / sfreq], [60, 60], "--", c="0.5")
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("frequency (Hz)")
    fig.tight_layout(); plt.savefig(os.path.join(path_img, "superlet_sp.png"))


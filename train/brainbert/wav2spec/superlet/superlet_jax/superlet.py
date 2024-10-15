#!/usr/bin/env python3
"""
Created on 16:45, Nov. 10th, 2023

@author: Norbert Zheng
"""
import jax
import jax.numpy as jnp
from functools import partial
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir, os.pardir, os.pardir))
    from morlet import morlet_transform
else:
    from .morlet import morlet_transform

__all__ = [
    "superlet_transform",
]

"""
core funcs
"""
# def superlet_transform func
def superlet_transform(data, sfreq, scales, cycle_order, cycle_base=3, use_adaptive=True):
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
        spec = None
    # Fractional Adaptive SLT.
    else:
        spec = adaptive_superlet_transform(data=data, sfreq=sfreq, scales=scales,
            cycle_order=cycle_order, cycle_base=cycle_base, cycle_mode="mul")
    # Return the final `spec`.
    return spec

# def adaptive_superlet_transform func
def adaptive_superlet_transform(data, sfreq, freqs, cycle_order, cycle_base=3, cycle_mode="mul"):
    """
    Computes the adaptive superlet transform of the provided signal.

    Args:
        data: (*, seq_len) - The raw time series. The last dimension is interpreted as the time axis.
        sfreq: float - The sample rate of `data` in Hz.
        freqs: (n_freqs,) - The set of frequencies (of interest) to use in wavelet transform.
        cycle_order: tuple - The order of the superlet set, containing [order_min, order_max]. `order_min`
            controls the minimal number of cycles within a SL together with `cycle_base` parameter:
            `cycle_min = cycle_base * order_min`. `order_max` controls the minimal number of cycles
            within a SL together with the `cycle_base` parameter: `cycle_max = cycle_base * order_min`.
            Note that for admissability reasons `cycle_min` should be at least 3!
        cycle_base: int - The number of cycles of the base Morlet wavelet. If set to lower
            than 3 increase `order_min` as to never have less than 3 cycles in a wavelet!
        mode: str - The use of additive or multiplicative adaptive superlets, should be one of ["add","mul"].

    Returns:
        spec: (*, n_freqs, seq_len) - The complex time-frequency representation of the input data.
    """
    # Create `n_cycles` according to `cycle_base` & `cycle_order[1]` & `mode`.
    # n_cycles - (cycle_order[1],)
    if cycle_mode == "add":
        n_cycles = jnp.arange(0, cycle_order[1]) + cycle_base
    elif cycle_mode == "mul":
        n_cycles = jnp.arange(1, cycle_order[1] + 1) * cycle_base
    else:
        raise ValueError("ERROR: Get unknown cycle mode {} in superlet_jax.superlet.".format(cycle_mode))
    # Get `orders` according to `freqs` & `cycle_order`.
    # orders - (n_freqs,); order_mask - (n_freqs,)
    orders = cycle_order[0] + ((cycle_order[1] - cycle_order[0]) * (freqs - min(freqs)) / (max(freqs) - min(freqs)))
    order_mask = get_order_mask(orders, cycle_order[1])
    # Every scale needs a different exponent for the geometric mean.
    # exps - (n_freqs,)
    exps = 1 / (orders - cycle_order[0] + 1)
    # Get the transformed spectrum.
    # spec - (*, n_cycles, n_freqs, seq_len)
    spec = _superlet_transform(data, sfreq, n_cycles, freqs)
    spec = spec.at[order_mask.T].set(1)
    # Geometric normalization according to scale dependent order.
    # spec - (*, n_freqs, seq_len)
    spec = jnp.moveaxis(jnp.power(jnp.moveaxis(jnp.prod(spec, axis=-3), -1, -2), exps), -1, -2)
    # Return the final `spec`.
    return spec

# def _superlet_transform func
@partial(jax.jit, static_argnums=1)
@partial(jax.vmap, in_axes=(None, None, 0, None))
def _superlet_transform(data, sfreq, c, fois):
    """
    Wrap Morlet Transform in superlet formulation.

    Args:
        data: (*, seq_len) - The raw time series. The last dimension is interpreted as the time axis.
        sfreq: float - The sample rate of `data` in Hz.
        c: int - The number of cycles of the wavelet.
        fois: (n_freqs,) - The set of frequencies (of interest) to use in wavelet transform.

    Returns:
        spec: (*, n_freqs, seq_len) - The complex time-frequency representation of the input data.
    """
    return morlet_transform(data, sfreq, c, fois) * jnp.sqrt(2)

# def get_order_mask func
@partial(jax.vmap, in_axes=(0, None))
def get_order_mask(orders, order_max):
    """
    Get the order mask.
    """
    return jnp.arange(1, order_max + 1) > orders

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
        tvec = jnp.arange(n_cycles / freq_i, step=(1. / sfreq))
        harmonic = jnp.cos(2 * jnp.pi * freq_i * tvec)
        f_neighbor = jnp.cos(2 * jnp.pi * (freq_i + 10) * tvec)
        packet = harmonic + f_neighbor
        # 2 cycles time neighbor.
        delta_t = jnp.zeros(int(2 / freq_i * sfreq))
        # 5 cycles break.
        pad = jnp.zeros(int(5 / freq_i * sfreq))
        signal.extend([pad, packet, delta_t, harmonic])
    signal.append(pad)
    # Stack the packets together with some padding.
    signal = jnp.concatenate(signal)
    # Additive white noise.
    if eps > 0.: signal = jnp.random.randn(len(signal)) * eps + signal
    # Return the final `signal`.
    return signal

if __name__ == "__main__":
    import os, time
    import matplotlib.pyplot as plt

    # Initialize macros.
    path_img = os.path.join(os.getcwd(), "__image__")
    if not os.path.exists(path_img): os.makedirs(path_img)

    # Initialize sampling frequency & amplitude.
    n_samples = 4; sfreq = 1000; amplitude = 10
    # Initialize `signal` according to 20Hz, 40Hz and 60Hz.
    signal = amplitude * gen_superlet_data(sfreq=sfreq, eps=0.)
    # Get frequencies of interest in Hz.
    foi = jnp.linspace(1, 100, 50); scales = 1 / (2. * jnp.pi * foi)
    # Record the start time of superlet.
    time_start = time.time()
    # Get the corresponding spectrum.
    #spec = superlet(np.stack([signal for _ in range(n_samples)], axis=0), sfreq=sfreq,
    #    scales=scales, cycle_order=(1, 30), cycle_base=5, use_adaptive=False)[0,...]
    spec = adaptive_superlet_transform(data=signal, sfreq=sfreq,
        freqs=foi, cycle_order=(1, 30), cycle_base=5, cycle_mode="mul")
    # Record the total time of superlet.
    time_stop = time.time()
    print("INFO: The total time for {:d} samples is {:.2f}s.".format(n_samples, time_stop - time_start))
    # Get amplitude scalogram.
    ampls = jnp.abs(spec)
    # Plot the spectrum figure.
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios":[1,3]}, figsize=(6, 6))
    ax1.plot(jnp.arange(signal.size) / sfreq, signal, c="cornflowerblue"); ax1.set_ylabel("signal (a.u.)")
    extent = [0, len(signal) / sfreq, foi[0], foi[-1]]
    im = ax2.imshow(ampls, cmap="magma", aspect="auto", extent=extent, origin="lower")
    plt.colorbar(im,ax = ax2, orientation="horizontal", shrink=0.7, pad=0.2, label="amplitude (a.u.)")
    ax2.plot([0, len(signal) / sfreq], [20, 20], "--", c="0.5")
    ax2.plot([0, len(signal) / sfreq], [40, 40], "--", c="0.5")
    ax2.plot([0, len(signal) / sfreq], [60, 60], "--", c="0.5")
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("frequency (Hz)")
    fig.tight_layout(); plt.savefig(os.path.join(path_img, "superlet_jax.png"))


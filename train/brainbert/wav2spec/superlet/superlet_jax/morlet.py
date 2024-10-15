#!/usr/bin/env python3
"""
Created on 17:14, Nov. 10th, 2023

@author: Norbert Zheng
"""
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir, os.pardir, os.pardir))

__all__ = [
    "wavelet_transform",
]

# def calculate_Bc func
def calculate_Bc(c, f, k_sd=5):
    """
    Calculate `B_c = \frac{c}{k_{sd}f}`.

    Args:
        c: int - The number of cycles of the wavelet.
        f: float - The frequency of interest, `1 / (2\pi f)` equals scale factor `s`.
        k_sd: int - The spanning SDs of the Gaussian envelop.

    Returns:
        B_c: float - The time spread parameter.
    """
    return c / (k_sd * f)

# def cxmorelet func
def cxmorelet(sfreq, c, f):
    """
    Calculate the convolution kernel for Morlet in superlet formulation.

    Args:
        sfreq: float - The sample rate of `data` in Hz.
        c: int - The number of cycles of the wavelet.
        f: float - The frequency of interest, `1 / (2\pi f)` equals scale factor `s`.

    Returns:
        psi: (2*sfreq,) - The morlet convolution kernel.
    """
    # Get support according to `sfreq`.
    # t - (2 * sfreq,)
    t = jnp.linspace(start=-1., stop=1., num=2*sfreq)
    # Calculate `B_c` according to `c` & `f`.
    # B_c - float
    B_c = calculate_Bc(c=c, f=f, k_sd=5)
    # Calculate `psi` according to `B_c` & `t` & `f`.
    # psi - (2*sfreq,)
    psi = (1. / (B_c * jnp.sqrt(2 * jnp.pi))) *\
        jnp.exp(-(t ** 2) / (2. * (B_c ** 2))) *\
        jnp.exp(1j * 2. * jnp.pi * f * t)
    # Return the final `psi`.
    return psi / jnp.sum(jnp.abs(psi))

# def morlet_transform func
@partial(jax.jit, static_argnums=1)
@partial(jax.vmap, in_axes=(None, None, None, 0))
def morlet_transform(data, sfreq, c, f):
    """
    The Continuous Wavelet Transform (CWT) for Morlet in superlet formulation.

    Args:
        data: (seq_len,) - The raw time series.
        sfreq: float - The sample rate of `data` in Hz.
        c: int - The number of cycles of the wavelet.
        f: (n_freqs,) - The frequency of interest, `1 / (2\pi f)` equals scale factor `s`.

    Returns:
        spec: (n_freqs, seq_len) - The morlet spectrum.
    """
    # Calculate morlet convolution kernel.
    # psi - (2*sfreq,)
    psi = cxmorelet(sfreq=sfreq, c=c, f=f)
    # Calculate the corresponding spectrum.
    # spec - (seq_len,)
    spec = jsp.signal.fftconvolve(data, psi, mode="same")
    # Return the final `spec`.
    return spec

if __name__ == "__main__":
    import numpy as np

    # Initialize macros.
    sfreq = 2048; seq_len = sfreq * 5; n_freqs = 10; c = 3

    # Initialize raw time series `X`.
    # X - (seq_len,)
    X = np.random.uniform(low=0., high=1., size=(seq_len,)).astype(np.float32)
    fois = np.random.uniform(low=0., high=1., size=(n_freqs,)).astype(np.float32)
    # Forward `morlet_transform`.
    # Z - (n_freqs, seq_len)
    Z = morlet_transform(X, sfreq, c, fois)


#!/usr/bin/env python3
"""
Created on 21:31, Nov. 30th, 2023

@author: Norbert Zheng
"""
import numpy as np
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))
from utils import DotDict

__all__ = [
    # Macros.
    "default_mask_params",
    # Core Functions.
    "mask_data",
]

"""
macros
"""
# def default_mask_params macro
default_mask_params = DotDict({
    # The default mask parameters related to random mask strategy.
    "random": {
        # The type of mask.
        "mask_type": "random",
        # The mask parameters related to time axis.
        "time": {
            # The probability of mask.
            "p_mask": 0.1,
            # The minimum number of consecutive steps.
            "step_min": 1,
            # The maximum number of consecutive steps.
            "step_max": 5,
        },
        # The mask parameters related to frequency axis.
        "frequency": {
            # The probability of mask.
            "p_mask": 0.1,
            # The minimum number of consecutive steps.
            "step_min": 1,
            # The maximum number of consecutive steps.
            "step_max": 2,
        },
    },
    # The default mask parameters related to adaptive mask strategy.
    "adaptive": {
        # The type of mask.
        "mask_type": "adaptive",
        ## The mask parameters related to time & frequency axis. As the mask segment along time axis is
        ## adaptive corresponding to the frequency axis, they share the same parameters.
        # The minimum frequency to mask, to avoid overflow when calculating the adaptive length.
        "freq_min": 0.01,
        # The maximum frequency to mask, to avoid overflow when calculating the adaptive length.
        "freq_max": 250.,
        # The number of frequencies to construct frequency linespace.
        "n_freqs": 40,
        # The probability of mask.
        "p_mask": 0.1,
    },
})

"""
core funcs
"""
# def mask_data func
def mask_data(data, mask_params):
    """
    Mask data according to mask parameters.

    Args:
        data: (seq_len, n_freqs) - The raw spectrum series.
        mask_params: DotDict - The mask parameters.

    Returns:
        data_masked: (seq_len, n_freqs) - The masked data.
        mask: (seq_len, n_freqs) - The mask matrix.
    """
    # If the type of mask is `random`, execute random mask.
    if mask_params.mask_type == "random":
        return mask_random(data, mask_params)
    # If the type of mask is `adaptive`, execute adaptive mask.
    elif mask_params.mask_type == "adaptive":
        return mask_adaptive(data, mask_params)
    # Get unknown mask type, raise error.
    else:
        raise ValueError("ERROR: Get unknown mask type {} in train.brainbert.mask.".format(mask_params.mask_type))

"""
mask funcs
"""
# def mask_random func
def mask_random(data, mask_params):
    """
    Mask data with random strategy.

    Args:
        data: (seq_len, n_freqs) - The raw spectrum series.
        mask_params: DotDict - The mask parameters.

    Returns:
        data_masked: (seq_len, n_freqs) - The masked data.
        mask: (seq_len, n_freqs) - The mask matrix.
    """
    # Initialize `data_masked` & `mask`.
    # data_masked - (seq_len, n_freqs); mask - (seq_len, n_freqs)
    data_masked = data.copy(); mask = np.zeros_like(data, dtype=np.bool_)
    # Initialize the default mask fill value.
    default_mask_value = _default_mask_value(data)
    ## Mask data along the time axis.
    # Generate mask intervals along the time axis.
    mask_intervals_time = _mask_random_intervals(data=data, p_mask=mask_params.time.p_mask,
        step_min=mask_params.time.step_min, step_max=mask_params.time.step_max, axis=0)
    # Update `mask` according to `mask_intervals_time`.
    for (start_i, end_i) in mask_intervals_time: mask[start_i:end_i,:] = True
    # Update `data_masked` according to `mask_intervals_time`.
    for (start_i, end_i) in mask_intervals_time:
        # Fill `data_masked` with different values according to probability.
        p_value_i = np.random.random()
        # If `p_value_i` is in [0.0,0.1), do not change values.
        # TODO: Look at attention scores.
        if 0. <= p_value_i < 0.1:
            pass
        # If `p_value_i` is in [0.1,0.2), replace with another segment.
        elif 0.1 <= p_value_i < 0.2:
            # Initialize the start of random segment.
            diff_i = end_i - start_i; start_random_i = np.random.randint(low=0, high=(data.shape[0] - diff_i))
            # Replace the data segment with another randomly selected segment.
            # Note: We use `data_masked`, instead of `data`, to avoid overwrite the original masked part.
            data_masked[start_i:end_i,:] = data_masked[start_random_i:(start_random_i+diff_i),:]
        # If `p_value_i` is in [0.2,1.0), replace with default mask value.
        else:
            data_masked[start_i:end_i,:] = default_mask_value
    ## Mask data along the frequency axis.
    # Generate mask intervals along the frequency axis.
    mask_intervals_freq = _mask_random_intervals(data=data, p_mask=mask_params.frequency.p_mask,
        step_min=mask_params.frequency.step_min, step_max=mask_params.frequency.step_max, axis=1)
    # Update `mask` according to `mask_intervals_freq`.
    for (start_i, end_i) in mask_intervals_freq: mask[:,start_i:end_i] = True
    # Update `data_masked` according to `mask_intervals_freq`.
    for (start_i, end_i) in mask_intervals_freq:
        # Fill `data_masked` with different values according to probability.
        p_value_i = np.random.random()
        # If `p_value_i` is in [0.0,0.1), do not change values.
        # TODO: Look at attention scores.
        if 0. <= p_value_i < 0.1:
            pass
        # If `p_value_i` is in [0.1,0.2), replace with another segment.
        elif 0.1 <= p_value_i < 0.2:
            # Initialize the start of random segment.
            diff_i = end_i - start_i; start_random_i = np.random.randint(low=0, high=(data.shape[1] - diff_i))
            # Replace the data segment with another randomly selected segment.
            # Note: We use `data_masked`, instead of `data`, to avoid overwrite the original masked part.
            data_masked[:,start_i:end_i] = data_masked[:,start_random_i:(start_random_i+diff_i)]
        # If `p_value_i` is in [0.2,1.0), replace with default mask value.
        else:
            data_masked[:,start_i:end_i] = default_mask_value
    # Return the final `data_masked` & `mask`.
    return data_masked, mask.astype(dtype=data_masked.dtype)

# def _mask_random_intervals func
def _mask_random_intervals(data, p_mask, step_min=1, step_max=2, axis=0):
    """
    Generate mask intervals along time/frequency axis with random strategy.

    Args:
        data: (seq_len, n_freqs) - The raw spectrum series.
        p_mask: float - The probability of mask.
        step_min: int - The minimum number of consecutive steps.
        step_max: int - The maximum number of consecutive steps.
        axis: int - The axis along which to generate mask intervals, `0` for time axis, `1` for frequency axis.

    Returns:
        intervals: (n_intervals[list],) - The tuple of mask intervals, each of which contains [start,end].
    """
    assert step_min <= step_max and step_max < data.shape[axis]
    # Construct mask intervals according to the valid start points.
    intervals = []
    for step_idx in range(data.shape[axis] - step_max):
        if np.random.random() < p_mask:
            # Make sure that there is no overlap among intervals.
            if (len(intervals) == 0) or (intervals[-1][1] < step_idx):
                intervals.append((step_idx, step_idx + np.random.randint(low=step_min, high=step_max)))
    # Return the final `intervals`.
    return intervals

# def mask_adaptive func
def mask_adaptive(data, mask_params):
    """
    Mask data with adaptive strategy.

    Args:
        data: (seq_len, n_freqs) - The raw spectrum series.
        mask_params: DotDict - The mask parameters.

    Returns:
        data_masked: (seq_len, n_freqs) - The masked data.
        mask: (seq_len, n_freqs) - The mask matrix.
    """
    # Mask data along time axis.
    data_masked, mask_time = _mask_adaptive_time(data, mask_params)
    # Mask data along freq axis.
    # Note: We use `data_masked`, instead of `data`, to avoid overwrite the original masked part.
    data_masked, mask_freq = _mask_adaptive_freq(data_masked, mask_params)
    # Aggregate to get the final `mask`.
    mask = mask_time | mask_freq
    # Return the final `data_masked` & `mask`.
    return data_masked, mask.astype(dtype=data_masked.dtype)

# def _mask_adaptive_time func
def _mask_adaptive_time(data, mask_params):
    """
    Mask data along time axis with adaptive strategy.

    Args:
        data: (seq_len, n_freqs) - The raw spectrum series.
        mask_params: DotDict - The mask parameters.

    Returns:
        data_masked: (seq_len, n_freqs) - The masked data.
        mask: (seq_len, n_freqs) - The mask matrix.
    """
    # Initialize `data_masked` & `mask`.
    # data_masked - (seq_len, n_freqs); mask - (seq_len, n_freqs)
    data_masked = data.copy(); mask = np.zeros_like(data, dtype=np.bool_)
    # Initialize the default mask fill value.
    default_mask_value = _default_mask_value(data)
    # Initialize `freqs` & `mask_len` according to `freq_min` & `freq_max` & `n_freqs`.
    # freqs - (n_freqs[list],); mask_len - (n_freqs[list],)
    freqs = np.linspace(start=mask_params.freq_min, stop=mask_params.freq_max, num=mask_params.n_freqs)
    min_mask_len = np.random.randint(low=1, high=2)
    mask_len = [int(max(min_mask_len, (200. / (25. + freq_i)))) for freq_i in freqs]
    # Construct mask centers according to the valid center points.
    # Note: The mask is centered on the time position.
    mask_centers = []
    for time_idx in range(max(mask_len), data.shape[0] - max(mask_len)):
        if np.random.random() < mask_params.p_mask:
            if (len(mask_centers) == 0) or (np.abs(mask_centers[-1] - time_idx) > (2 * max(mask_len)) + 1):
                # Note: We further limit that there is at least one unmask point between mask intervals.
                mask_centers.append(time_idx)
    # Update `data_masked` & `mask` according to `mask_centers`.
    for center_i in mask_centers:
        # Fill `mask` according to `center_i`.
        mask = _set_mask_adaptive_time(data=mask, time_idx=center_i, mask_len=mask_len, mask_value=True)
        # Fill `data_masked` with different values according to probability.
        p_value_i = np.random.random()
        # If `p_value_i` is in [0.0,0.1), do not change values.
        # TODO: Look at attention scores.
        if 0. <= p_value_i < 0.1:
            pass
        # If `p_value_i` is in [0.1,0.2), replace with another segment.
        elif 0.1 <= p_value_i < 0.2:
            # Initialize the center of random segment.
            center_random_i = np.random.randint(low=max(mask_len), high=(data.shape[0] - max(mask_len)))
            # Replace the data segment with another randomly selected segment.
            # Note: We use `data_masked`, instead of `data`, to avoid overwrite the original masked part.
            mask_value_i = _get_mask_adaptive_time(data=data_masked, time_idx=center_random_i, mask_len=mask_len)
            data_masked = _set_mask_adaptive_time(
                data=data_masked, time_idx=center_i, mask_len=mask_len, mask_value=mask_value_i
            )
        # If `p_value_i` is in [0.2,1.0), replace with default mask value.
        else:
            data_masked = _set_mask_adaptive_time(
                data=data_masked, time_idx=center_i, mask_len=mask_len, mask_value=default_mask_value
            )
    # Return the final `data_masked` & `mask`.
    return data_masked, mask

# def _set_mask_adaptive_time func
def _set_mask_adaptive_time(data, time_idx, mask_len, mask_value):
    """
    Set mask value of data along time axis with adaoptive strategy.

    Args:
        data: (seq_len, n_freqs) - The raw spetrum series.
        time_idx: int - The index along time axis to set mask value.
        mask_len: (n_freqs[list],) - The mask length of each frequency.
        mask_value: (n_freqs[list],) or float - The mask value of each frequency.

    Returns:
        data_masked: (seq_len, n_freqs) - The masked data.
    """
    assert data.shape[1] == len(mask_len)
    if isinstance(mask_value, list):
        assert len(mask_len) == len(mask_value)
        for mask_len_i, mask_value_i in zip(mask_len, mask_value): assert 2 * mask_len_i == len(mask_value_i)
    for freq_idx in range(len(mask_len)):
        data[
            max(0,time_idx-mask_len[freq_idx]):min(data.shape[0],time_idx+mask_len[freq_idx]), freq_idx
        ] = mask_value[freq_idx] if isinstance(mask_value, list) else mask_value
    # Return the final `data_masked`.
    return data

# def _get_mask_adaptive_time func
def _get_mask_adaptive_time(data, time_idx, mask_len):
    """
    Get mask value from data along time axis with adaptive strategy.

    Args:
        data: (seq_len, n_freqs) - The raw spectrum series.
        time_idx: int - The index along time axis to get mask value.
        mask_len: (n_freqs[list],) - The mask length of each frequency.

    Returns:
        mask_value: (n_freqs[list],) - The mask value of each frequency.
    """
    assert data.shape[1] == len(mask_len)
    # Get `mask_value` according to `mask_len`.
    mask_value = [data[
        max(0,time_idx-mask_len[freq_idx]):min(data.shape[0],time_idx+mask_len[freq_idx]), freq_idx
    ] for freq_idx in range(len(mask_len))]
    # Return the final `mask_value`.
    return mask_value

# def _mask_adaptive_freq func
def _mask_adaptive_freq(data, mask_params):
    """
    Mask data along frequency axis with adaptive strategy.

    Args:
        data: (seq_len, n_freqs) - The raw spectrum series.
        mask_params: DotDict - The mask parameters.

    Returns:
        data_masked: (seq_len, n_freqs) - The masked data.
        mask: (seq_len, n_freqs) - The mask matrix.
    """
    # Initialize `data_masked` & `mask`.
    # data_masked - (seq_len, n_freqs); mask - (seq_len, n_freqs)
    data_masked = data.copy(); mask = np.zeros_like(data, dtype=np.bool_)
    # Initialize the default mask fill value.
    default_mask_value = _default_mask_value(data)
    # Initialize `freqs` & `mask_len` according to `freq_min` & `freq_max` & `n_freqs`.
    # freqs - (n_freqs[list],); mask_len - (n_freqs[list],)
    freqs = np.linspace(start=mask_params.freq_min, stop=mask_params.freq_max, num=mask_params.n_freqs)
    mask_len = [max(1, int(4.9 * freq_i / 250.)) for freq_i in freqs]
    # Construct mask starts according to the valid start points.
    mask_starts = [freq_idx for freq_idx in range(data.shape[1] - max(mask_len)) if np.random.random() < mask_params.p_mask]
    # Update `data_masked` & `mask` according to `mask_starts`.
    for start_i in mask_starts:
        # Initialize `diff_i` according to `mask_len`.
        diff_i = mask_len[start_i]
        # Update `mask` according to `diff_i`.
        mask[:,start_i:(start_i+diff_i)] = True
        # Fill `data_masked` with different values according to probability.
        p_value_i = np.random.random()
        # If `p_value_i` is in [0.0,0.1), do not change values.
        # TODO: Look at attention scores.
        if 0. <= p_value_i < 0.1:
            pass
        # If `p_value_i` is in [0.1,0.2), replace with another segment.
        elif 0.1 <= p_value_i < 0.2:
            # Initialize the start of random segment.
            start_random_i = np.random.randint(low=0, high=(data.shape[1] - diff_i))
            # Replace the data segment with another randomly selected segment.
            # Note: We use `data_masked`, instead of `data`, to avoid overwrite the original masked part.
            data_masked[:,start_i:(start_i+diff_i)] = data_masked[:,start_random_i:(start_random_i+diff_i)]
        # If `p_value_i` is in [0.2,1.0), replace with default mask value.
        else:
            data_masked[:,start_i:(start_i+diff_i)] = default_mask_value
    # Return the final `data_masked` & `mask`.
    return data_masked, mask

"""
tool funcs
"""
# def _default_mask_value func
def _default_mask_value(data):
    """
    Get the default value to fill mask part.

    Args:
        data: (seq_len, n_freqs) - The raw spectrum series.

    Returns:
        value: float - The default value to fill mask part.
    """
    return 0.

"""
plot funcs
"""
def plot_mask(data, data_masked, mask, img_fname=None):
    """
    Plot the original spectrum series & the masked spectrum series.

    Args:
        data: (seq_len, n_freqs) - The raw spectrum series.
        data_masked: (seq_len, n_freqs) - The masked spectrum series.
        mask: (seq_len, n_freqs) - The masked spectrum series.
        img_fname: str - The path to save image.

    Returns:
        None
    """
    import matplotlib.pyplot as plt
    # Transpose `data` & `data_masked` & `mask`.
    # data - (n_freqs, seq_len); data_masked - (n_freqs, seq_len); mask - (n_freqs, seq_len)
    data = data.T; data_masked = data_masked.T; mask = mask.T
    # Initialize the plotting figure.
    fig_size = (10, 15); fig, axes = plt.subplots(3, 1, figsize=fig_size)
    # Plot the raw spectrum series.
    axes[0].imshow(data, cmap="viridis", origin="lower"); axes[0].set_title("Raw Spectrum Series")
    axes[0].set_xlabel("Time (step)"); axes[0].set_ylabel("Frequency (step)")
    # Plot the masked spectrum series.
    axes[1].imshow(data_masked, cmap="viridis", origin="lower"); axes[1].set_title("Masked Spectrum Series")
    axes[1].set_xlabel("Time (step)"); axes[1].set_ylabel("Frequency (step)")
    # Plot the high-lighted msak positions.
    axes[2].imshow(data, cmap="viridis", origin="lower"); axes[2].set_title("High-lighted Mask Positions")
    axes[2].set_xlabel("Time (step)"); axes[2].set_ylabel("Frequency (step)")
    row_idxs, col_idxs = np.where(mask > 0.)
    for row_idx, col_idx in zip(row_idxs, col_idxs):
        axes[2].add_patch(plt.Rectangle(xy=((col_idx-0.5), (row_idx-0.5)),
            width=1., height=1., color="red"))
    # If `img_fname` is None, directly show image.
    if img_fname is None:
        fig.show()
    # If `img_fname` is not None, save image to the specified path.
    else:
        fig.savefig(fname=img_fname)

if __name__ == "__main__":
    # Initialize macros.
    seq_len = 80; n_freqs = 40
    path_img = os.path.join(os.getcwd(), "__image__")
    if not os.path.exists(path_img): os.makedirs(path_img)

    # Initialize random seed.
    np.random.seed(4)

    # Initialize the spectrum series.
    # data - (seq_len, n_freqs)
    data = np.random.normal(loc=0., scale=1., size=(seq_len, n_freqs)).astype(np.float32)
    # Mask data with random strategy.
    # data_masked - (seq_len, n_freqs); mask - (seq_len, n_freqs)
    data_masked, mask = mask_data(data, default_mask_params.random)
    plot_mask(data, data_masked, mask, img_fname=os.path.join(path_img, "mask.random.png"))
    # Mask data with adaptive strategy.
    data_masked, mask = mask_data(data, default_mask_params.adaptive)
    plot_mask(data, data_masked, mask, img_fname=os.path.join(path_img, "mask.adaptive.png"))


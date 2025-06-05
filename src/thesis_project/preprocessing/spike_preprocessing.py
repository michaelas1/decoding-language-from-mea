import pickle
from typing import Optional
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.discriminant_analysis import StandardScaler

from thesis_project.data_loading import load_session_ids


def normalize_spikerates(spikerates):
    scaled_spikerates = np.zeros(spikerates.shape)
    for i in range(spikerates.shape[-1]):
        # treat each channel as a separate feature
        scaled_spikerates[:, :, i] = StandardScaler().fit_transform(spikerates[:, :, i])

    return scaled_spikerates


def blur_spikerates(spikerates, kernel_sd: float = 40):
    """
    Convolves spikerates with a Gaussian filter.
    """
    # blur along the timeseries axis
    preprocessed_spikerates = gaussian_filter1d(spikerates, sigma=kernel_sd, axis=1)
    return preprocessed_spikerates


def bin_spiketimes(
    input_dir: str,
    output_dir: str,
    bin_size: int = 50,
    session_ids: Optional[str] = None,
    conversion_factor: int = 1000,
    add_blur: bool = False,
    kernel_sd: int = 2,
):
    """
    Bin the spiketimes to obtain spikerates.
    :param input_dir: Directory where spiketimes are stored
    :param output_dir: Directory where spikerates should be stored
    :param bin_size: Bin size in ms
    :param session_ids: The session IDs to load. Infers them from
        the directory content if None
    :param conversion_factor: The conversion factor form spiketimes
        to bin_size. Per default converts from s to ms (* 1000)
    :param add_blur: Whether to add a Gaussian blur to the results
    """

    if session_ids is None:
        session_ids = load_session_ids(data_dir=input_dir)

    spiketimes_dict = {}
    min_time = 0
    max_time = 0

    for session_id in session_ids:
        spiketimes_path = (
            f"{input_dir}/{session_id}_naming_spiketimes_prep_and_production.pkl"
        )
        with open(spiketimes_path, "rb") as file:
            spiketimes = pickle.load(file)
            spiketimes_dict[session_id] = spiketimes

            max_entry = max(
                [
                    np.max(subentry)
                    for entry in np.asarray(spiketimes, dtype=object)
                    for subentry in entry
                    if len(subentry)
                ]
            )

            if max_entry > max_time:
                max_time = max_entry

    duration = (max_time - min_time) * conversion_factor  # convert s to ms
    seq_len = int(np.ceil(duration / bin_size))

    for session_id, spiketimes in spiketimes_dict.items():

        n_channels = len(spiketimes[0])
        n_trials = len(spiketimes)

        spikerates = np.zeros((n_trials, seq_len, n_channels))
        for i, trial in enumerate(spiketimes):
            for j, channel in enumerate(trial):
                for event in channel:
                    bin_idx = int((event * conversion_factor) / bin_size)
                    spikerates[i, bin_idx, j] += 1

        if add_blur:
            spikerates = normalize_spikerates(spikerates)
            spikerates = blur_spikerates(spikerates, kernel_sd=kernel_sd)
            path = f"{output_dir}/{session_id}_naming_spikerates_bin_size_{bin_size}_blur_sd_{kernel_sd}_prep_and_production.pkl"
        else:
            path = f"{output_dir}/{session_id}_naming_spikerates_bin_size_{bin_size}_prep_and_production.pkl"

        with open(path, "wb") as file:
            pickle.dump(spikerates, file)

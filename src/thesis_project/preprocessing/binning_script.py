import pickle

import numpy as np
from thesis_project.data_loading import construct_spikerates_filename, load_session_ids
from thesis_project.preprocessing.spike_preprocessing import bin_spiketimes, blur_spikerates, normalize_spikerates
from thesis_project.settings import EXPERIMENT2_DIR


INPUT_DIR = "thesis_project/neuro_data/experiment1"
OUTPUT_DIR = "thesis_project/neuro_data/experiment1/binned_spikerates"

def blur_experiment1_spikerates():

    # rebin to bin size = 20ms
    bin_spiketimes(INPUT_DIR, OUTPUT_DIR, bin_size=20)

    # rebin with Gaussian blur
    bin_spiketimes(INPUT_DIR, OUTPUT_DIR, bin_size=20, add_blur=True)

    # expeirmenting with blurring larger bin sizes
    bin_spiketimes(INPUT_DIR, OUTPUT_DIR, bin_size=50, add_blur=True, kernel_sd=2)

    bin_spiketimes(INPUT_DIR, OUTPUT_DIR, bin_size=50, add_blur=True, kernel_sd=1)


def blur_experiment2_spikerates():

    spikerates_dir = EXPERIMENT2_DIR+"/binned_spikerates"

    for bin_size in [50, 100]:
        # load spikerates
        spikerates_path = construct_spikerates_filename(
                "20240708", spikerates_dir, bin_size=bin_size, experiment="experiment2"
            )
        with open(spikerates_path, "rb") as file:
            spikerates = pickle.load(file)
            try:
                spikerates = np.asarray(spikerates)
            except Exception:
                spikerates = np.asarray(spikerates, dtype=object)

        spikerates = normalize_spikerates(spikerates)


        for kernel_sd in [1, 2]:
            # blur spikerates
            blurred_spikerates = blur_spikerates(spikerates, kernel_sd=kernel_sd)

            # save spikerates
            spikerates_path = construct_spikerates_filename(
            "20240708", spikerates_dir, bin_size=bin_size, blur_sd=kernel_sd, experiment="experiment2"
            )

            with open(spikerates_path, "wb") as file:
                pickle.dump(blurred_spikerates, file)


if __name__ == "__main__":
    blur_experiment2_spikerates()
import pdb
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from GravNetLayersRagged import RandomSampling, MultiBackScatter


def create_random_data(n_events, n_features, n_points_min, n_points_max, track_frequency=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed=seed)

    n_points = np.random.randint(n_points_min, n_points_max, n_events)
    n_total = np.sum(n_points)
    row_splits = np.cumsum(n_points)
    row_splits = np.insert(row_splits, 0, 0)
    index = np.array([])
    for i in range(len(row_splits)-1):
        start = row_splits[i]
        stop = row_splits[i+1]
        index = np.concatenate((index, np.arange(0, stop-start)))
        # index.append(np.arange(start, stop))
    # index = np.flatten(np.array(index))
    index = index.reshape((-1, 1))
    data = np.random.normal(size=(n_total, n_features))
    data = np.concatenate((data, index), axis=1)
    is_track = np.random.uniform(size=(n_total, 1)) < track_frequency
    return data, is_track, row_splits



if __name__ == '__main__':
    N_EVENTS = 3
    N_FEATURES = 5
    N_POINTS_MIN = 30
    N_POINTS_MAX = 50
    TRACK_FREQUENCY = 0.1

    # create data with the same shape as used in the network
    data, is_track, row_splits = create_random_data(
            n_events=N_EVENTS, n_features=N_FEATURES, n_points_min=N_POINTS_MIN,
            n_points_max=N_POINTS_MAX, track_frequency=TRACK_FREQUENCY, seed=8)

    print("For testing purporses random data has been created")
    print(f"Created {N_EVENTS} events that have between {N_POINTS_MIN} and \
{N_POINTS_MAX} hits per event. {100 * TRACK_FREQUENCY} % of the \
hits are treated as tracks. All hits have {N_FEATURES} features")
    print()

    reductions = np.arange(2., 20., 2.)
    print(f"Testing `RandomSampling` for reduction factors between 2 and 18")

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15,15))
    ax = ax.flatten()
    passed = True

    for i, red in enumerate(reductions):
        values = []
        while len(values) < 10:
            x, new_rs, (old_size, indices_selected) = RandomSampling(red)([data, is_track, row_splits])
            backgathered = MultiBackScatter()([x, [(old_size, indices_selected)]])

            if data.shape != backgathered.shape:
                print("Failed for reduction", red)
                print("Original shape", data.shape)
                print("New shape", backgathered.shape)
                passed = False

            reduced = x.shape[0]
            full = data.shape[0]
            values.append(reduced / full)
        ax[i].set_title(f"Reduction {red}")
        ax[i].hist(values, bins=10)

    if passed:
        print("Passed all tests")
    else:
        print("Failed some tests")

    print()
    print("Testing row splits")
    for i in range(len(row_splits)-1):
        start = row_splits[i]
        stop = row_splits[i+1]

        if not data[start][-1] == 0:
            pdb.set_trace()
            print("Some error with row splits")

        is_zero = backgathered[start:stop, -1] == 0
        is_offset = backgathered[start:stop, -1] - np.arange(0, stop-start) == 0
        is_either = np.logical_or(is_zero, is_offset)
        pdb.set_trace()
        if not np.all(is_either):
            pdb.set_trace()
            print(i)
            print("Some error with row splits")


    fig.savefig("reductions.png")





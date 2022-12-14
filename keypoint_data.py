import os

import numpy as np
from keras.utils import to_categorical

from data_collection import DATA_PATH , gestures, no_sequence, sequence_length


def append_data():
    label_map = {label: num for num, label in enumerate(gestures)}
    sequences, labels = [], []
    for gesture in gestures:
        for sequence in range(no_sequence):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, gesture, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[gesture])

    return np.array(sequences), to_categorical(labels).astype(int)

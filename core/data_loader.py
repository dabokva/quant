import stim
import numpy as np
from pathlib import Path
import tensorflow as tf

def make_dataset(data_path):

    circuit = stim.Circuit.from_file(data_path / 'circuit_ideal.stim')

    detection_events = stim.read_shot_data_file(
        path=data_path / 'detection_events.b8',
        format='b8',
        num_detectors=circuit.num_detectors).astype(np.float32)

    observable_flips = stim.read_shot_data_file(
        path=data_path / 'obs_flips_actual.b8',
        format='b8',
        num_observables=circuit.num_observables).astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((detection_events, observable_flips))

    return dataset
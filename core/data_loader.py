from pathlib import Path

import stim
import tensorflow as tf


def load_experimental_shots(data_path):
    data_path = Path(data_path)
    circuit = stim.Circuit.from_file(str(data_path / 'circuit_ideal.stim'))

    detection_events = stim.read_shot_data_file(
        path=data_path / 'detection_events.b8',
        format='b8',
        num_detectors=circuit.num_detectors)

    observable_flips = stim.read_shot_data_file(
        path=data_path / 'obs_flips_actual.b8',
        format='b8',
        num_observables=circuit.num_observables)

    if len(detection_events) != len(observable_flips):
        raise ValueError('Detector events and observable labels have different shot counts')
    if len(detection_events) == 0:
        raise ValueError('The experimental dataset is empty')

    return tf.data.Dataset.from_tensor_slices((detection_events, observable_flips))


def make_dataset(data_path):
    return load_experimental_shots(data_path)

import stim
import numpy as np
from pathlib import Path
import tensorflow as tf

def load_circuit(data_path):
    circuit = stim.Circuit.from_file(data_path / 'circuit_ideal.stim')
    return circuit

def make_dataset(data_path):

    circuit = load_circuit(data_path)

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


def make_3d_mapper(circuit):
    coordinate_map = circuit.get_detector_coordinates()

    coordinates = []

    for detector_index in range(circuit.num_detectors):
        coordinate = coordinate_map[detector_index]

        if len(coordinate) < 3:
            raise ValueError(f'Detector {detector_index} has no x, y, t coordinate')
        coordinates.append(coordinate[:3])

    coordinates = np.asarray(coordinates, dtype=np.float32)

    xs = coordinates[:, 0]
    ys = coordinates[:, 1]
    ts = coordinates[:, 2]

    if not np.allclose(xs, np.round(xs)):
        raise ValueError('Detector x coordinates are not integers')

    if not np.allclose(ys, np.round(ys)):
        raise ValueError('Detector y coordinates are not integers')

    xs = np.round(xs).astype(np.int32)
    ys = np.round(ys).astype(np.int32)

    x_min = xs.min()
    y_min = ys.min()

    x_indices = xs - x_min
    y_indices = ys - y_min

    time_values = np.unique(ts)

    time_to_index = {time: index for index, time in enumerate(time_values)}

    time_indices = np.asarray([time_to_index[time] for time in ts], dtype=np.int32)

    num_times = len(time_values)
    height = int(y_indices.max() + 1)
    width = int(x_indices.max() + 1)

    volume_shape = (num_times, height, width)

    scatter_indices_np = np.column_stack([
        time_indices,
        y_indices,
        x_indices,
    ]).astype(np.int32)

    unique_positions = np.unique(scatter_indices_np, axis=0)

    if len(unique_positions) != circuit.num_detectors:
        raise ValueError('Multiple detectors occupy the same t, y, x position')

    scatter_indices = tf.constant(scatter_indices_np, dtype=tf.int32)

    detector_mask = tf.scatter_nd(
        indices=scatter_indices,
        updates=tf.ones(circuit.num_detectors, dtype=tf.float32),
        shape=volume_shape,
    )

    def map_shot(detector_events, logical_flip):
        detector_events = tf.cast(detector_events, tf.float32)
        event_volume = tf.scatter_nd(indices=scatter_indices, updates=detector_events, shape=volume_shape)
        cnn_input = tf.stack([event_volume, detector_mask], axis=-1,)
        return cnn_input, logical_flip

    print(f'Detectors: {circuit.num_detectors}')
    print(f'Time slices: {num_times}')
    print(f'Height: {height}')
    print(f'Width: {width}')
    print('Input shape per shot: ' f'({num_times}, {height}, {width}, 2)')

    return map_shot
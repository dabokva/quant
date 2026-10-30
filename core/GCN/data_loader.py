import numpy as np
import tensorflow as tf

PARTITION_NAMES = ('train', 'validation', 'calibration', 'test')

def split_dataset(dataset, random_seed=73, calibration_fraction=0.0):
    num_shots = int(dataset.cardinality().numpy())
    indices = partition_indices(num_shots, random_seed, calibration_fraction)
    return datasets_from_indices(dataset, indices), indices


def partition_indices(num_shots, random_seed=73, calibration_fraction=0.0):
    if num_shots <= 0:
        raise ValueError('Dataset must have a known, positive shot count')
    if not 0.0 <= calibration_fraction < 0.1:
        raise ValueError('calibration_fraction must be in [0, 0.1)')
    num_train = int(0.8 * num_shots)
    num_development = int(0.1 * num_shots)
    num_calibration = int(calibration_fraction * num_shots)
    counts = {
        'train': num_train,
        'validation': num_development - num_calibration,
        'calibration': num_calibration,
        'test': num_shots - num_train - num_development,
    }
    if min(counts[name] for name in ('train', 'validation', 'test')) <= 0:
        raise ValueError('Not enough shots for nonempty training, validation and test sets')
    if calibration_fraction > 0 and num_calibration == 0:
        raise ValueError('Requested calibration fraction produces an empty subset')

    order = np.random.default_rng(random_seed).permutation(num_shots)
    indices = {}
    offset = 0
    for name in PARTITION_NAMES:
        count = counts[name]
        indices[name] = order[offset:offset + count]
        offset += count
    validate_shot_indices(indices, num_shots)
    return indices


def validate_shot_indices(indices, num_shots):
    seen = set()
    for name in PARTITION_NAMES:
        values = np.asarray(indices[name])
        if values.ndim != 1 or values.dtype.kind not in 'iu':
            raise ValueError(f'{name} shot indices must be a one-dimensional integer array')
        if name != 'calibration' and len(values) == 0:
            raise ValueError(f'{name} subset is empty')
        if np.any(values < 0) or np.any(values >= num_shots):
            raise ValueError(f'{name} shot indices are out of range')
        ids = set(values.tolist())
        if len(ids) != len(values):
            raise ValueError(f'{name} contains duplicate shot indices')
        if seen.intersection(ids):
            raise ValueError(f'{name} overlaps another partition')
        seen.update(ids)
    if len(seen) != num_shots:
        raise ValueError('Shot partitions must cover the complete experimental collection')


def prepare_dataset(dataset, mapper, batch_size=32, training=False, random_seed=73):
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    if training:
        count = int(dataset.cardinality().numpy())
        dataset = dataset.shuffle(
            min(count, 10_000), seed=random_seed, reshuffle_each_iteration=True,
        )
    return dataset.batch(batch_size).map(mapper, num_parallel_calls=1).prefetch(1)


def datasets_from_indices(dataset, indices):
    count = int(dataset.cardinality().numpy())
    validate_shot_indices(indices, count)
    # One compact batch avoids Python objects per shot and repeated shuffle buffers.
    events, labels = next(iter(dataset.batch(count)))
    return {name: tf.data.Dataset.from_tensor_slices(
        (tf.gather(events, values), tf.gather(labels, values)))
        for name, values in indices.items()}

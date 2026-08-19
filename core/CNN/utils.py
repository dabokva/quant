import numpy as np
import tensorflow as tf

from pathlib import Path
from core.CNN.data_loader import load_circuit, make_dataset, make_3d_mapper


random_seed = 73


data_path = Path('./data/sycamore_surface_code_d3_d5/sample_00/d5_at_q6_5/X/r25')
circuit = load_circuit(data_path)

flat_dataset = make_dataset(data_path)
flat_dataset = flat_dataset.map(lambda events, label: (tf.cast(events, tf.float32), tf.cast(label, tf.float32)), num_parallel_calls=tf.data.AUTOTUNE)
num_shots = int(flat_dataset.cardinality().numpy())
if num_shots <= 0: raise ValueError('Dataset cardinality is unknown or the dataset is empty')

flat_dataset = flat_dataset.shuffle(
    buffer_size=num_shots,
    seed=random_seed,
    reshuffle_each_iteration=False,
)

num_train = int(0.8 * num_shots)
num_validation = int(0.1 * num_shots)
num_test = num_shots - num_train - num_validation

train_flat = flat_dataset.take(num_train)
remaining = flat_dataset.skip(num_train)
validation_flat = remaining.take(num_validation)
test_flat = remaining.skip(num_validation)

print(f'Training shots: {num_train}')
print(f'Validation shots: {num_validation}')
print(f'Test shots: {num_test}')

map_to_3d = make_3d_mapper(circuit)


batch_size = 256


train_dataset = (
    train_flat.map(map_to_3d, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(
        buffer_size=min(num_train, 10_000),
        seed=random_seed,
        reshuffle_each_iteration=True,
    )
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)


labels = np.concatenate([label.numpy().reshape(-1) for _, label in train_dataset])

positive_rate = labels.mean()
majority_accuracy = max(positive_rate, 1 - positive_rate)

print(f'Positive rate: {positive_rate:.4f}')
print(f'Majority baseline: {majority_accuracy:.4f}')

for volumes, labels in train_dataset.take(1):
    events = volumes[..., 0]
    masks = volumes[..., 1]

    events_per_shot = tf.reduce_sum(
        events,
        axis=(1, 2, 3),
    )

    print('Volume shape:', volumes.shape)
    print('Label mean:', tf.reduce_mean(labels).numpy())
    print('Mask values:', tf.unique(tf.reshape(masks, [-1])).y.numpy())
    print(
        'Mean events per shot:',
        tf.reduce_mean(events_per_shot).numpy(),
    )
    print(
        'Minimum events:',
        tf.reduce_min(events_per_shot).numpy(),
    )
    print(
        'Maximum events:',
        tf.reduce_max(events_per_shot).numpy(),
    )

    invalid_events = events * (1 - masks)

    print(
        'Events outside mask:',
        tf.reduce_sum(invalid_events).numpy(),
    )
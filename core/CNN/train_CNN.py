import numpy as np
import tensorflow as tf
from tensorflow.keras import metrics, losses, callbacks, optimizers

from pathlib import Path
from core.CNN.data_loader import load_circuit, make_dataset, make_3d_mapper
from core.CNN.CNN import CNN_Predecoder


random_seed = 73
tf.keras.utils.set_random_seed(random_seed)


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

validation_dataset = (
    validation_flat
    .map(map_to_3d, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

test_dataset = (
    test_flat
    .map(map_to_3d, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

overfit_dataset = (
    train_dataset
    .unbatch()
    .take(512)
    .cache()
    .shuffle(512, seed=42)
    .batch(64)
)


input_shape = train_dataset.element_spec[0].shape[1:]
CNN = CNN_Predecoder()
CNN(tf.zeros(shape=(1, *input_shape), dtype=tf.float32))

CNN.compile(
    optimizer = optimizers.Adam(learning_rate=1e-3,),
    loss = losses.BinaryCrossentropy(),
    metrics = [metrics.BinaryAccuracy(name='accuracy')]
)

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
    min_delta=1e-5,
    restore_best_weights=True,
    verbose=1,
)
terminate_on_nan = (callbacks.TerminateOnNaN())

history = CNN.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=200,
    callbacks=[
        early_stopping,
        terminate_on_nan,
    ],
    verbose=1,
)

test_results = CNN.evaluate(
    test_dataset,
    return_dict=True,
    verbose=1,
)

print('\nTest results')

for metric, value in test_results.items():
    print(f'{metric}: {value:.6f}')
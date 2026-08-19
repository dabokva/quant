
from pathlib import Path
import time

import numpy as np
import tensorflow as tf

from core.data_loader import make_dataset
from core.baseline.mlp_baseline import make_mlp


data_path = Path('./data/sycamore_surface_code_d3_d5/sample_00/d5_at_q6_5/X/r25')

model_dir = Path('./scripts/baseline')

batch_size = 512
validation_fraction = 0.2
random_seed = 73

num_warmup_runs = 100
num_latency_runs = 1000

tf.keras.utils.set_random_seed(random_seed)

full_dataset = make_dataset(data_path)

full_dataset = full_dataset.map(
    lambda x, y: (
        tf.cast(x, tf.float32),
        tf.cast(y, tf.float32),
    ), num_parallel_calls=tf.data.AUTOTUNE)

num_detectors = int(full_dataset.element_spec[0].shape[0])
num_shots = int(full_dataset.cardinality().numpy())

full_dataset = full_dataset.shuffle(
    buffer_size=num_shots,
    seed=random_seed,
    reshuffle_each_iteration=False,
)

num_validation = int(validation_fraction * num_shots)
num_train = num_shots - num_validation

train_dataset = full_dataset.take(num_train)
validation_dataset = full_dataset.skip(num_train)

train_dataset = (
    train_dataset
    .shuffle(
        buffer_size=num_train,
        seed=random_seed,
        reshuffle_each_iteration=True,
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE))


validation_dataset = (
    validation_dataset
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)


print(f'Training shots: {num_train}')
print(f'Validation shots: {num_validation}')
print(f'Number of detectors: {num_detectors}')


model = make_mlp(num_detectors)
model.summary()


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    min_delta=1e-4,
    restore_best_weights=True,
    verbose=1,
)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=500,
    callbacks=[early_stopping],
)

validation_results = model.evaluate(
    validation_dataset,
    return_dict=True,
    verbose=1,
)

print('\nValidation results')
for metric, value in validation_results.items():
    print(f'{metric}: {value:.6f}')

model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / 'mlp_d3_r05.keras'

model.save(str(model_path))

print(f'\nModel saved to: {model_path.resolve()}')

x_sample, _ = next(iter(validation_dataset.unbatch()))
x_sample = tf.expand_dims(x_sample, axis=0)

@tf.function(input_signature=[tf.TensorSpec(shape=(1, num_detectors), dtype=tf.float32)])

def infer_single_shot(x):
    return model(x, training=False)

for _ in range(num_warmup_runs):
    infer_single_shot(x_sample).numpy()

latencies_us = np.empty(num_latency_runs, dtype=np.float64)

for index in range(num_latency_runs):
    start_ns = time.perf_counter_ns()
    probability = infer_single_shot(x_sample)

    probability.numpy()

    end_ns = time.perf_counter_ns()

    latencies_us[index] = (end_ns - start_ns) / 1000

print('\nSingle-shot GPU inference latency')
print(f'Mean: {np.mean(latencies_us):.3f} us')
print(f'Median: {np.median(latencies_us):.3f} us')
print(f'P95: {np.percentile(latencies_us, 95):.3f} us')
print(f'P99: {np.percentile(latencies_us, 99):.3f} us')
print(f'Minimum: {np.min(latencies_us):.3f} us')
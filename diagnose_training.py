from pathlib import Path
import json
import numpy as np
import tensorflow as tf
import stim

from core.GCN.GCN import GCNPredecoder
from core.GCN.graph_data import FEATURE_NAMES, make_feature_mapper_from_static
from core.GCN.metrics import probability_metrics
from core.GCN.run_io import read_configuration, read_indices, file_sha256

SETTINGS = {
    'run_directory': 'experiments/sycamore_benchmark_fixed120k/fresh/batch_64/baseline/collection_gcn_20261013_152010_212105',
    'checkpoint': 'best.weights.h5',
    'sample_index': 0,
    'num_shots': 128,
    'batch_size': 4,
    'seed': 73,
}


def diagnose(run_directory, checkpoint='best.weights.h5', sample_index=0,
             num_shots=128, batch_size=4, seed=73):
    run = Path(run_directory).resolve()
    config = read_configuration(run)
    if not config.get('collection'):
        raise ValueError('This diagnostic expects a collection run')
    if num_shots < 1 or batch_size < 1:
        raise ValueError('num_shots and batch_size must be positive')
    sample = config['collection']['samples'][sample_index]
    data = Path(config['data_root']) / sample['relative_path']
    checkpoint_path = run / checkpoint
    if not checkpoint_path.is_file():
        raise ValueError(f'Checkpoint not found: {checkpoint_path}')
    checkpoint_hash = file_sha256(checkpoint_path)
    for name in ('detection_events.b8', 'obs_flips_actual.b8'):
        if file_sha256(data/name) != sample['sha256'][name]:
            raise ValueError(f'Recorded data changed: {data/name}')
    indices = read_indices(run, config['collection']['total_shots'])
    ids = indices['validation']
    ids = ids[(ids >= sample['offset']) & (ids < sample['offset']+sample['shots'])]
    if len(ids) == 0:
        raise ValueError('This acquisition has no validation shots')
    all_labels = stim.read_shot_data_file(path=data/'obs_flips_actual.b8', format='b8', num_observables=1)[:, 0]
    validation_positive_fraction = float(all_labels[ids-sample['offset']].mean())
    ids = np.sort(np.random.default_rng(seed).choice(ids, min(num_shots, len(ids)), replace=False))
    local = ids - sample['offset']
    detectors = sample['number_of_detectors']
    packed = np.memmap(data/'detection_events.b8', dtype=np.uint8, mode='r',
                       shape=(sample['shots'], (detectors+7)//8))
    events = np.unpackbits(packed[local], axis=1, count=detectors, bitorder='little')
    labels = all_labels[local]
    with np.load(run/sample['graph_file'], allow_pickle=False) as archive:
        if archive['feature_names'].tolist() != FEATURE_NAMES:
            raise ValueError('Unexpected feature layout')
        static = archive['static_node_features']
        edges, weights = archive['edge_index'], archive['edge_weights']
    mapper = make_feature_mapper_from_static(static, detectors)
    model = GCNPredecoder(edges, weights, hidden_units=config['hidden_units'],
        number_of_gcn_layers=config['number_of_gcn_layers'], dense_units=config['dense_units'],
        dense2_units=config.get('dense2_units'), dropout_rate=config['dropout_rate'],
        precision=config.get('precision', 'float32'))
    model(tf.zeros((1, len(static), len(FEATURE_NAMES)), tf.float32))
    model.load_weights(str(checkpoint_path))
    if file_sha256(checkpoint_path) != checkpoint_hash:
        raise RuntimeError('Checkpoint changed while loading; rerun after checkpoint writing finishes')

    @tf.function(input_signature=[tf.TensorSpec((None, detectors), tf.float32)])
    def predict(batch):
        features, _ = mapper(batch, tf.zeros((tf.shape(batch)[0], 1)))
        return model(features, training=False)

    predictions, zero_predictions = [], []
    for start in range(0, len(events), batch_size):
        batch = tf.convert_to_tensor(events[start:start+batch_size], tf.float32)
        features, _ = mapper(batch, tf.zeros((len(batch), 1)))
        np.testing.assert_array_equal(features[:, :detectors, 0].numpy(), batch.numpy())
        np.testing.assert_array_equal(features[:, detectors:, 0].numpy(), 0)
        predictions.extend(predict(batch).numpy().ravel())
        zero_predictions.extend(predict(tf.zeros_like(batch)).numpy().ravel())
    p = np.asarray(predictions, dtype=np.float64)
    zero = np.asarray(zero_predictions, dtype=np.float64)
    stats = probability_metrics(labels, p)
    auc = tf.keras.metrics.AUC()
    auc.update_state(labels, p)
    return {
        'checkpoint': str(checkpoint_path), 'checkpoint_sha256': checkpoint_hash,
        'sample': sample['name'], 'subset': 'validation', 'shots': len(ids),
        'full_acquisition_validation_positive_fraction': validation_positive_fraction,
        'sampled_positive_fraction': float(labels.mean()),
        'sampled_accuracy': stats['accuracy'], 'sampled_bce': stats['bce'],
        'sampled_exact_roc_auc': stats['roc_auc'],
        'sampled_keras_roc_auc': float(auc.result()) if stats['roc_auc'] is not None else None,
        'probability_min': float(p.min()), 'probability_max': float(p.max()),
        'probability_mean': float(p.mean()), 'probability_std': float(p.std()),
        'predicted_positive_fraction': float((p > .5).mean()),
        'unique_detector_records': len(np.unique(events, axis=0)),
        'detector_event_fraction': float(events.mean()),
        'event_feature_matches_records': True,
        'mean_abs_probability_change_when_events_zeroed': float(np.abs(p-zero).mean()),
        'max_abs_probability_change_when_events_zeroed': float(np.abs(p-zero).max()),
        'note': 'Small sample from one acquisition; not a final performance estimate. '
                'Zero-event inputs probe sensitivity and are not a realistic performance baseline.',
    }


if __name__ == '__main__':
    tf.config.set_visible_devices([], 'GPU')
    options = dict(SETTINGS)
    path = Path(options['run_directory']).expanduser()
    options['run_directory'] = path if path.is_absolute() else Path(__file__).resolve().parent/path
    with tf.device('/CPU:0'):
        result = diagnose(**options)
    print(json.dumps(result, indent=2, allow_nan=False))

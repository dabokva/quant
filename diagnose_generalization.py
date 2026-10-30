from datetime import datetime
from pathlib import Path
import math
import time

import numpy as np
import stim
import tensorflow as tf

from core.GCN.GCN import GCNPredecoder
from core.GCN.graph_data import FEATURE_NAMES, make_feature_mapper_from_static
from core.GCN.metrics import probability_metrics
from core.GCN.run_io import read_configuration, read_indices, file_sha256, write_json, select_device

SETTINGS = {
    'run_directory': '/work/experiments/sycamore_benchmark_fixed120k/fresh/batch_64/baseline/collection_gcn_20261013_152010_212105',
    'output_root': 'diagnostics/generalization',
    'sample_index': 0,
    'training_shots': 2000,  # 1,000 of each class, all from the reserved training split.
    'batch_size': 16,
    'epochs': 20,            # Full passes over these same 2,000 shots. No early stopping.
    # 'learning_rate': 0.001,
    'learning_rate': 0.0001,
    'seed': 73,
    'device': 'gpu',
}


def select_training_ids(train_ids, sample, labels, count, seed):
    if type(count) is not int or count < 2 or count % 2:
        raise ValueError('training_shots must be an even integer of at least two')
    ids = np.asarray(train_ids, dtype=np.int64)
    ids = ids[(ids >= sample['offset']) & (ids < sample['offset']+sample['shots'])]
    rng = np.random.default_rng(seed)
    chosen = []
    for label in (0, 1):
        group = ids[labels[ids-sample['offset']] == label]
        if len(group) < count//2:
            raise ValueError(f'{sample["name"]} has fewer than {count//2} training shots of label {label}')
        chosen.extend(rng.choice(group, count//2, replace=False))
    return np.sort(np.asarray(chosen, dtype=np.int64))


def diagnose(run_directory, output_root, sample_index=0, training_shots=2000,
             batch_size=16, epochs=50, learning_rate=.001, seed=73, device='gpu'):
    for name, value in [('batch_size', batch_size), ('epochs', epochs)]:
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('learning_rate must be finite and positive')
    run = Path(run_directory).expanduser().resolve()
    root = Path(output_root).expanduser().resolve()
    if root == run or run in root.parents:
        raise ValueError('Diagnostic output must be outside the source run directory')
    config = read_configuration(run)
    if not config.get('collection'):
        raise ValueError('This diagnostic requires a collection run with saved partitions')
    sample = config['collection']['samples'][sample_index]
    data = Path(config['data_root'])/sample['relative_path']
    print(f'Checking recorded data and graph for {sample["name"]}...', flush=True)
    for name in ('detection_events.b8', 'obs_flips_actual.b8'):
        if file_sha256(data/name) != sample['sha256'][name]:
            raise ValueError(f'Data changed since the source run: {data/name}')
    graph_path = run/sample['graph_file']
    if file_sha256(graph_path) != sample['graph_sha256']:
        raise ValueError('Acquisition graph changed since the source run')
    partitions = read_indices(run, config['collection']['total_shots'])
    all_labels = stim.read_shot_data_file(path=data/'obs_flips_actual.b8', format='b8', num_observables=1)[:, 0]
    train_ids = select_training_ids(partitions['train'], sample, all_labels, training_shots, seed)
    val_ids = partitions['validation']
    val_ids = val_ids[(val_ids >= sample['offset']) & (val_ids < sample['offset']+sample['shots'])]
    if not len(val_ids):
        raise ValueError('The selected acquisition has no reserved validation shots')
    if np.intersect1d(train_ids, val_ids).size:
        raise ValueError('Training and validation identifiers overlap')
    detectors = sample['number_of_detectors']
    packed = np.memmap(data/'detection_events.b8', mode='r', dtype=np.uint8,
                       shape=(sample['shots'], (detectors+7)//8))
    def records(ids):
        return (np.unpackbits(packed[ids-sample['offset']], axis=1, count=detectors,
                              bitorder='little').astype(np.float32), all_labels[ids-sample['offset']])
    train_events, train_labels = records(train_ids)
    val_events, val_labels = records(val_ids)
    with np.load(graph_path, allow_pickle=False) as saved:
        if saved['feature_names'].tolist() != FEATURE_NAMES:
            raise ValueError('Unexpected feature layout')
        static, edges, coefficients = (saved[k] for k in ('static_node_features', 'edge_index', 'edge_weights'))
    chosen_device = select_device(device)
    tf.keras.utils.set_random_seed(seed)
    output = root/(sample['name']+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'diagnostic_shot_indices.npz', train=train_ids, validation=val_ids)
    report = {
        'source_run': str(run), 'source_graph_sha256': sample['graph_sha256'], 'sample': sample['name'],
        'initialization': 'fresh', 'architecture': {k: config.get(k) for k in
            ('hidden_units', 'number_of_gcn_layers', 'dense_units', 'dense2_units')},
        'training_shots': len(train_ids), 'training_positive_fraction': float(train_labels.mean()),
        'validation_shots': len(val_ids), 'validation_positive_fraction': float(val_labels.mean()),
        'training_selection': 'fixed balanced subset of reserved training partition',
        'validation_selection': 'complete original validation partition for this acquisition',
        'batch_size': batch_size, 'maximum_epochs': epochs, 'learning_rate': learning_rate,
        'dropout': 0., 'precision': 'float32', 'seed': seed, 'device': chosen_device,
        'updates_per_epoch': math.ceil(len(train_ids)/batch_size),
        'note': 'Exploratory single-acquisition diagnostic. Validation has no gradient updates. '
                'No calibration or test records are evaluated; not a final benchmark result.',
        'history': [], 'status': 'running',
    }
    write_json(output/'generalization_report.json', report)
    print(f'Diagnostic output: {output}\n{len(train_ids):,} fixed training shots, '
          f'{len(val_ids):,} separate validation shots; dropout=0; fresh model; '
          f'{epochs} epochs, {report["updates_per_epoch"]} updates/epoch.', flush=True)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    with tf.device(chosen_device):
        mapper = make_feature_mapper_from_static(static, detectors)
        model = GCNPredecoder(edges, coefficients, hidden_units=config['hidden_units'],
            number_of_gcn_layers=config['number_of_gcn_layers'], dense_units=config['dense_units'],
            dense2_units=config.get('dense2_units'), dropout_rate=0., precision='float32')
        model(tf.zeros((1, len(static), len(FEATURE_NAMES)), tf.float32))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))
        initial = [v.numpy().copy() for v in model.trainable_variables]

        @tf.function(input_signature=[tf.TensorSpec((None, detectors), tf.float32),
                                      tf.TensorSpec((None, 1), tf.float32)])
        def update(events, labels):
            features, _ = mapper(events, labels)
            logs = model.train_step((features, labels))
            tf.debugging.assert_all_finite(logs['loss'], 'Non-finite training loss')
            return logs['loss']

        @tf.function(input_signature=[tf.TensorSpec((None, detectors), tf.float32)])
        def predict(events):
            features, _ = mapper(events, tf.zeros((tf.shape(events)[0], 1)))
            return model(features, training=False)

        def measure(events, labels):
            p = np.concatenate([predict(events[i:i+batch_size]).numpy().ravel()
                                for i in range(0, len(events), batch_size)])
            metrics = probability_metrics(labels, p)
            return p, {k: metrics[k] for k in ('shots', 'accuracy', 'bce', 'roc_auc')} | {
                'probability_min': float(p.min()), 'probability_max': float(p.max()),
                'probability_std': float(p.std()), 'predicted_positive_fraction': float((p > .5).mean())}

        def record(epoch, epoch_started):
            train_p, train_metrics = measure(train_events, train_labels)
            val_p, val_metrics = measure(val_events, val_labels)
            delta = [float(np.linalg.norm(v.numpy()-old)) for v, old in zip(model.trainable_variables, initial)]
            row = {'epoch': epoch, 'optimizer_updates': int(model.optimizer.iterations.numpy()),
                   'train': train_metrics, 'validation': val_metrics,
                   'weight_change_l2': float(np.linalg.norm(delta)),
                   'elapsed_seconds': time.perf_counter()-started,
                   'epoch_seconds_including_evaluation': time.perf_counter()-epoch_started}
            report['history'].append(row)
            for name, ids, labels, p in [('train', train_ids, train_labels, train_p),
                                        ('validation', val_ids, val_labels, val_p)]:
                np.savez_compressed(output/f'{name}_predictions.npz', epoch=epoch,
                                    shot_indices=ids, labels=labels, probabilities=p)
            write_json(output/'generalization_report.json', report)
            val_auc = '--' if val_metrics['roc_auc'] is None else f'{val_metrics["roc_auc"]:.3f}'
            print(f'Epoch {epoch:2d}/{epochs}: '
                  f'train loss={train_metrics["bce"]:.6f} acc={train_metrics["accuracy"]:.3f} '
                  f'AUC={train_metrics["roc_auc"]:.3f} | '
                  f'val loss={val_metrics["bce"]:.6f} acc={val_metrics["accuracy"]:.3f} '
                  f'AUC={val_auc} | p_std={train_metrics["probability_std"]:.3g}/'
                  f'{val_metrics["probability_std"]:.3g} '
                  f'updates={row["optimizer_updates"]} elapsed={row["elapsed_seconds"]:.0f}s', flush=True)

        try:
            record(0, started)
            for epoch in range(1, epochs+1):
                epoch_started = time.perf_counter()
                model.reset_metrics()
                order = rng.permutation(len(train_ids))
                for start in range(0, len(order), batch_size):
                    positions = order[start:start+batch_size]
                    update(train_events[positions], train_labels[positions, None].astype(np.float32))
                record(epoch, epoch_started)
            model.save_weights(str(output/'final.weights.h5'))
            report['status'] = 'complete'
        except KeyboardInterrupt:
            report['status'] = 'interrupted'
        except Exception as error:
            report.update(status='failed', error=str(error))
            write_json(output/'generalization_report.json', report)
            raise
        write_json(output/'generalization_report.json', report)
    print(f'{report["status"]}: {output/"generalization_report.json"}', flush=True)
    return output, report


if __name__ == '__main__':
    options = dict(SETTINGS)
    if options['device'] == 'cpu':
        tf.config.set_visible_devices([], 'GPU')
    else:
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    for key in ('run_directory', 'output_root'):
        path = Path(options[key]).expanduser()
        options[key] = path if path.is_absolute() else Path(__file__).resolve().parent/path
    diagnose(**options)

from datetime import datetime
from pathlib import Path
import json
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
    'output_root': 'diagnostics/overfit',
    'sample_index': 0,
    'num_shots': 64,
    'batch_size': 16,
    'steps': 1000,
    'log_every': 50,
    'learning_rate': 0.001,
    'seed': 73,
    'device': 'gpu',
    'checkpoint': None,
}


def choose_shots(training_ids, sample, all_labels, count, seed):
    if type(count) is not int or count < 2 or count % 2:
        raise ValueError('num_shots must be a positive even integer of at least two')
    ids = np.asarray(training_ids, dtype=np.int64)
    ids = ids[(ids >= sample['offset']) & (ids < sample['offset']+sample['shots'])]
    rng = np.random.default_rng(seed)
    selected = []
    for label in (0, 1):
        candidates = ids[all_labels[ids-sample['offset']] == label]
        if len(candidates) < count//2:
            raise ValueError(f'Not enough reserved training shots of label {label} in {sample["name"]}')
        selected.extend(rng.choice(candidates, count//2, replace=False))
    return np.sort(np.asarray(selected, dtype=np.int64))


def diagnose(run_directory, output_root, sample_index=0, num_shots=64, batch_size=16,
             steps=1000, log_every=50, learning_rate=.001, seed=73, device='gpu', checkpoint=None):
    for name, value in [('steps', steps), ('batch_size', batch_size), ('log_every', log_every)]:
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('learning_rate must be finite and positive')
    run = Path(run_directory).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    if destination == run or run in destination.parents:
        raise ValueError('Diagnostic output must be outside the original run directory')
    config = read_configuration(run)
    if not config.get('collection'):
        raise ValueError('Select a collection run with saved shot partitions')
    sample = config['collection']['samples'][sample_index]
    data = Path(config['data_root'])/sample['relative_path']
    print(f'Checking {sample["name"]}; selecting TRAINING shots only...', flush=True)
    for name in ('detection_events.b8', 'obs_flips_actual.b8'):
        if file_sha256(data/name) != sample['sha256'][name]:
            raise ValueError(f'Data changed since training: {data/name}')
    graph_path = run/sample['graph_file']
    if file_sha256(graph_path) != sample['graph_sha256']:
        raise ValueError('Saved acquisition graph changed since training')
    labels_all = stim.read_shot_data_file(path=data/'obs_flips_actual.b8', format='b8', num_observables=1)[:, 0]
    indices = read_indices(run, config['collection']['total_shots'])
    ids = choose_shots(indices['train'], sample, labels_all, num_shots, seed)
    labels = labels_all[ids-sample['offset']]
    detectors = sample['number_of_detectors']
    packed = np.memmap(data/'detection_events.b8', dtype=np.uint8, mode='r',
                       shape=(sample['shots'], (detectors+7)//8))
    events = np.unpackbits(packed[ids-sample['offset']], axis=1, count=detectors, bitorder='little').astype(np.float32)
    _, inverse = np.unique(events, axis=0, return_inverse=True)
    conflicting = sum(len(np.unique(labels[inverse == group])) > 1 for group in np.unique(inverse))
    with np.load(graph_path, allow_pickle=False) as archive:
        if archive['feature_names'].tolist() != FEATURE_NAMES:
            raise ValueError('Unexpected feature layout')
        static = archive['static_node_features']
        edges, coefficients = archive['edge_index'], archive['edge_weights']
    tf.keras.utils.set_random_seed(seed)
    with tf.device(select_device(device)):
        mapper = make_feature_mapper_from_static(static, detectors)
        model = GCNPredecoder(edges, coefficients, hidden_units=config['hidden_units'],
            number_of_gcn_layers=config['number_of_gcn_layers'], dense_units=config['dense_units'],
            dense2_units=config.get('dense2_units'), dropout_rate=0., precision='float32')
        model(tf.zeros((1, len(static), len(FEATURE_NAMES)), tf.float32))
        source_hash = None
        if checkpoint is not None:
            path = run/checkpoint
            source_hash = file_sha256(path)
            model.load_weights(str(path))
            if source_hash != file_sha256(path):
                raise RuntimeError('Checkpoint changed during loading; rerun after its writer finishes')
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        variables = model.trainable_variables
        names = [getattr(v, 'path', v.name) for v in variables]
        initial = [v.numpy().copy() for v in variables]

        @tf.function(input_signature=[tf.TensorSpec((None, detectors), tf.float32)])
        def predict(batch):
            features, _ = mapper(batch, tf.zeros((tf.shape(batch)[0], 1)))
            return model(features, training=False)

        @tf.function(input_signature=[tf.TensorSpec((None, detectors), tf.float32),
                                      tf.TensorSpec((None, 1), tf.float32)])
        def update(batch, target):
            with tf.GradientTape() as tape:
                features, _ = mapper(batch, target)
                probability = model(features, training=True)
                loss = model.compute_loss_value(target, probability)
            gradients = tape.gradient(loss, variables)
            tf.debugging.assert_all_finite(loss, 'Non-finite training loss')
            pairs = []
            for gradient, variable in zip(gradients, variables):
                if gradient is not None:
                    tf.debugging.assert_all_finite(gradient, 'Non-finite gradient')
                    pairs.append((gradient, variable))
            if not pairs:
                raise ValueError('No gradients connect the loss to trainable variables')
            optimizer.apply_gradients(pairs)
            return (tf.linalg.global_norm([g for g, _ in pairs]),
                    tf.stack([tf.linalg.global_norm([g]) if g is not None else tf.constant(0.) for g in gradients]),
                    tf.constant([g is None for g in gradients]))

        output = destination/(sample['name']+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        output.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(output/'selected_training_shots.npz', shot_indices=ids,
                            local_shot_indices=ids-sample['offset'], labels=labels)
        report = {
            'source_run': str(run), 'sample': sample['name'], 'subset': 'train',
            'initialization': checkpoint or 'fresh', 'source_checkpoint_sha256': source_hash,
            'shots': len(ids), 'positive_shots': int(labels.sum()), 'seed': seed,
            'batch_size': batch_size, 'maximum_updates': steps, 'learning_rate': learning_rate,
            'dropout': 0., 'precision': 'float32', 'device': select_device(device),
            'number_of_detectors': detectors, 'number_of_nodes': len(static),
            'number_of_error_nodes': len(static)-detectors,
            'unique_detector_records': len(np.unique(inverse)),
            'conflicting_duplicate_records': int(conflicting),
            'interpretation': 'Memorization diagnostic on training data only; not a validation/test result. '
                              'Nonzero gradients/weight changes alone do not establish useful learning.',
            'history': [], 'status': 'running',
        }
        start_time = time.perf_counter()

        def record(step, gradient_info=None):
            p = np.concatenate([predict(events[i:i+batch_size]).numpy().ravel()
                                for i in range(0, len(events), batch_size)])
            metrics = probability_metrics(labels, p)
            delta = [float(np.linalg.norm(v.numpy()-old)) for v, old in zip(variables, initial)]
            row = {'step': step, 'elapsed_seconds': time.perf_counter()-start_time,
                   'loss': metrics['bce'], 'accuracy': metrics['accuracy'], 'roc_auc': metrics['roc_auc'],
                   'probability_min': float(p.min()), 'probability_max': float(p.max()),
                   'probability_std': float(p.std()), 'predicted_positive_fraction': float((p > .5).mean()),
                   'weight_change_l2': float(np.linalg.norm(delta)),
                   'weight_change_l2_by_variable': dict(zip(names, delta))}
            if gradient_info is not None:
                norm, per_variable, missing = gradient_info
                row.update(gradient_norm=float(norm),
                    gradient_norm_by_variable=dict(zip(names, per_variable.numpy().astype(float).tolist())),
                    variables_without_gradient=[name for name, absent in zip(names, missing.numpy()) if absent])
            report['history'].append(row)
            write_json(output/'overfit_report.json', report)
            np.savez_compressed(output/'training_predictions.npz', shot_indices=ids, labels=labels, probabilities=p)
            print(f'Step {step:4d}/{steps}: loss={row["loss"]:.6f} acc={row["accuracy"]:.3f} '
                  f'AUC={row["roc_auc"]:.3f} p=[{p.min():.6f}, {p.max():.6f}] '
                  f'std={p.std():.3g} grad={row.get("gradient_norm", 0):.3g} '
                  f'weight_change={row["weight_change_l2"]:.3g} '
                  f'elapsed={row["elapsed_seconds"]:.0f}s', flush=True)
            return row

        print(f'Diagnostic output: {output}\n{len(ids)} fixed training shots; '
              f'{int(labels.sum())} positive; dropout=0; initialization={checkpoint or "fresh"}', flush=True)
        rng = np.random.default_rng(seed)
        record(0)
        try:
            order, cursor = rng.permutation(len(ids)), 0
            for step in range(1, steps+1):
                if cursor == len(ids):
                    order, cursor = rng.permutation(len(ids)), 0
                positions = order[cursor:cursor+batch_size]
                cursor += len(positions)
                gradients = update(events[positions], labels[positions, None].astype(np.float32))
                if step == 1 or step % log_every == 0 or step == steps:
                    row = record(step, gradients)
                    if row['accuracy'] == 1. and row['loss'] < .05:
                        report['status'] = 'memorization_target_reached'
                        break
            else:
                report['status'] = 'update_limit_reached'
        except KeyboardInterrupt:
            report['status'] = 'interrupted'
        except Exception as error:
            report.update(status='failed', error=str(error))
            write_json(output/'overfit_report.json', report)
            raise
        write_json(output/'overfit_report.json', report)
        print(f'{report["status"]}: {output/"overfit_report.json"}', flush=True)
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
    _, result = diagnose(**options)
    print('Final training-only metrics:', json.dumps({k: result['history'][-1][k]
          for k in ('step', 'loss', 'accuracy', 'roc_auc', 'probability_std', 'weight_change_l2')}, indent=2))

from datetime import datetime
from pathlib import Path
import platform
import math
import time

import numpy as np
import stim
import tensorflow as tf

from core.GCN.collection import inspect_collection, save_graphs, read_collection, GraphBank, split_collection
from core.GCN.progress import Progress, track
from core.GCN.GCN import GCNPredecoder
from core.GCN.graph_data import FEATURE_NAMES
from core.GCN.run_io import write_json, read_configuration, read_indices, save_predictions, select_device


def training_schedule(train_ids, mode, epochs, fresh_shots_per_epoch, seed):
    if mode not in ('fixed', 'fresh') or epochs < 1 or fresh_shots_per_epoch < 1:
        raise ValueError('Invalid training mode or epoch settings')
    ids = np.asarray(train_ids, np.int64)
    if len(ids) == 0:
        raise ValueError('Training pool is empty')
    if mode == 'fixed':
        return ids, np.array([0, len(ids)], np.int64), epochs
    order = np.random.default_rng(seed).permutation(ids)
    epochs = min(epochs, math.ceil(len(order)/fresh_shots_per_epoch))
    stops = np.minimum(np.arange(epochs+1, dtype=np.int64)*fresh_shots_per_epoch, len(order))
    return order[:stops[-1]], stops, epochs


def fixed_training_subset(train_ids, manifest, limit, seed):
    ids = np.asarray(train_ids, dtype=np.int64)
    if limit is None:
        return ids
    if type(limit) is not int or limit < 1:
        raise ValueError('fixed_training_shots must be a positive integer or None')
    samples = manifest['samples']
    groups = [np.sort(ids[(ids >= s['offset']) & (ids < s['offset'] + s['shots'])])
              for s in samples]
    capacity = np.asarray([len(group) for group in groups], dtype=np.int64)
    if int(capacity.sum()) != len(ids):
        raise ValueError('Training IDs must belong to the acquisition collection')
    if not len(samples) or np.any(capacity == 0):
        raise ValueError('Each acquisition must contain reserved training shots')
    target = min(limit, len(ids))
    if target < len(samples):
        raise ValueError('fixed_training_shots must include at least one shot per acquisition')
    if target == len(ids):
        return ids
    quota = np.zeros(len(samples), dtype=np.int64)
    remaining = target
    while remaining:
        active = np.flatnonzero(quota < capacity)
        share, extra = divmod(remaining, len(active))
        if share == 0:
            quota[active[:extra]] += 1
            break
        added = np.minimum(capacity[active] - quota[active], share)
        quota[active] += added
        remaining -= int(added.sum())
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(group, int(count), replace=False)
                                   for group, count in zip(groups, quota)]))


def fit_collection(model, bank, events, labels, indices, run, epochs, batch_size, seed,
                   mode='fixed', fresh_shots_per_epoch=60000, schedule_seed=73,
                   early_stopping_patience=15, early_stopping_min_delta=1e-5,
                   fixed_training_shots=None):
    run = Path(run)
    if early_stopping_patience < 0 or not np.isfinite(early_stopping_min_delta) or early_stopping_min_delta < 0:
        raise ValueError('Invalid early-stopping settings')
    training_pool = (fixed_training_subset(indices['train'], bank.manifest, fixed_training_shots, schedule_seed)
                     if mode == 'fixed' else indices['train'])
    order, boundaries, limit = training_schedule(training_pool, mode, epochs, fresh_shots_per_epoch, schedule_seed)
    np.savez_compressed(run/'training_schedule.npz', order=order, boundaries=boundaries, mode=np.asarray(mode))
    sample_counts = {s['name']: int(np.count_nonzero((order >= s['offset']) & (order < s['offset']+s['shots'])))
                     for s in bank.manifest['samples']}
    print(f'Training schedule: mode={mode}; {len(order):,} scheduled unique shots '
          f'from {sum(count > 0 for count in sample_counts.values())} acquisitions; '
          f'{len(indices["train"]):,} reserved training shots', flush=True)
    history = tf.keras.callbacks.History()
    callback_list = tf.keras.callbacks.CallbackList([
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_patience,
                                        min_delta=early_stopping_min_delta, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(run/'best.weights.h5'), monitor='val_loss',
                                            save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.TerminateOnNaN(), history,
    ], model=model, epochs=limit, verbose=0)
    signature = [{'features': tf.TensorSpec((None, None, 13), tf.float32),
                  'edge_index': tf.TensorSpec((2, None), tf.int32),
                  'edge_weights': tf.TensorSpec((None,), tf.float32)}, tf.TensorSpec((None, 1), tf.float32)]
    @tf.function(input_signature=signature)
    def train_batch(x, y):
        return model.train_step((x, y))
    @tf.function(input_signature=signature)
    def validation_batch(x, y):
        return model.test_step((x, y))
    model.stop_training = False
    callback_list.on_train_begin()
    seen = np.zeros(len(labels), dtype=bool)
    epoch_records = []
    for epoch in range(limit):
        epoch_started = time.perf_counter()
        ids = order if mode == 'fixed' else order[boundaries[epoch]:boundaries[epoch+1]]
        callback_list.on_epoch_begin(epoch)
        model.reset_metrics()
        consumed = 0
        with Progress(len(ids), f'Train {epoch+1}/{limit}', unit='shot') as bar:
            for step, (positions, sample_index, batch) in enumerate(events[ids].batches(batch_size, seed+epoch)):
                callback_list.on_train_batch_begin(step)
                batch_ids = ids[positions]
                values = train_batch(bank.inputs(sample_index, batch), tf.convert_to_tensor(labels[batch_ids, None], tf.float32))
                logs = {key: float(value) for key, value in values.items()}
                seen[batch_ids] = True
                consumed += len(batch_ids)
                callback_list.on_train_batch_end(step, logs)
                bar.update(len(batch_ids))
                if model.stop_training:
                    break
        train_logs = {key: float(value) for key, value in model.metric_results().items()}
        model.reset_metrics()
        val_ids = indices['validation']
        for positions, sample_index, batch in track(events[val_ids].batches(batch_size), f'Validate {epoch+1}/{limit}',
                                                     total=len(val_ids), unit='shot', count=lambda item: len(item[0])):
            validation_batch(bank.inputs(sample_index, batch), tf.convert_to_tensor(labels[val_ids[positions], None], tf.float32))
        logs = {**train_logs, **{f'val_{key}': float(value) for key, value in model.metric_results().items()}}
        epoch_records.append({'epoch': epoch+1, 'training_shots': consumed,
                              'elapsed_seconds_including_validation': time.perf_counter()-epoch_started})
        callback_list.on_epoch_end(epoch, logs)
        write_json(run/'training_progress.json', {'completed_epochs': epoch+1, 'epochs': epoch_records,
                   'history': {k: list(map(float, v)) for k,v in history.history.items()},
                   'early_stopping': {'patience': early_stopping_patience, 'min_delta': early_stopping_min_delta}})
        print(f'Epoch {epoch+1}/{limit}: {consumed} training shots; loss={logs["loss"]:.6g}; val_loss={logs["val_loss"]:.6g}', flush=True)
        if model.stop_training:
            break
    callback_list.on_train_end(logs)
    used_ids = np.flatnonzero(seen)
    np.savez_compressed(run/'training_used.npz', shot_indices=used_ids)
    write_json(run/'training_exposure.json', {
        'mode': mode, 'requested_epochs': epochs, 'available_epochs': limit,
        'completed_epochs': len(history.epoch), 'fresh_shots_per_epoch': fresh_shots_per_epoch if mode == 'fresh' else None,
        'unique_training_shots_seen': len(used_ids), 'reserved_training_shots': len(indices['train']),
        'fixed_training_shots': fixed_training_shots if mode == 'fixed' else None,
        'scheduled_unique_training_shots': len(order), 'scheduled_shots_per_sample': sample_counts,
        'unused_training_shots': len(indices['train'])-len(used_ids), 'epochs': epoch_records,
        'stop_reason': 'early_stopping_or_invalid_loss' if model.stop_training else
                       ('fresh_pool_exhausted' if mode == 'fresh' and boundaries[-1] == len(indices['train']) else 'epoch_limit')})
    return history


def train_collection(args):
    from core.GCN.compression import export_weights
    tf.keras.utils.set_random_seed(args.seed)
    tf.keras.mixed_precision.set_global_policy('float32')
    device = select_device(args.device)
    root = args.data_root.expanduser().resolve()
    manifest = inspect_collection(root, args.configuration, args.expected_samples)
    run = args.run_root.resolve()/datetime.now().strftime('collection_gcn_%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True, exist_ok=False)
    save_graphs(root, manifest, run)
    first = manifest['samples'][0]
    config = {'data_path': str(root), 'data_root': str(root), 'collection': manifest,
              'dataset_sha256': {s['name']: {k: v for k, v in s['sha256'].items() if k != 'dems/correlations.dem'} for s in manifest['samples']},
              'dem_sha256': {s['name']: s['sha256']['dems/correlations.dem'] for s in manifest['samples']},
              'training_mode': args.training_mode+'_dataset', 'split_strategy': 'within_each_sample',
              'graph_type': 'per_sample_detector_error_factor_graph', 'sample_count': manifest['sample_count'],
              'random_seed': args.seed, 'split_seed': args.split_seed, 'batch_size': args.batch_size,
              'maximum_epochs': args.epochs, 'fresh_shots_per_epoch': args.fresh_shots_per_epoch,
              'fixed_training_shots': getattr(args, 'fixed_training_shots', None) if args.training_mode == 'fixed' else None,
              'calibration_fraction': args.calibration_fraction, 'weights_file': 'restored.weights.h5',
              'precision': 'float32', 'hidden_units': args.hidden_units, 'dense_units': args.dense_units,
              'dense2_units': args.dense_units//2, 'number_of_gcn_layers': args.gcn_layers,
              'number_of_features': len(FEATURE_NAMES), 'feature_names': FEATURE_NAMES,
              'dropout_rate': args.dropout, 'learning_rate': args.learning_rate,
              'requested_device': device, 'tensorflow_version': tf.__version__, 'numpy_version': np.__version__,
              'early_stopping': {'patience': getattr(args, 'early_stopping_patience', 15),
                                 'min_delta': getattr(args, 'early_stopping_min_delta', 1e-5)},
              'stim_version': stim.__version__, 'python_version': platform.python_version(), 'platform': platform.platform(),
              'test_evaluated': bool(args.evaluate_test), 'representative_graph': first['name'],
              **{key: first[key] for key in ('number_of_detectors', 'number_of_errors', 'number_of_nodes', 'number_of_edges', 'number_of_incidences')}}
    if args.split_from:
        source = read_configuration(args.split_from)
        if source.get('collection') != manifest:
            raise ValueError('Split source differs from this collection or graph set')
        indices = read_indices(args.split_from, manifest['total_shots'])
        config['split_seed'] = source['split_seed']
    else:
        indices = split_collection(manifest, args.split_seed, args.calibration_fraction)
    config.update({f'num_{key}': len(values) for key, values in indices.items()})
    config['sample_partitions'] = {s['name']: {name: int(np.sum((ids >= s['offset']) & (ids < s['offset']+s['shots'])))
                                             for name, ids in indices.items()} for s in manifest['samples']}
    np.savez_compressed(run/'shot_indices.npz', **indices)
    write_json(run/'configuration.json', config)
    events, labels = read_collection(root, config)
    print(f'Collection: {manifest["sample_count"]} samples, {manifest["total_shots"]} shots; mode={args.training_mode}', flush=True)
    with tf.device(device):
        bank = GraphBank(run, manifest)
        graph = bank.graphs[0]
        model = GCNPredecoder(graph['edge_index'], graph['edge_weights'], args.hidden_units, args.gcn_layers,
                              args.dense_units, args.dropout)
        model(bank.inputs(0, events.read(0, [0])))
        model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate))
        history = fit_collection(model, bank, events, labels, indices, run, args.epochs, args.batch_size,
                                 args.seed, args.training_mode, args.fresh_shots_per_epoch, config['split_seed'],
                                 config['early_stopping']['patience'], config['early_stopping']['min_delta'],
                                 fixed_training_shots=config['fixed_training_shots'])
        model.save_weights(str(run/'restored.weights.h5'))
        config.update(export_weights(model, run/'inference.weights.npz'))
        config['completed_epochs'] = len(history.epoch)
        for name in ['validation', 'calibration'] + (['test'] if args.evaluate_test else []):
            ids = indices[name]
            if len(ids):
                p = bank.predict(model, events[ids], args.batch_size)
                save_predictions(run/f'{name}_predictions.npz', ids, labels[ids], p, manifest)
    write_json(run/'configuration.json', config)
    write_json(run/'history.json', {'epochs': [int(e)+1 for e in history.epoch],
                                    'history': {k: list(map(float, v)) for k, v in history.history.items()},
                                    'test_results': None, 'configuration': config})
    if args.evaluate_test:
        from core.GCN.evaluate import evaluate
        evaluate(run, root, batch_size=args.batch_size, device=args.device)
    print(f'Collection run saved to: {run}')
    return run

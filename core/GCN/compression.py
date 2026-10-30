import argparse
from pathlib import Path
import shutil

import numpy as np
import tensorflow as tf

from core.GCN.GCN import GCNPredecoder
from core.GCN.data_loader import prepare_dataset
from core.GCN.metrics import probability_metrics
from core.GCN.run_io import (load_run, read_records, read_indices, predict_records,
                             save_predictions, write_json, run_identity, select_device)


def retained_indices(scores, count):
    if not 1 <= count <= len(scores):
        raise ValueError('Pruned dimensions must be positive and no larger than the source')
    return np.sort(np.argsort(-np.asarray(scores), kind='stable')[:count])


def prune_model(source, hidden_units, dense_units, dense2_units):
    if source.precision != 'float32':
        raise ValueError('Prune and fine-tune a float32 source before reduced-precision export')
    ip, ip_bias = source.input_projection.get_weights()
    h = ip.shape[1]
    attention, attention_bias = source.attention_score.get_weights()
    head1, head1_bias = source.dense1.get_weights()
    head2, head2_bias = source.dense2.get_weights()
    out, out_bias = source.output_layer.get_weights()
    score = np.abs(ip).sum(axis=0) + np.abs(ip_bias) + np.abs(attention[:, 0])
    score += np.abs(head1).reshape(3, h, -1).sum(axis=(0, 2))
    for conv, norm in zip(source.gcn_layers, source.normalization_layers):
        kernel, bias = conv.get_weights()
        score += np.abs(kernel).sum(axis=0) + np.abs(kernel).sum(axis=1) + np.abs(bias)
        gamma, beta = norm.get_weights()
        score += np.abs(gamma) + np.abs(beta)
    keep = retained_indices(score, hidden_units)
    keep1 = retained_indices(np.abs(head1).sum(axis=0) + np.abs(head1_bias) + np.abs(head2).sum(axis=1), dense_units)
    keep2 = retained_indices(np.abs(head2).sum(axis=0) + np.abs(head2_bias) + np.abs(out).sum(axis=1), dense2_units)
    model = GCNPredecoder(source.edge_index, source.edge_weights, hidden_units=hidden_units,
                         number_of_gcn_layers=len(source.gcn_layers), dense_units=dense_units,
                         dense2_units=dense2_units, dropout_rate=source.architecture['dropout_rate'])
    model(tf.zeros((1, int(tf.reduce_max(source.edge_index)) + 1, ip.shape[0])))
    model.input_projection.set_weights([ip[:, keep], ip_bias[keep]])
    for old, new, old_norm, new_norm in zip(source.gcn_layers, model.gcn_layers,
                                            source.normalization_layers, model.normalization_layers):
        kernel, bias = old.get_weights()
        new.set_weights([kernel[np.ix_(keep, keep)], bias[keep]])
        new_norm.set_weights([value[keep] for value in old_norm.get_weights()])
    model.attention_score.set_weights([attention[keep], attention_bias])
    pooled = np.concatenate([keep + offset * h for offset in range(3)])
    model.dense1.set_weights([head1[np.ix_(pooled, keep1)], head1_bias[keep1]])
    model.dense2.set_weights([head2[np.ix_(keep1, keep2)], head2_bias[keep2]])
    model.output_layer.set_weights([out[keep2], out_bias])
    return model, {'hidden': keep.tolist(), 'dense1': keep1.tolist(), 'dense2': keep2.tolist()}


def export_weights(model, path, storage='float32'):
    half = set()
    if storage == 'mixed_float16':
        for layer in (model.input_projection, model.dense1, model.dense2):
            half.update(id(v) for v in layer.weights)
        half.update(id(layer.kernel) for layer in model.gcn_layers)
    elif storage != 'float32':
        raise ValueError('Unsupported parameter export format')
    arrays = {f'weight_{i}': v.numpy().astype(np.float16 if id(v) in half else np.float32)
              for i, v in enumerate(model.weights)}
    np.savez_compressed(path, **arrays)
    return {'parameter_count': model.count_params(),
            'parameter_bytes': sum(array.nbytes for array in arrays.values()),
            'resident_parameter_bytes': sum(int(np.prod(v.shape)) * 4 for v in model.weights),
            'export_file_bytes': Path(path).stat().st_size, 'parameter_storage': storage}


def compress(run_directory, data_path, output, mode='prune', hidden_units=64, dense_units=64,
             dense2_units=32, epochs=30, batch_size=32, learning_rate=1e-4,
             max_val_bce_increase=.01, seed=73, device='cpu',
             early_stopping_patience=15, early_stopping_min_delta=1e-5,
             fixed_training_shots=None):
    if mode not in ('prune', 'mixed_float16', 'export'):
        raise ValueError('Unknown compression mode')
    if epochs < 1 or batch_size < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Invalid fine-tuning settings')
    if not np.isfinite(max_val_bce_increase) or max_val_bce_increase < 0:
        raise ValueError('Validation tolerance must be finite and nonnegative')
    if early_stopping_patience < 0 or not np.isfinite(early_stopping_min_delta) or early_stopping_min_delta < 0:
        raise ValueError('Invalid early-stopping settings')
    output = Path(output)
    if output.exists():
        raise ValueError('Output run already exists; choose a new directory')
    tf.keras.utils.set_random_seed(seed)
    with tf.device(select_device(device)):
        source, mapper, config = load_run(run_directory)
        if config.get('precision', 'float32') != 'float32':
            raise ValueError('Compression requires a float32 source')
        events, labels = read_records(data_path, config)
        indices = read_indices(run_directory, len(labels))
        validation = indices['validation']
        parent_p = predict_records(source, mapper, events[validation], batch_size)
        source_metrics = probability_metrics(labels[validation], parent_p)
        output.mkdir(parents=True, exist_ok=False)
        for name in ('graph_structure.npz', 'shot_indices.npz'):
            shutil.copy2(Path(run_directory) / name, output / name)
        if config.get('collection'):
            shutil.copytree(Path(run_directory)/'graphs', output/'graphs')
            for name in ('collection.json', 'training_schedule.npz', 'training_used.npz', 'training_exposure.json'):
                shutil.copy2(Path(run_directory)/name, output/name)
        history = None
        selection = None
        if mode == 'prune':
            model, selection = prune_model(source, hidden_units, dense_units, dense2_units)
            if config.get('collection'):
                from core.GCN.collection_training import fit_collection
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))
                history = fit_collection(model, mapper, events, labels, indices, output,
                                         epochs, batch_size, seed, mode='fixed',
                                         schedule_seed=config.get('split_seed', 73),
                                         fixed_training_shots=fixed_training_shots,
                                         early_stopping_patience=early_stopping_patience,
                                         early_stopping_min_delta=early_stopping_min_delta)
            else:
                datasets = {}
                for name in ('train', 'validation'):
                    ids = indices[name]
                    compact = tf.data.Dataset.from_tensor_slices((events[ids], labels[ids, None]))
                    datasets[name] = prepare_dataset(compact, mapper, batch_size, name == 'train', seed)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))
                history = model.fit(datasets['train'], validation_data=datasets['validation'], epochs=epochs,
                                    shuffle=False, verbose=1, callbacks=[
                                        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_patience,
                                                                         min_delta=early_stopping_min_delta, restore_best_weights=True),
                                        tf.keras.callbacks.ModelCheckpoint(str(output / 'best.weights.h5'),
                                                                           monitor='val_loss', save_best_only=True,
                                                                           save_weights_only=True),
                                        tf.keras.callbacks.TerminateOnNaN()])
            model.save_weights(str(output / 'restored.weights.h5'))
        else:
            model = source
        storage = 'mixed_float16' if mode == 'mixed_float16' else 'float32'
        stats = export_weights(model, output / 'inference.weights.npz', storage)
        updated = {**config, **model.architecture, **stats, 'kind': mode,
                   'weights_file': 'inference.weights.npz', 'precision': storage,
                   'parent_run': str(Path(run_directory).resolve()), 'parent_identity': run_identity(run_directory),
                   'retained_indices': selection, 'random_seed': seed,
                   'batch_size': batch_size, 'maximum_epochs': epochs if mode == 'prune' else 0,
                   'learning_rate': learning_rate if mode == 'prune' else None,
                   'completed_epochs': len(history.epoch) if history else 0,
                   'test_evaluated': False}
        if mode == 'prune':
            updated['early_stopping'] = {'patience': early_stopping_patience, 'min_delta': early_stopping_min_delta}
        if config.get('collection') and mode == 'prune':
            updated['fine_tuning_mode'] = 'fixed_dataset'
            updated['fixed_training_shots'] = fixed_training_shots
        updated.pop('experiment_label', None)
        write_json(output / 'configuration.json', updated)
        reloaded, mapper, _ = load_run(output)
        val_p = predict_records(reloaded, mapper, events[validation], batch_size)
        metrics = probability_metrics(labels[validation], val_p)
        updated['validation_selection'] = {'source_bce': source_metrics['bce'], 'candidate_bce': metrics['bce'],
                                           'max_bce_increase': max_val_bce_increase,
                                           'eligible': metrics['bce'] <= source_metrics['bce'] + max_val_bce_increase}
        updated['validation_probability_max_abs_change'] = float(np.max(np.abs(parent_p - val_p)))
        write_json(output / 'configuration.json', updated)
        for name in ('validation', 'calibration'):
            ids = indices[name]
            p = val_p if name == 'validation' else predict_records(reloaded, mapper, events[ids], batch_size)
            if len(ids):
                save_predictions(output / f'{name}_predictions.npz', ids, labels[ids], p, config.get('collection'))
        write_json(output / 'history.json', {
            'epochs': [int(e) + 1 for e in history.epoch] if history else [],
            'history': {name: list(map(float, values)) for name, values in history.history.items()} if history else {},
            'configuration': updated, 'test_results': None, 'validation_results': metrics})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['prune', 'mixed_float16', 'export'], default='prune')
    parser.add_argument('--hidden-units', type=int, default=64)
    parser.add_argument('--dense-units', type=int, default=64)
    parser.add_argument('--dense2-units', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--max-val-bce-increase', type=float, default=.01)
    parser.add_argument('--seed', type=int, default=73)
    parser.add_argument('--device', choices=['cpu', 'gpu', 'auto'], default='cpu')
    args = parser.parse_args()
    result = compress(args.run_dir, args.data_path, args.output, args.mode, args.hidden_units,
                      args.dense_units, args.dense2_units, args.epochs, args.batch_size,
                      args.learning_rate, args.max_val_bce_increase, args.seed, args.device)
    print(f'Compression run saved to: {result}')


if __name__ == '__main__':
    main()

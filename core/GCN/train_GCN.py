import argparse
from datetime import datetime
import json
from pathlib import Path
import platform

import numpy as np
import stim
import tensorflow as tf
from tensorflow.keras import callbacks, optimizers

from core.data_loader import load_experimental_shots
from core.GCN.data_loader import prepare_dataset, split_dataset, datasets_from_indices
from core.GCN.graph_class import Graph
from core.GCN.graph_data import FEATURE_NAMES, make_gcn_structure, make_node_feature_mapper
from core.GCN.GCN import GCNPredecoder
from core.GCN.run_io import (dataset_fingerprints, file_sha256, select_device,
                             read_configuration, verify_dataset, read_indices, write_json)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', type=Path, default=PROJECT_ROOT / (
        'data/sycamore_surface_code_d3_d5/sample_00/d5_at_q6_5/X/r25'))
    parser.add_argument('--dem-path', type=Path, help='Defaults to DATA_PATH/dems/correlations.dem')
    parser.add_argument('--data-root', type=Path, help='Collection root containing sample_00, sample_01, ...')
    parser.add_argument('--configuration', default='d5_at_q6_5/X/r25')
    parser.add_argument('--expected-samples', type=int)
    parser.add_argument('--training-mode', choices=['fixed', 'fresh'], default='fixed')
    parser.add_argument('--fresh-shots-per-epoch', type=int, default=60000)
    parser.add_argument('--run-root', type=Path, default=Path(__file__).parent / 'runs')
    parser.add_argument('--device', choices=['auto', 'cpu', 'gpu'], default='auto')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=73, help='Initialization and training seed')
    parser.add_argument('--split-seed', type=int, default=73)
    parser.add_argument('--split-from', type=Path, help="Reuse a run's exact saved partitions")
    parser.add_argument('--evaluate-test', action='store_true',
                        help='Opt in to test evaluation; leave off during development')
    parser.add_argument('--hidden-units', type=int, default=96)
    parser.add_argument('--gcn-layers', type=int, default=3)
    parser.add_argument('--dense-units', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--calibration-fraction', type=float, default=0.0,
                        help='Fraction of all shots reserved from the 10%% development pool')
    return parser


def train(args):
    if min(args.batch_size, args.epochs, args.hidden_units, args.gcn_layers) <= 0:
        raise ValueError('Batch size, epochs, hidden units and GCN layers must be positive')
    if args.dense_units < 2 or not 0 <= args.dropout < 1 or args.learning_rate <= 0:
        raise ValueError('Invalid dense width, dropout or learning rate')
    if args.data_root:
        if args.dem_path:
            raise ValueError('A collection uses each sample\'s own dems/correlations.dem')
        from core.GCN.collection_training import train_collection
        return train_collection(args)
    if args.training_mode != 'fixed':
        raise ValueError('Fresh mode requires --data-root pointing at the sample collection')
    tf.keras.utils.set_random_seed(args.seed)
    tf.keras.mixed_precision.set_global_policy('float32')
    device = select_device(args.device)
    data_path = args.data_path.resolve()
    circuit_path = data_path / 'circuit_ideal.stim'
    dem_path = (args.dem_path or data_path / 'dems/correlations.dem').resolve()
    graph = Graph(circuit_path=circuit_path, dem_path=dem_path)
    edge_index, edge_weights = make_gcn_structure(graph)
    mapper, number_of_features = make_node_feature_mapper(graph)
    number_of_nodes = len(graph.graph_nodes)
    number_of_detectors = len(graph.detector_nodes)
    flat_dataset = load_experimental_shots(data_path)
    if flat_dataset.element_spec[0].shape != tf.TensorShape([number_of_detectors]):
        raise ValueError('Expected one flat detector vector per experimental shot')
    if flat_dataset.element_spec[1].shape != tf.TensorShape([1]):
        raise ValueError('Expected one logical-observable label per shot')
    if args.split_from:
        source_config = read_configuration(args.split_from)
        verify_dataset(data_path, source_config)
        indices = read_indices(args.split_from, int(flat_dataset.cardinality()))
        partitions = datasets_from_indices(flat_dataset, indices)
    else:
        partitions, indices = split_dataset(flat_dataset, args.split_seed, args.calibration_fraction)
    datasets = {
        name: prepare_dataset(partitions[name], mapper, args.batch_size,
                              training=name == 'train', random_seed=args.seed)
        for name in ('train', 'validation', 'test')
    }
    print(f'Detectors: {number_of_detectors}; errors: {len(graph.error_nodes)}; '
          f'nodes: {number_of_nodes}; directed edges: {edge_index.shape[1]}')
    print('Shot counts:', {name: len(values) for name, values in indices.items()})
    print(f'Requested execution device: {device}')

    run_name = datetime.now().strftime('factor_gcn_%Y%m%d_%H%M%S_%f')
    run_directory = args.run_root.resolve() / run_name
    run_directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(run_directory / 'shot_indices.npz', **indices)
    example_features, _ = mapper(tf.zeros(number_of_detectors), tf.zeros(1))
    np.savez_compressed(
        run_directory / 'graph_structure.npz', edge_index=edge_index,
        edge_weights=edge_weights, static_node_features=example_features.numpy()[:, 1:],
        feature_names=np.asarray(FEATURE_NAMES), number_of_detectors=number_of_detectors,
    )
    configuration = {
        'data_path': str(data_path), 'circuit_path': str(circuit_path), 'dem_path': str(dem_path),
        'dataset_sha256': dataset_fingerprints(data_path), 'dem_sha256': file_sha256(dem_path),
        'graph_type': 'detector_error_factor_graph', 'training_mode': 'fixed_dataset',
        'random_seed': args.seed, 'batch_size': args.batch_size, 'maximum_epochs': args.epochs,
        **{f'num_{name}': len(values) for name, values in indices.items()},
        'calibration_fraction': len(indices['calibration']) / int(flat_dataset.cardinality()),
        'split_seed': args.split_seed if not args.split_from else source_config.get('split_seed'),
        'split_from': str(args.split_from.resolve()) if args.split_from else None,
        'weights_file': 'restored.weights.h5', 'test_evaluated': bool(args.evaluate_test),
        'number_of_nodes': number_of_nodes, 'number_of_detectors': number_of_detectors,
        'number_of_errors': len(graph.error_nodes), 'number_of_incidences': len(graph.edges),
        'number_of_edges': int(edge_index.shape[1]), 'number_of_features': number_of_features,
        'feature_names': FEATURE_NAMES, 'hidden_units': args.hidden_units,
        'number_of_gcn_layers': args.gcn_layers, 'dense_units': args.dense_units,
        'dense2_units': args.dense_units // 2,
        'dropout_rate': args.dropout, 'learning_rate': args.learning_rate,
        'precision': 'float32', 'requested_device': device,
        'tensorflow_version': tf.__version__, 'stim_version': stim.__version__,
        'numpy_version': np.__version__, 'python_version': platform.python_version(),
        'platform': platform.platform(),
    }
    (run_directory / 'configuration.json').write_text(json.dumps(configuration, indent=2))
    with tf.device(device):
        model = GCNPredecoder(
            edge_index, edge_weights, hidden_units=args.hidden_units,
            number_of_gcn_layers=args.gcn_layers, dense_units=args.dense_units,
            dropout_rate=args.dropout,
        )
        model(tf.zeros((1, number_of_nodes, number_of_features), dtype=tf.float32))
        model.compile(optimizer=optimizers.Adam(learning_rate=args.learning_rate))
        model.summary()
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss', mode='min', patience=15, min_delta=1e-5,
            restore_best_weights=True, verbose=1,
        )
        checkpoint = callbacks.ModelCheckpoint(
            filepath=str(run_directory / 'best.weights.h5'), monitor='val_loss', mode='min',
            save_best_only=True, save_weights_only=True, verbose=1,
        )
        history = model.fit(
            datasets['train'], validation_data=datasets['validation'], epochs=args.epochs,
            callbacks=[early_stopping, checkpoint, callbacks.TerminateOnNaN()],
            shuffle=False, verbose=1,  # Training order is controlled by the dataset.
        )
        model.save_weights(str(run_directory / 'restored.weights.h5'))
        test_results = (model.evaluate(datasets['test'], return_dict=True, verbose=1)
                        if args.evaluate_test else None)
        export_subsets = ['validation', 'calibration'] + (['test'] if args.evaluate_test else [])
        for name in export_subsets:
            if not len(indices[name]):
                continue
            dataset = prepare_dataset(partitions[name], mapper, args.batch_size)
            probabilities = model.predict(dataset, verbose=0).reshape(-1)
            labels = np.fromiter((label[0] for _, label in partitions[name].as_numpy_iterator()),
                                 dtype=np.uint8, count=len(indices[name]))
            np.savez_compressed(run_directory / f'{name}_predictions.npz',
                                shot_indices=indices[name], labels=labels, probabilities=probabilities)

    from core.GCN.compression import export_weights
    configuration.update(export_weights(model, run_directory / 'inference.weights.npz'))
    configuration['inference_export_file'] = 'inference.weights.npz'
    configuration['parameter_count'] = model.count_params()
    configuration['parameter_bytes'] = sum(int(np.prod(v.shape)) * 4 for v in model.weights)
    configuration['weights_file_bytes'] = (run_directory / 'restored.weights.h5').stat().st_size
    configuration['completed_epochs'] = len(history.epoch)
    history_data = {
        'epochs': [int(epoch) + 1 for epoch in history.epoch],
        'history': {name: [float(v) for v in values] for name, values in history.history.items()},
        'test_results': ({name: float(value) for name, value in test_results.items()}
                         if test_results is not None else None),
        'configuration': configuration,
    }
    write_json(run_directory / 'configuration.json', configuration)
    write_json(run_directory / 'history.json', history_data)
    print(f'Weights, graph, shot indices, predictions and history saved to: {run_directory}')
    return run_directory


if __name__ == '__main__':
    train(make_parser().parse_args())

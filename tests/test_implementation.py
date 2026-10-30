import json

import numpy as np
import pytest
import stim
import tensorflow as tf

from core.data_loader import load_experimental_shots, make_dataset
from core.GCN.benchmark import benchmark
from core.GCN.compare_decoders import compare, load_neural_predictions
from core.GCN.data_loader import prepare_dataset, split_dataset, validate_shot_indices
from core.GCN.GCN import GraphConvolution
from core.GCN.graph_class import Graph
from core.GCN.graph_data import make_gcn_structure, make_node_feature_mapper
from core.GCN.run_io import load_run
from core.GCN.train_GCN import make_parser, train


@pytest.fixture
def experimental_data(tmp_path):
    data = tmp_path / 'data'
    (data / 'dems').mkdir(parents=True)
    (data / 'circuit_ideal.stim').write_text('''
R 0
M 0
DETECTOR(0, 0, 0) rec[-1]
DETECTOR(0, 0, 1, 2, 0, 1, 0, 0, 0) rec[-1]
DETECTOR(1, 0, 2, 1, 0, 1) rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
''')
    (data / 'dems/correlations.dem').write_text('''
error(0.1) D0 D1 L0 ^ D1 D2 L0
error(0.2) D1
error(0.05) L0
error(0.03) D2 D2
''')
    events = np.random.default_rng(73).integers(0, 2, (40, 3)).astype(bool)
    labels = np.logical_xor(events[:, :1], events[:, 1:2])
    stim.write_shot_data_file(data=events, path=data / 'detection_events.b8',
                              format='b8', num_detectors=3)
    stim.write_shot_data_file(data=labels, path=data / 'obs_flips_actual.b8',
                              format='b8', num_observables=1)
    return data, events, labels


def make_graph(data):
    return Graph(data / 'circuit_ideal.stim', data / 'dems/correlations.dem')


def test_graph_coordinates_xor_and_isolated_mechanisms(experimental_data):
    graph = make_graph(experimental_data[0])
    assert graph.node_weights == {0: 1, 1: 2, 2: 1}
    assert graph.detector_nodes[0].previous_measurement_coordinates is None
    assert graph.detector_nodes[1].coordinates == (1, 0, 1)
    assert graph.detector_nodes[1].parity_measurements == [(0, 0, 1), (2, 0, 1)]
    assert graph.detector_nodes[2].previous_measurement_coordinates == (1, 0, 1)
    assert graph.detector_nodes[0].coordinate_record == (0, 0, 0)
    assert graph.detector_nodes[0].init_boundary
    assert graph.detector_nodes[1].bulk
    assert graph.detector_nodes[2].final_boundary
    assert graph.error_nodes[3].detector_ids == (0, 2)
    assert graph.error_nodes[3].observable_ids == ()
    assert len(graph.error_nodes[3].components) == 2
    assert graph.error_nodes[4].is_boundary
    assert graph.error_nodes[5].is_undetectable
    assert graph.error_nodes[5].observable_ids == (0,)
    assert graph.error_nodes[6].is_undetectable
    assert graph.error_nodes[6].observable_ids == ()
    edges, weights = make_gcn_structure(graph)
    assert edges.shape == (2, 2 * len(graph.edges) + len(graph.graph_nodes))
    pairs = list(zip(*edges))
    assert len(pairs) == len(set(pairs))
    assert weights[pairs.index((5, 5))] == 1
    assert weights[pairs.index((6, 6))] == 1


def test_loader_and_mapper_preserve_labels_and_batch_shapes(experimental_data):
    data, events, labels = experimental_data
    dataset = load_experimental_shots(str(data))
    assert int(make_dataset(data).cardinality()) == 40
    loaded = list(dataset.as_numpy_iterator())
    np.testing.assert_array_equal(np.stack([x for x, _ in loaded]), events)
    np.testing.assert_array_equal(np.stack([y for _, y in loaded]), labels)
    graph = make_graph(data)
    mapper, width = make_node_feature_mapper(graph)
    features, targets = mapper(events[:4], labels[:4])
    assert features.shape == (4, 7, width) == (4, 7, 13)
    np.testing.assert_array_equal(features[:, :3, 0], events[:4])
    np.testing.assert_array_equal(features[:, 3:, 0], 0)
    np.testing.assert_array_equal(targets, labels[:4])
    assert features.dtype == targets.dtype == tf.float32
    np.testing.assert_array_equal(features[0, 5, 10:13], [0, 1, 1])
    for i in range(4):
        single, _ = mapper(events[i], labels[i])
        np.testing.assert_array_equal(features[i], single)
    traced = tf.function(mapper, input_signature=[
        tf.TensorSpec((None, None), tf.float32), tf.TensorSpec((None, 1), tf.float32)])
    assert traced(tf.cast(events[:4], tf.float32), tf.cast(labels[:4], tf.float32))[0].shape == (4, 7, 13)
    with pytest.raises(tf.errors.InvalidArgumentError, match='width'):
        traced(tf.zeros((1, 2)), tf.zeros((1, 1)))


def test_loader_rejects_misaligned_records(experimental_data):
    data, _, labels = experimental_data
    stim.write_shot_data_file(data=labels[:-1], path=data / 'obs_flips_actual.b8',
                              format='b8', num_observables=1)
    with pytest.raises(ValueError, match='shot counts'):
        load_experimental_shots(data)


def test_fixed_partitions_and_calibration_are_disjoint():
    tf.keras.utils.set_random_seed(73)
    dataset = tf.data.Dataset.from_tensor_slices((np.arange(100), np.arange(100)))
    partitions, indices = split_dataset(dataset, 73, calibration_fraction=0.05)
    assert {k: len(v) for k, v in indices.items()} == {
        'train': 80, 'validation': 5, 'calibration': 5, 'test': 10}
    for name, subset in partitions.items():
        for _ in range(2):
            values = list(subset.as_numpy_iterator())
            np.testing.assert_array_equal([x for x, y in values], indices[name])
            assert all(x == y for x, y in values)
    shuffled = prepare_dataset(partitions['train'], lambda x, y: (x, y), 8, training=True)
    first = np.concatenate([x for x, _ in shuffled.as_numpy_iterator()])
    second = np.concatenate([x for x, _ in shuffled.as_numpy_iterator()])
    assert not np.array_equal(first, second)
    np.testing.assert_array_equal(np.sort(first), np.sort(indices['train']))
    overlapping = {k: v.copy() for k, v in indices.items()}
    overlapping['test'][0] = indices['train'][0]
    with pytest.raises(ValueError, match='overlaps'):
        validate_shot_indices(overlapping, 100)


def test_convolution_matches_normalized_dense_aggregation(experimental_data):
    edges, weights = make_gcn_structure(make_graph(experimental_data[0]))
    x = np.arange(2 * 7 * 2, dtype=np.float32).reshape(2, 7, 2) / 10
    layer = GraphConvolution(2)
    layer(tf.constant(x), tf.constant(edges), tf.constant(weights))
    kernel = np.asarray([[1, 2], [3, 4]], np.float32)
    layer.set_weights([kernel, np.array([0.5, -0.5], np.float32)])
    adjacency = np.zeros((7, 7), np.float32)
    for (source, target), weight in zip(edges.T, weights):
        adjacency[target, source] = weight
    expected = adjacency @ (x @ kernel) + [0.5, -0.5]
    actual = layer(tf.constant(x), tf.constant(edges), tf.constant(weights))
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_run_round_trip_comparison_and_cpu_benchmark(experimental_data, tmp_path):
    data, events, labels = experimental_data
    args = make_parser().parse_args([
        '--data-path', str(data), '--run-root', str(tmp_path / 'runs'),
        '--epochs', '1', '--batch-size', '8', '--device', 'cpu', '--evaluate-test',
    ])
    run = train(args)
    expected = {'configuration.json', 'history.json', 'shot_indices.npz', 'graph_structure.npz',
                'test_predictions.npz', 'best.weights.h5', 'restored.weights.h5'}
    assert expected.issubset({path.name for path in run.iterdir()})
    saved = json.loads((run / 'history.json').read_text())
    assert saved['configuration']['num_train'] == 32
    assert saved['configuration']['num_validation'] == 4
    assert saved['configuration']['num_test'] == 4
    assert set(saved['test_results']) == {'loss', 'accuracy', 'roc_auc'}
    assert all(np.isfinite(list(saved['test_results'].values())))
    model, mapper, _ = load_run(run)
    with np.load(run / 'test_predictions.npz') as predictions:
        ids = predictions['shot_indices']
        x, _ = mapper(events[ids], labels[ids])
        np.testing.assert_allclose(model(x, training=False).numpy().ravel(),
                                   predictions['probabilities'], rtol=1e-5, atol=1e-6)
    results = compare(data, run, {'perfect_reference': data / 'obs_flips_actual.b8'})
    assert results['results']['perfect_reference']['incorrect'] == 0
    assert results['results']['neural']['accuracy'] == pytest.approx(saved['test_results']['accuracy'])
    with np.load(run / 'shot_indices.npz') as split:
        constant = int(labels[split['train']].mean() > 0.5)
    assert results['results']['constant']['class_selected_on_training'] == constant
    report = benchmark(run, data, warmup=1, repetitions=2)
    assert len(report['measurements']) == 2
    assert {v['scope'] for v in report['measurements']} == {'model_only', 'detector_to_prediction'}
    assert all(v['latency']['median'] > 0 for v in report['measurements'])
    bad = tmp_path / 'bad_predictions.npz'
    np.savez(bad, shot_indices=ids, labels=1 - labels[ids, 0], probabilities=np.zeros(len(ids)))
    with pytest.raises(ValueError, match='labels'):
        load_neural_predictions(bad, ids, labels[ids, 0])

import json
import numpy as np
import pytest
import stim
import tensorflow as tf
from test_implementation import experimental_data
from core.GCN.GCN import GCNPredecoder, GraphConvolution
from core.GCN.benchmark import benchmark
from core.GCN.calibration import (fit_temperature, calibrate_probabilities, select_threshold,
                                  gate_metrics, fit_gate, verify_gate, calibrate_tensor)
from core.GCN.compression import prune_model, compress
from core.GCN.data_loader import split_dataset
from core.GCN.evaluate import evaluate
from core.GCN.metrics import probability_metrics, classification_metrics, paired_difference, upper_error_bound
from core.GCN.report import export_reports, seed_group, tex_escape
from core.GCN.run_io import load_run, read_indices, write_json
from core.GCN.train_GCN import train, make_parser


def test_metrics_ties_bounds_and_pairing():
    metrics = probability_metrics([0, 1], [.5, .5])
    assert metrics['roc_auc'] == .5
    assert metrics['bce'] == pytest.approx(np.log(2))
    assert metrics['brier'] == .25 and metrics['ece'] == 0
    assert probability_metrics([0, 0], [.1, .2])['roc_auc'] is None
    assert upper_error_bound(0, 7500) == pytest.approx(1 - .05 ** (1 / 7500))
    assert classification_metrics([], [])['error_rate'] is None
    paired = paired_difference([0, 1], [0, 1], [1, 0], repetitions=50)
    assert paired['error_rate_difference'] == -1 and paired['ci'] == [-1, -1]
    with pytest.raises(ValueError):
        probability_metrics([0], [np.nan])


def test_split_seed_independent_of_initialization():
    dataset = tf.data.Dataset.from_tensor_slices((np.arange(100), np.arange(100)))
    tf.keras.utils.set_random_seed(1)
    _, first = split_dataset(dataset, 73, .05)
    tf.keras.utils.set_random_seed(222)
    _, second = split_dataset(dataset, 73, .05)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


def test_pruning_full_copy_and_smaller_network():
    tf.keras.utils.set_random_seed(73)
    model = GCNPredecoder([[0, 1], [0, 1]], [1., 1.], hidden_units=8, dense_units=8, dense2_units=4)
    x = tf.random.uniform((2, 2, 13))
    expected = model(x, training=False).numpy()
    copied, _ = prune_model(model, 8, 8, 4)
    np.testing.assert_array_equal(copied(x, training=False), expected)
    smaller, retained = prune_model(model, 4, 4, 2)
    assert smaller.count_params() < model.count_params()
    assert len(retained['hidden']) == 4 and smaller(x).shape == (2, 1)
    np.testing.assert_array_equal(smaller.edge_index, model.edge_index)
    with pytest.raises(ValueError):
        prune_model(model, 9, 4, 2)


def test_mixed_accumulation_stays_float32():
    edges, weights = tf.zeros((2, 70000), tf.int32), tf.ones(70000, tf.float32)
    layer = GraphConvolution(1, dtype='mixed_float16')
    layer(tf.ones((1, 1, 1)), edges, weights)
    layer.set_weights([np.ones((1, 1)), np.zeros(1)])
    result = layer(tf.ones((1, 1, 1)), edges, weights)
    assert result.dtype == tf.float32 and float(result[0, 0, 0]) == 70000
    assert layer.kernel.dtype == 'float32'


def test_temperature_and_gate_with_unsupported_target():
    labels = np.tile([0, 1, 1, 0], 100)
    p = np.tile([.01, .99, .01, .99], 100)
    temperature = fit_temperature(labels, p)
    assert temperature > 1
    assert probability_metrics(labels, calibrate_probabilities(p, temperature))['bce'] < probability_metrics(labels, p)['bce']
    good_labels = np.tile([0, 1], 5000)
    good_p = np.where(good_labels, .99, .01)
    gate = select_threshold(good_labels, good_p, .01, thresholds=[.5, .9, .99])
    assert gate['enabled'] and gate['threshold'] == .5
    impossible = select_threshold(good_labels, good_p, 1e-8)
    assert not impossible['enabled'] and impossible['threshold'] is None
    impossible['temperature'] = 1
    result = gate_metrics(good_labels, good_p, impossible, fallback=good_labels)
    assert result['coverage'] == 0 and result['accepted']['error_rate'] is None
    assert result['hybrid']['error_rate'] == 0
    boundary = np.array([.01, .005, .015, .1, .99], dtype=np.float32)
    q = calibrate_probabilities(boundary, 1)
    grid = np.array([.9, .985, .99, .995], dtype=np.float32)
    selected = select_threshold(np.zeros(len(q)), q, .99, min_accepted=1, thresholds=grid)
    deployed = calibrate_tensor(boundary, tf.constant(1., tf.float32))
    for row in selected['calibration_curve']:
        count = int(tf.reduce_sum(tf.cast(tf.maximum(deployed, 1-deployed) >= row['threshold'], tf.int32)))
        assert row['accepted'] == count


def test_report_groups_and_latex_escaping():
    config = {'kind': 'prune', 'hidden_units': 64, 'number_of_gcn_layers': 3,
              'dense_units': 64, 'dense2_units': 32, 'dropout_rate': .1}
    assert seed_group(config)[0] != seed_group({**config, 'hidden_units': 32})[0]
    assert seed_group({**config, 'experiment_label': 'pruned64'})[0] == 'pruned64'
    escaped = tex_escape('mixed_64 & 95%')
    assert r'\_\allowbreak ' in escaped and r'\&' in escaped and r'\%' in escaped
    assert all(ord(char) >= 32 for char in escaped)


def test_final_selection_locked_before_test_access(tmp_path, monkeypatch):
    from core.GCN import experiments, run_io
    run = tmp_path/'run'
    run.mkdir()
    write_json(run/'gate.json', {'threshold': .9})
    write_json(tmp_path/'manifest.json', {
        'configuration': {'data_path': 'unused', 'gate': {'target_error': .01}},
        'configuration_sha256': 'config',
        'runs': {'73/baseline': {'path': str(run), 'eligible': True, 'status': 'development_complete'}}})
    write_json(tmp_path/'selection.json', {'runs': ['73/baseline']})
    identity = {'weights': 'original'}
    monkeypatch.setattr(run_io, 'run_identity', lambda _: dict(identity))
    calls = []
    def execute(module, options):
        assert (tmp_path/'final_selection.json').exists()
        calls.append(module)
    monkeypatch.setattr(experiments, 'execute', execute)
    experiments.finalize(tmp_path/'manifest.json', tmp_path/'selection.json')
    assert calls == ['evaluate', 'report']
    write_json(run/'gate.json', {'threshold': .95})
    with pytest.raises(ValueError, match='changed after test access'):
        experiments.finalize(tmp_path/'manifest.json', tmp_path/'selection.json')
    write_json(run/'gate.json', {'threshold': .9})
    identity['weights'] = 'retrained'
    with pytest.raises(ValueError, match='changed after test access'):
        experiments.finalize(tmp_path/'manifest.json', tmp_path/'selection.json')
    assert calls == ['evaluate', 'report']


def test_complete_evaluation_workflow(experimental_data, tmp_path):
    data, events, labels = experimental_data
    events, labels = np.tile(events, (5, 1)), np.tile(labels, (5, 1))
    stim.write_shot_data_file(data=events, path=data/'detection_events.b8', format='b8', num_detectors=3)
    stim.write_shot_data_file(data=labels, path=data/'obs_flips_actual.b8', format='b8', num_observables=1)
    args = make_parser().parse_args(['--data-path', str(data), '--run-root', str(tmp_path/'runs'),
                                    '--epochs', '1', '--batch-size', '16', '--calibration-fraction', '.05'])
    base = train(args)
    assert not (base/'test_predictions.npz').exists()
    assert json.loads((base/'history.json').read_text())['test_results'] is None
    gate = fit_gate(base, data, target_error=.99, method='none', min_accepted=1)
    indices = read_indices(base, len(events))
    assert not set(gate['fit_shot_indices'] + gate['selection_shot_indices']) & set(indices['test'])
    with pytest.raises(ValueError, match='different'):
        verify_gate({**gate, 'run_identity': {}}, base, indices)
    pruned = compress(base, data, tmp_path/'pruned', epochs=1, batch_size=16, max_val_bce_increase=1)
    mixed = compress(pruned, data, tmp_path/'mixed', mode='mixed_float16', batch_size=16, max_val_bce_increase=1)
    pmodel, _, pcfg = load_run(pruned)
    mmodel, _, mcfg = load_run(mixed)
    assert mmodel.precision == 'mixed_float16'
    assert pmodel.count_params() == mmodel.count_params() < 75266
    assert mcfg['parameter_bytes'] < pcfg['parameter_bytes']
    assert pcfg['batch_size'] == 16 and pcfg['maximum_epochs'] == 1
    for run in (pruned, mixed):
        assert not (run/'test_predictions.npz').exists()
    fit_gate(mixed, data, .99, method='none', min_accepted=1)
    live_dem = tmp_path/'live.dem'
    live_dem.write_text('error(0.1) D0 L0\nerror(0.1) D1\nerror(0.1) D2\n')
    report = evaluate(mixed, data, references={'perfect': data/'obs_flips_actual.b8'},
                      gate_path=mixed/'gate.json', live_dem=live_dem, bootstrap=50, batch_size=16)
    assert report['results']['perfect']['error_rate'] == 0 and 'pymatching' in report['hybrids']
    assert report['results']['neural']['shots'] == 20
    with np.load(mixed/'evaluation_test.references.npz') as refs:
        assert refs['reference_names'].tolist() == ['perfect', 'pymatching']
        np.testing.assert_array_equal(refs['shot_indices'], indices['test'])
    timing = benchmark(mixed, data, warmup=1, repetitions=3, num_shots=3,
                       throughput_repetitions=1, throughput_batch_size=2,
                       gate_path=mixed/'gate.json', live_dem=live_dem)
    assert {item['scope'] for item in timing['measurements']} == {
        'model_only', 'detector_to_prediction', 'detector_to_gate', 'live_hybrid', 'live_reference'}
    throughput = timing['measurements'][0]['throughput']
    assert throughput['completed_shots'] == 3 and throughput['actual_batch_sizes'] == [2, 1]
    assert timing['measurements'][0]['memory']['sampled_peak_rss_bytes'] > 0
    write_json(mixed/'benchmark_cpu.json', timing)
    exported = export_reports([mixed/'evaluation_test.json'], [mixed/'benchmark_cpu.json'], tmp_path/'tables')
    assert (exported/'prediction_table.tex').is_file() and (exported/'latency.csv').is_file()
    assert (exported/'diagnostics_1.pdf').is_file()
    assert 'pending' not in (exported/'prediction_table.tex').read_text()
    with pytest.raises(ValueError, match='unique seeds'):
        export_reports([mixed/'evaluation_test.json'] * 2, [], tmp_path/'duplicates')

import json
import shutil
import numpy as np
import pytest
import stim
from test_implementation import experimental_data
from core.GCN.collection import inspect_collection, read_collection, split_collection
from core.GCN.collection_training import training_schedule, fixed_training_subset
from core.GCN.train_GCN import train, make_parser
from core.GCN.run_io import read_indices, load_run, predict_records, training_ids, write_json
from core.GCN.calibration import fit_gate
from core.GCN.compression import compress
from core.GCN.evaluate import evaluate
from core.GCN.benchmark import benchmark
from core.GCN.report import export_reports
from core.GCN.compare_decoders import compare


@pytest.fixture
def acquisition_collection(experimental_data, tmp_path):
    data, events, labels = experimental_data
    root = tmp_path/'collection'
    for index in range(3):
        target = root/f'sample_{index:02d}'/'d5_at_q6_5/X/r25'
        shutil.copytree(data, target)
        stim.write_shot_data_file(data=np.tile(events, (5, 1)), path=target/'detection_events.b8', format='b8', num_detectors=3)
        stim.write_shot_data_file(data=np.tile(labels, (5, 1)), path=target/'obs_flips_actual.b8', format='b8', num_observables=1)
        if index == 2:
            with (target/'dems/correlations.dem').open('a') as out:
                out.write('error(0.13) D0 L0\n')
        (target/'dems/live.dem').write_text('error(0.1) D0 L0\nerror(0.1) D1\nerror(0.1) D2\n')
    return root


def test_collection_discovery_packed_rows_and_isolation(acquisition_collection):
    root = acquisition_collection
    manifest = inspect_collection(root, expected_samples=3)
    assert manifest['total_shots'] == 600
    indices = split_collection(manifest)
    assert {k: len(v) for k,v in indices.items()} == {'train':480, 'validation':30, 'calibration':30, 'test':60}
    events, labels = read_collection(root, {'collection':manifest})
    selected = events[[401, 0, 202, 199]]
    recovered = np.empty((4,3), bool)
    for pos, sample, batch in selected.batches(2):
        recovered[pos] = batch
    assert np.array_equal(np.logical_xor(recovered[:,0],recovered[:,1]), labels[selected.ids])
    with pytest.raises(ValueError, match='expected 53'):
        inspect_collection(root, expected_samples=53)
    target = root/'sample_01/d5_at_q6_5/X/r25/obs_flips_actual.b8'
    target.write_bytes(target.read_bytes()[:-1])
    with pytest.raises(ValueError, match='changed'):
        read_collection(root, {'collection':manifest})
    with pytest.raises(ValueError, match='misaligned'):
        inspect_collection(root)


def test_fresh_schedule_never_repeats_and_keeps_remainder():
    ids = np.arange(11)
    order, bounds, epochs = training_schedule(ids, 'fresh', 200, 4, 73)
    assert epochs == 3 and bounds.tolist() == [0,4,8,11]
    assert sorted(order.tolist()) == ids.tolist()
    assert len(set(order[:bounds[1]]) & set(order[bounds[1]:])) == 0
    fixed, bounds, epochs = training_schedule(ids,'fixed',5,4,73)
    assert epochs == 5 and np.array_equal(fixed,ids)
    with pytest.raises(ValueError):
        training_schedule(ids,'fresh',2,0,73)


def test_fixed_subset_covers_all_acquisitions_without_leakage():
    manifest = {'samples': [{'name': f'sample_{i:02d}', 'offset': 100*i, 'shots': 100}
                            for i in range(53)]}
    train = np.concatenate([np.arange(100*i, 100*i+80) for i in range(53)])
    original = train.copy()
    chosen = fixed_training_subset(train, manifest, 1200, 73)
    assert len(chosen) == len(np.unique(chosen)) == 1200
    assert np.isin(chosen, train).all()
    np.testing.assert_array_equal(train, original)
    counts = np.bincount(chosen // 100, minlength=53)
    assert counts.min() == 22 and counts.max() == 23
    np.testing.assert_array_equal(chosen, fixed_training_subset(train[::-1], manifest, 1200, 73))
    assert not np.array_equal(chosen, fixed_training_subset(train, manifest, 1200, 74))
    for limit in (None, len(train), len(train)+100):
        np.testing.assert_array_equal(train, fixed_training_subset(train, manifest, limit, 73))
    with pytest.raises(ValueError, match='one shot per acquisition'):
        fixed_training_subset(train, manifest, 52, 73)


def test_fixed_subset_redistributes_quota_for_small_acquisitions():
    manifest = {'samples': [{'name': str(i), 'offset': 100*i, 'shots': 100} for i in range(3)]}
    train = np.array([0, 100, 101, 102, *range(200, 210)])
    chosen = fixed_training_subset(train, manifest, 9, 73)
    assert np.bincount(chosen // 100).tolist() == [1, 3, 5]
    for invalid in (0, -1, True, 3.5):
        with pytest.raises(ValueError):
            fixed_training_subset(train, manifest, invalid, 73)
    with pytest.raises(ValueError, match='Each acquisition'):
        fixed_training_subset(train[1:], manifest, 9, 73)


def test_all_53_sample_folders_are_included(acquisition_collection):
    root = acquisition_collection
    for index in range(3, 53):
        shutil.copytree(root/'sample_00', root/f'sample_{index:02d}')
    manifest = inspect_collection(root, expected_samples=53)
    assert [s['name'] for s in manifest['samples']] == [f'sample_{i:02d}' for i in range(53)]
    assert manifest['total_shots'] == 10600
    events, labels = read_collection(root, {'collection': manifest})
    assert len(events) == len(labels) == 10600
    sample, local = events.locate([0, 199, 200, 10599])
    assert sample.tolist() == [0,0,1,52] and local.tolist() == [0,199,0,199]


def test_collection_timing_aggregation_uses_raw_samples():
    from core.GCN.report import aggregate_measurements
    from core.GCN.benchmark import latency_statistics
    rows = [{'execution_device':'/CPU:0','input_placement':'host','scope':'model_only',
             'latency':latency_statistics(times),
             'throughput':{'completed_shots':10,'elapsed_seconds':seconds,'actual_batch_sizes':[5,5]},
             'memory':{'sampled_peak_rss_bytes':peak}}
            for times, seconds, peak in [([1,2,3],1,100),([100],3,200)]]
    result = aggregate_measurements(rows)[0]
    assert result['latency']['median'] == 2.5
    assert result['throughput']['shots_per_second'] == 5
    assert result['memory']['sampled_peak_rss_bytes'] == 200


def test_collection_development_to_reports(acquisition_collection, tmp_path):
    root = acquisition_collection
    args = make_parser().parse_args(['--data-root',str(root),'--expected-samples','3',
                                    '--training-mode','fresh','--fresh-shots-per-epoch','100',
                                    '--run-root',str(tmp_path/'runs'),'--epochs','2','--batch-size','32',
                                    '--hidden-units','8','--dense-units','8','--calibration-fraction','.05'])
    run = train(args)
    config = json.loads((run/'configuration.json').read_text())
    assert config['training_mode'] == 'fresh_dataset'
    assert [s['number_of_nodes'] for s in config['collection']['samples']] == [7,7,8]
    indices = read_indices(run,600)
    used = training_ids(run,indices)
    assert len(used) == 200 and np.isin(used,indices['train']).all()
    exposure = json.loads((run/'training_exposure.json').read_text())
    assert exposure['unused_training_shots'] == 280
    assert not (run/'test_predictions.npz').exists()
    model,bank,_ = load_run(run)
    events,labels = read_collection(root,config)
    ids = np.array([401,1,202,402,2])
    p = predict_records(model,bank,events[ids],2)
    np.testing.assert_allclose(p, np.array([predict_records(model,bank,events[[i]],1)[0] for i in ids]), atol=1e-6)
    args.training_mode='fixed'; args.epochs=1; args.split_from=run
    fixed = train(args)
    assert len(training_ids(fixed,read_indices(fixed,600))) == 480
    for key, values in indices.items():
        np.testing.assert_array_equal(values,read_indices(fixed,600)[key])
    pruned = compress(run,root,tmp_path/'pruned',hidden_units=4,dense_units=4,dense2_units=2,
                      epochs=1,batch_size=32,max_val_bce_increase=1)
    mixed = compress(pruned,root,tmp_path/'mixed',mode='mixed_float16',max_val_bce_increase=1)
    fit_gate(mixed,root,.99,method='none',min_accepted=1)
    report = evaluate(mixed,root,references={'perfect':'obs_flips_actual.b8'},
                      gate_path=mixed/'gate.json',live_dem='dems/live.dem',bootstrap=20)
    assert report['results']['perfect']['incorrect'] == 0
    assert len(report['per_sample']) == 3
    assert all(v['results']['neural']['shots'] == 20 for v in report['per_sample'].values())
    with np.load(mixed/'evaluation_test.predictions.npz') as saved:
        assert set(saved['sample_names']) == {'sample_00','sample_01','sample_02'}
        assert np.max(saved['local_shot_indices']) < 200
    legacy = compare(root,mixed,{'perfect':'obs_flips_actual.b8'})
    assert legacy['results']['perfect']['incorrect'] == 0
    timing = benchmark(mixed,root,warmup=1,repetitions=2,num_shots=2,throughput_repetitions=1,
                       throughput_batch_size=2,gate_path=mixed/'gate.json',live_dem='dems/live.dem')
    assert timing['sample_count'] == 3
    assert {v['sample'] for v in timing['measurements']} == {'sample_00','sample_01','sample_02'}
    write_json(mixed/'benchmark_cpu.json',timing)
    out = export_reports([mixed/'evaluation_test.json'],[mixed/'benchmark_cpu.json'],tmp_path/'reports')
    assert (out/'prediction_table.tex').is_file()
    assert r'\endfirsthead' in (out/'prediction_table.tex').read_text()
    assert (out/'per_sample_predictions.csv').read_text().count('sample_02') == 4
    assert 'sample_02' in (out/'graph_table.tex').read_text().replace('\\_\\allowbreak ', '_')

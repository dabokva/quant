from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import pytest

from benchmark_config import CONFIG
from core.GCN import experiments
from test_collection import acquisition_collection, experimental_data


def small_config(root, output):
    c = deepcopy(CONFIG)
    c.update(data_root=str(root), output_root=str(output), expected_samples=3,
             device='cpu', epochs=3, fresh_shots_per_epoch=120, fixed_training_shots=120,
             max_hours=None, bootstrap=10)
    c['model'].update(hidden_units=8, dense_units=8, gcn_layers=1)
    c['early_stopping'].update(patience=1, min_delta=1e6)
    c['pruning'].update(hidden_units=4, dense_units=4, dense2_units=2,
                        epochs=3, max_val_bce_increase=10)
    c['pruning']['early_stopping'].update(patience=1, min_delta=1e6)
    c['gate'].update(method='none', target_error=.99, min_accepted=1)
    c['benchmark'].update(device='cpu', warmup=1, repetitions=2,
                         num_shots=2, throughput_repetitions=1, throughput_batch_size=2)
    return c


def test_requested_matrix_and_validation(tmp_path):
    config = experiments.prepare_configuration(CONFIG, tmp_path)
    plan = experiments.study_plan(config)
    assert len(plan) == 12
    assert {p['batch_size'] for p in plan} == {64, 128}
    assert {p['variant'] for p in plan} == {'baseline', 'pruned64', 'mixed64'}
    assert config['epochs'] == 100 and config['pruning']['epochs'] == 20
    assert config['benchmark']['warmup'] == 50 and config['benchmark']['repetitions'] == 500
    assert config['data_root'] == str(tmp_path/'data')
    assert config['fixed_training_shots'] == 120000
    legacy = deepcopy(CONFIG)
    legacy.pop('fixed_training_shots')
    assert experiments.prepare_configuration(legacy)['fixed_training_shots'] is None
    for mutate in (lambda c:c.update(seeds=[73,74]), lambda c:c.update(batch_sizes=[64,64]),
                   lambda c:c['early_stopping'].update(patience=-1),
                   lambda c:c['model'].update(typo=1), lambda c:c.update(max_hours=float('nan')),
                   lambda c:c.update(fixed_training_shots=True), lambda c:c.update(fixed_training_shots=52),
                   lambda c:c.update(fixed_training_shots=1.5)):
        invalid = deepcopy(CONFIG)
        mutate(invalid)
        with pytest.raises(ValueError):
            experiments.prepare_configuration(invalid)


def test_real_matrix_deadline_resume_and_final_freeze(acquisition_collection, tmp_path):
    c = small_config(acquisition_collection, tmp_path/'study')
    calls, interrupt = [], [True]
    def execute(kind, payload, deadline):
        calls.append((kind, (payload.get('item') or {}).get('key')))
        if kind == 'gate' and interrupt[0]:
            interrupt[0] = False
            raise experiments.TimeBudgetExceeded('test deadline')
        if kind in ('test', 'benchmark'):
            assert (tmp_path/'study/final_selection.json').exists()
        return experiments.perform_stage(kind, payload)
    path = experiments.run_study(c, _executor=execute)
    partial = experiments.read(path)
    assert partial['status'] == 'time_limit'
    assert not (path.parent/'final_selection.json').exists()
    first = c['training_modes'][0]+'/batch_64/baseline'
    first_path = partial['runs'][first]['path']
    assert partial['stages'][first+'/validation']['status'] == 'complete'

    experiments.run_study(c, _executor=execute)
    manifest = experiments.read(path)
    assert manifest['status'] == 'complete' and len(manifest['runs']) == 12
    assert manifest['runs'][first]['path'] == first_path
    assert calls.count(('train', first)) == 1
    assert calls.count(('validation', first)) == 1
    partitions = None
    fixed_ids = None
    for item in manifest['plan']:
        run = Path(manifest['runs'][item['key']]['path'])
        config = experiments.read(run/'configuration.json')
        assert config['batch_size'] == item['batch_size']
        assert config['random_seed'] == 73 and config['experiment_label'] == item['key']
        with np.load(run/'shot_indices.npz') as archive:
            current = {name:archive[name] for name in archive.files}
        if partitions is None:
            partitions = current
        for name in partitions:
            np.testing.assert_array_equal(partitions[name], current[name])
        if item['mode'] == 'fixed' or item['variant'] != 'baseline':
            with np.load(run/'training_schedule.npz') as saved:
                selected = saved['order']
            if fixed_ids is None:
                fixed_ids = selected
            np.testing.assert_array_equal(fixed_ids, selected)
            assert len(selected) == 120 and np.isin(selected, current['train']).all()
            assert np.bincount(selected // 200).tolist() == [40, 40, 40]
        if item['variant'] != 'mixed64':
            progress = experiments.read(run/'training_progress.json')
            assert progress['completed_epochs'] == 2
            assert all(e['elapsed_seconds_including_validation'] > 0 for e in progress['epochs'])
            exposure = experiments.read(run/'training_exposure.json')
            expected = 240 if item['mode'] == 'fresh' and item['variant'] == 'baseline' else 120
            assert exposure['unique_training_shots_seen'] == expected
            assert exposure['reserved_training_shots'] == 480
    table = (path.parent/'results/predictions.csv').read_text()
    assert 'batch_64' in table and 'batch_128' in table and '(seed ' not in table
    assert (path.parent/'results/variant_summary.csv').is_file()
    assert not (path.parent/'results/seed_summary.csv').exists()
    before = len(calls)
    experiments.run_study(c, _executor=execute)
    assert [kind for kind,_ in calls[before:]] == ['freeze', 'report']
    changed = deepcopy(c)
    changed['epochs'] += 1
    with pytest.raises(ValueError, match='configuration differs'):
        experiments.run_study(changed, _executor=execute)
    changed = deepcopy(c)
    changed['fixed_training_shots'] += 1
    with pytest.raises(ValueError, match='configuration differs'):
        experiments.run_study(changed, _executor=execute)
    gate = Path(first_path)/'gate.json'
    experiments.write(gate, {**experiments.read(gate), 'tampered': True})
    before = len(calls)
    with pytest.raises(ValueError, match='changed after test access'):
        experiments.run_study(c, _executor=execute)
    assert [kind for kind,_ in calls[before:]] == ['freeze']


def test_live_worker_is_stopped_at_deadline():
    before = {p.pid for p in multiprocessing.active_children()}
    with pytest.raises(experiments.TimeBudgetExceeded):
        experiments.execute_stage('train', {}, time.monotonic()+.02)
    assert {p.pid for p in multiprocessing.active_children()} == before


def test_device_models_are_reused_and_each_graph_is_rebound(monkeypatch):
    from core.GCN import benchmark as module
    loaded, bound = [], []
    class Bank:
        def bind(self, model, index):
            bound.append(index)
            return index
    model = object()
    monkeypatch.setattr(module, 'load_run', lambda *args: loaded.append(args) or (model, Bank(), {}))
    context = module.CollectionBenchmark(None, None, {}, None, None, {})
    for sample in range(3):
        assert context.on_device('/GPU:0', 'run', 'weights', sample) == (model, sample)
    assert len(loaded) == 1 and bound == [0,1,2]

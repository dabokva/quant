from copy import deepcopy
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import re
import time
import traceback
from types import SimpleNamespace

from core.GCN.progress import announce

PROJECT = Path(__file__).resolve().parents[2]
OPERATIONAL = {'action', 'resume', 'max_hours'}

class TimeBudgetExceeded(RuntimeError):
    pass


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def prepare_configuration(config, project_root=PROJECT):
    config = deepcopy(config)
    config.setdefault('fixed_training_shots', None)
    def keys(value, names, name):
        if not isinstance(value, dict) or set(value) != set(names.split()):
            raise ValueError(f'{name}: expected keys {names}')
    keys(config, 'data_root output_root configuration expected_samples action resume max_hours '
         'random_seed split_seed training_modes batch_sizes device epochs fresh_shots_per_epoch fixed_training_shots '
         'calibration_fraction early_stopping model pruning mixed_precision gate bootstrap '
         'references reference_format live_dem benchmark', 'configuration')
    keys(config['early_stopping'], 'patience min_delta', 'early_stopping')
    keys(config['model'], 'hidden_units gcn_layers dense_units dropout learning_rate', 'model')
    keys(config['pruning'], 'enabled hidden_units dense_units dense2_units epochs learning_rate '
         'max_val_bce_increase early_stopping', 'pruning')
    keys(config['pruning']['early_stopping'], 'patience min_delta', 'pruning.early_stopping')
    keys(config['gate'], 'target_error method alpha min_accepted', 'gate')
    keys(config['benchmark'], 'enabled device placement num_shots warmup repetitions '
         'throughput_batch_size throughput_repetitions', 'benchmark')
    def positive(value, name, integer=True):
        valid_type = type(value) is int if integer else type(value) in (int, float)
        if not valid_type or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be a positive {"integer" if integer else "number"}')
    for name in ('expected_samples', 'epochs', 'fresh_shots_per_epoch', 'bootstrap'):
        positive(config[name], name)
    if config['fixed_training_shots'] is not None:
        positive(config['fixed_training_shots'], 'fixed_training_shots')
        if config['fixed_training_shots'] < config['expected_samples']:
            raise ValueError('fixed_training_shots must include at least one shot per acquisition')
    for name in ('random_seed', 'split_seed'):
        if type(config[name]) is not int or not 0 <= config[name] < 2**31:
            raise ValueError(f'{name} must be an integer in [0, 2**31)')
    for value in (config['resume'], config['mixed_precision'], config['pruning']['enabled'], config['benchmark']['enabled']):
        if type(value) is not bool:
            raise ValueError('resume, mixed_precision and enabled settings must be boolean')
    if config['action'] not in ('all', 'develop', 'final', 'report'):
        raise ValueError('action must be all, develop, final, or report')
    if config['max_hours'] is not None:
        positive(config['max_hours'], 'max_hours', False)
    for name in ('training_modes', 'batch_sizes'):
        values = config[name]
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise ValueError(f'{name} must be a nonempty list without duplicates')
    if any(mode not in ('fixed', 'fresh') for mode in config['training_modes']):
        raise ValueError('training_modes accepts fixed and fresh only')
    for value in config['batch_sizes']:
        positive(value, 'batch_sizes')
    if config['device'] not in ('cpu', 'gpu'):
        raise ValueError('device must be cpu or gpu')
    if not 0 < config['calibration_fraction'] < .1:
        raise ValueError('calibration_fraction must be between 0 and 0.1')
    for section in (config['early_stopping'], config['pruning']['early_stopping']):
        if type(section['patience']) is not int or section['patience'] < 0:
            raise ValueError('Early-stopping patience must be a nonnegative integer')
        if not math.isfinite(section['min_delta']) or section['min_delta'] < 0:
            raise ValueError('Early-stopping min_delta must be finite and nonnegative')
    model, pruning, gate, bench = (config[k] for k in ('model', 'pruning', 'gate', 'benchmark'))
    for name in ('hidden_units', 'gcn_layers', 'dense_units'):
        positive(model[name], f'model.{name}')
    if model['dense_units'] < 2 or not 0 <= model['dropout'] < 1:
        raise ValueError('Invalid dense_units or dropout')
    positive(model['learning_rate'], 'model.learning_rate', False)
    for name in ('hidden_units', 'dense_units', 'dense2_units', 'epochs'):
        positive(pruning[name], f'pruning.{name}')
    positive(pruning['learning_rate'], 'pruning.learning_rate', False)
    if pruning['enabled'] and (pruning['hidden_units'] > model['hidden_units'] or
            pruning['dense_units'] > model['dense_units'] or pruning['dense2_units'] > model['dense_units']//2):
        raise ValueError('Pruned widths cannot exceed the baseline widths')
    if not math.isfinite(pruning['max_val_bce_increase']) or pruning['max_val_bce_increase'] < 0:
        raise ValueError('Invalid pruning validation tolerance')
    if config['mixed_precision'] and not pruning['enabled']:
        raise ValueError('mixed_precision requires the pruned64 source')
    if not 0 < gate['target_error'] < 1 or not 0 < gate['alpha'] < 1:
        raise ValueError('Invalid gate error target or alpha')
    positive(gate['min_accepted'], 'gate.min_accepted')
    if gate['method'] not in ('temperature', 'none'):
        raise ValueError('Invalid gate method')
    if bench['device'] not in ('cpu', 'gpu', 'both') or bench['placement'] not in ('host', 'device', 'both'):
        raise ValueError('Invalid benchmark device or placement')
    for name in ('num_shots', 'warmup', 'repetitions', 'throughput_batch_size', 'throughput_repetitions'):
        positive(bench[name], f'benchmark.{name}')
    if not isinstance(config['references'], dict) or any(not isinstance(p, str) or not p for p in config['references'].values()):
        raise ValueError('references must map decoder names to file paths')
    if config['reference_format'] not in ('b8', '01'):
        raise ValueError('reference_format must be b8 or 01')
    if config['live_dem'] is not None and not isinstance(config['live_dem'], str):
        raise ValueError('live_dem must be a path string or None')
    for name in ('data_root', 'output_root'):
        path = Path(config[name]).expanduser()
        config[name] = str((Path(project_root)/path if not path.is_absolute() else path).resolve())
    return config


def study_plan(config):
    variants = [('baseline', None)]
    if config['pruning']['enabled']:
        variants.append(('pruned64', 'baseline'))
    if config['mixed_precision']:
        variants.append(('mixed64', 'pruned64'))
    return [{'key': f'{mode}/batch_{batch}/{name}', 'mode': mode, 'batch_size': batch,
             'variant': name, 'source': f'{mode}/batch_{batch}/{parent}' if parent else None}
            for mode in config['training_modes'] for batch in config['batch_sizes'] for name, parent in variants]


def _stamp(run, item):
    path = Path(run)/'configuration.json'
    config = read(path)
    config.update(experiment_label=item['key'], study_protocol='single_seed_batch_comparison')
    write(path, config)


def perform_stage(kind, payload):
    from core.GCN.run_io import write_json, run_identity, file_sha256
    config, item = payload['config'], payload.get('item')
    data = Path(config['data_root'])
    run = Path(payload['run']) if payload.get('run') else None
    if kind == 'train':
        from core.GCN.collection_training import train_collection
        args = SimpleNamespace(data_root=data, configuration=config['configuration'],
            expected_samples=config['expected_samples'], run_root=Path(payload['run_root']),
            seed=config['random_seed'], split_seed=config['split_seed'],
            split_from=Path(payload['split_from']) if payload.get('split_from') else None,
            training_mode=item['mode'], batch_size=item['batch_size'], epochs=config['epochs'],
            fresh_shots_per_epoch=config['fresh_shots_per_epoch'], device=config['device'],
            fixed_training_shots=config['fixed_training_shots'],
            calibration_fraction=config['calibration_fraction'], evaluate_test=False,
            early_stopping_patience=config['early_stopping']['patience'],
            early_stopping_min_delta=config['early_stopping']['min_delta'], **config['model'])
        run = train_collection(args)
        _stamp(run, item)
        return str(run)
    if kind == 'compress':
        from core.GCN.compression import compress
        options = {k:v for k,v in config['pruning'].items() if k not in ('enabled', 'early_stopping')}
        run = compress(payload['source'], data, payload['output'],
            mode='prune' if item['variant'] == 'pruned64' else 'mixed_float16',
            batch_size=item['batch_size'], seed=config['random_seed'], device=config['device'],
            fixed_training_shots=config['fixed_training_shots'],
            early_stopping_patience=config['pruning']['early_stopping']['patience'],
            early_stopping_min_delta=config['pruning']['early_stopping']['min_delta'], **options)
        _stamp(run, item)
        return str(run)
    if kind in ('validation', 'test'):
        from core.GCN.evaluate import evaluate
        final = kind == 'test'
        evaluate(run, data, subset='test' if final else 'validation',
            batch_size=item['batch_size'], device=config['device'], bootstrap=config['bootstrap'],
            seed=config['random_seed'], gate_path=run/'gate.json' if final else None,
            references=config['references'] if final else None,
            reference_format=config['reference_format'], live_dem=config['live_dem'] if final else None)
    elif kind == 'gate':
        from core.GCN.calibration import fit_gate
        fit_gate(run, data, batch_size=item['batch_size'], device=config['device'],
                 seed=config['random_seed'], **config['gate'])
    elif kind == 'freeze':
        frozen = {'runs': list(payload['runs']), 'configuration_sha256': payload['signature'],
            'identities': {k:run_identity(p) for k,p in payload['runs'].items()},
            'gate_hashes': {k:file_sha256(Path(p)/'gate.json') for k,p in payload['runs'].items()}}
        lock = Path(config['output_root'])/'final_selection.json'
        if lock.exists() and read(lock) != frozen:
            raise ValueError('Final selection or model contents changed after test access')
        write_json(lock, frozen)
    elif kind == 'benchmark':
        from core.GCN.benchmark import benchmark
        half = read(run/'configuration.json')['precision'] == 'mixed_float16'
        options = {k:v for k,v in config['benchmark'].items() if k != 'enabled'}
        result = benchmark(run, data, seed=config['random_seed'], gate_path=run/'gate.json',
            live_dem=config['live_dem'], rtol=1e-3 if half else 1e-5, atol=1e-4 if half else 1e-6, **options)
        write_json(run/'benchmark_final.json', result)
    elif kind == 'report':
        from core.GCN.report import export_reports
        export_reports(payload['evaluations'], payload['benchmarks'], Path(config['output_root'])/'results')
    else:
        raise ValueError(f'Unknown study stage: {kind}')
    return None


def _worker(kind, payload, connection):
    try:
        connection.send({'result': perform_stage(kind, payload)})
    except BaseException:
        connection.send({'error': traceback.format_exc()})
    finally:
        connection.close()


def execute_stage(kind, payload, deadline):
    """Enforce the deadline in the parent, outside the timed inference loops."""
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeBudgetExceeded('Study time budget reached')
    context = multiprocessing.get_context('spawn')
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(kind, payload, sender))
    process.start()
    sender.close()
    try:
        while True:
            remaining = None if deadline is None else deadline-time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeBudgetExceeded(f'Time budget reached during {kind}')
            if receiver.poll(.25 if remaining is None else min(.25, remaining)):
                try:
                    message = receiver.recv()
                except EOFError:
                    raise RuntimeError(f'{kind} worker exited without a result (possibly out of memory)') from None
                process.join(timeout=1)
                if 'error' in message:
                    raise RuntimeError(message['error'])
                return message['result']
            if not process.is_alive():
                raise RuntimeError(f'{kind} worker exited with code {process.exitcode}')
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
        receiver.close()
        process.close()


def run_study(config, project_root=PROJECT, _executor=None):
    config = prepare_configuration(config, project_root)
    plan, root, data = study_plan(config), Path(config['output_root']), Path(config['data_root'])
    samples = [p for p in data.glob('sample_*') if p.is_dir() and re.fullmatch(r'sample_\d+', p.name)]
    if len(samples) != config['expected_samples']:
        raise ValueError(f'Found {len(samples)} sample folders in {data}; expected {config["expected_samples"]}')
    relative = Path(config['configuration'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('configuration must be a relative path inside each sample')
    for sample in samples:
        for name in ('circuit_ideal.stim', 'detection_events.b8', 'obs_flips_actual.b8', 'dems/correlations.dem'):
            if not (sample/relative/name).is_file():
                raise ValueError(f'Missing data file: {sample/relative/name}')
    scientific = {k:v for k,v in config.items() if k not in OPERATIONAL
                  and not (k == 'fixed_training_shots' and v is None)}
    signature = hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()
    manifest_path = root/'manifest.json'
    if manifest_path.exists():
        if not config['resume']:
            raise ValueError('Output already exists; enable resume or choose another output_root')
        manifest = read(manifest_path)
        if manifest.get('schema_version') != 3 or manifest['configuration_sha256'] != signature:
            raise ValueError('Saved study configuration differs; use a new output_root for changed parameters')
    else:
        if config['action'] in ('final', 'report'):
            raise ValueError('No developed study exists in output_root')
        if root.exists() and any(root.iterdir()):
            raise ValueError('output_root contains unrelated files; choose an empty directory')
        root.mkdir(parents=True, exist_ok=True)
        manifest = {'schema_version': 3, 'configuration': config, 'configuration_sha256': signature,
                    'plan': plan, 'runs': {}, 'stages': {}, 'invocations': [], 'status': 'pending'}
        write(manifest_path, manifest)
    start = time.monotonic()
    deadline = start+config['max_hours']*3600 if config['max_hours'] is not None else None
    executor = _executor or execute_stage
    invocation = {'action': config['action'], 'max_hours': config['max_hours'], 'started_unix': time.time()}
    manifest['invocations'].append(invocation)
    manifest['status'] = 'running'
    write(manifest_path, manifest)
    announce(f'Study: {len(plan)} models; modes={config["training_modes"]}; batches={config["batch_sizes"]}; one seed={config["random_seed"]}')

    def stage(key, kind, payload, repeat=False):
        previous = manifest['stages'].get(key, {})
        if previous.get('status') == 'complete' and not repeat:
            announce(f'{key}: complete, skipped')
            return previous.get('result')
        record = {'status': 'running', 'attempt': previous.get('attempt', 0)+1}
        manifest['stages'][key] = record
        write(manifest_path, manifest)
        announce(f'Stage: {key}')
        tick = time.monotonic()
        try:
            result = executor(kind, {'config': config, **payload}, deadline)
        except BaseException as error:
            record.update(status='incomplete', error=str(error), elapsed_seconds=time.monotonic()-tick)
            write(manifest_path, manifest)
            raise
        record.update(status='complete', elapsed_seconds=time.monotonic()-tick, result=result)
        write(manifest_path, manifest)
        return result

    try:
        frozen = (root/'final_selection.json').exists()
        if config['action'] == 'develop' and frozen:
            raise ValueError('Development is frozen after final selection')
        if config['action'] in ('all', 'develop') and not frozen:
            for item in plan:
                key = item['key']
                entry = manifest['runs'].get(key)
                if entry and entry.get('development_complete'):
                    continue
                if item['source'] and not manifest['runs'][item['source']]['eligible']:
                    manifest['runs'][key] = {'eligible': False, 'development_complete': True, 'status': 'source_ineligible'}
                    write(manifest_path, manifest)
                    continue
                if not entry:
                    if item['source']:
                        parent = root/key
                        parent.mkdir(parents=True, exist_ok=True)
                        attempt = manifest['stages'].get(key+'/model', {}).get('attempt', 0)+1
                        run = stage(key+'/model', 'compress', {'item': item,
                            'source': manifest['runs'][item['source']]['path'], 'output': str(parent/f'attempt_{attempt}')})
                    else:
                        split_from = next((e['path'] for k,e in manifest['runs'].items() if k.endswith('/baseline')), None)
                        run = stage(key+'/model', 'train', {'item': item, 'run_root': str(root/key), 'split_from': split_from})
                    eligible = read(Path(run)/'configuration.json').get('validation_selection', {}).get('eligible', True)
                    entry = manifest['runs'][key] = {'path': run, 'eligible': eligible, 'status': 'trained'}
                    write(manifest_path, manifest)
                if entry['eligible']:
                    stage(key+'/validation', 'validation', {'item': item, 'run': entry['path']})
                    stage(key+'/gate', 'gate', {'item': item, 'run': entry['path']})
                entry.update(development_complete=True, status='eligible' if entry['eligible'] else 'validation_rejected')
                write(manifest_path, manifest)
        if config['action'] == 'develop':
            manifest['status'] = 'development_complete'
        else:
            if any(not manifest['runs'].get(item['key'], {}).get('development_complete') for item in plan):
                raise ValueError('Finish development before final evaluation or reporting')
            selected = {item['key']: manifest['runs'][item['key']]['path'] for item in plan
                        if manifest['runs'][item['key']]['eligible']}
            stage('freeze', 'freeze', {'runs': selected, 'signature': signature}, repeat=True)
            if config['action'] != 'report':
                for item in plan:
                    key = item['key']
                    if key not in selected:
                        continue
                    stage(key+'/test', 'test', {'item': item, 'run': selected[key]})
                    if config['benchmark']['enabled']:
                        stage(key+'/benchmark', 'benchmark', {'item': item, 'run': selected[key]})
            evaluations = [str(Path(path)/'evaluation_test.json') for key,path in selected.items()
                           if manifest['stages'].get(key+'/test', {}).get('status') == 'complete']
            benchmarks = [str(Path(path)/'benchmark_final.json') for key,path in selected.items()
                          if manifest['stages'].get(key+'/benchmark', {}).get('status') == 'complete']
            if not evaluations:
                raise ValueError('No completed test evaluations to export')
            stage('report', 'report', {'evaluations': evaluations, 'benchmarks': benchmarks}, repeat=True)
            complete = len(evaluations) == len(selected) and (not config['benchmark']['enabled'] or len(benchmarks) == len(selected))
            manifest['status'] = 'complete' if complete else 'partial_report'
    except TimeBudgetExceeded as error:
        manifest['status'] = 'time_limit'
        announce(f'{error}. Completed stages are saved. Run again to resume; the unfinished stage restarts.')
    except KeyboardInterrupt:
        manifest['status'] = 'interrupted'
        raise
    except Exception:
        manifest['status'] = 'failed'
        raise
    finally:
        invocation.update(elapsed_seconds=time.monotonic()-start, status=manifest['status'])
        write(manifest_path, manifest)
    announce(f'Study status: {manifest["status"]}; manifest: {manifest_path}')
    return manifest_path

from dataclasses import dataclass, field
from functools import wraps
import json
from pathlib import Path
import platform
import threading
import time

import numpy as np
import psutil
import tensorflow as tf

from core.GCN.progress import announce, stage
from core.GCN.calibration import gate_decisions, verify_gate, calibrate_tensor
from core.GCN.reference import MatchingDecoder
from core.GCN.run_io import (load_run, read_records, read_indices, run_identity, file_sha256)


@dataclass
class CollectionBenchmark:
    model: object
    bank: object
    config: dict
    events: object
    labels: object
    indices: dict
    device_models: dict = field(default_factory=dict)

    def on_device(self, device, run, checkpoint, sample_index):
        if device not in self.device_models:
            self.device_models[device] = load_run(run, checkpoint)
        model, bank, _ = self.device_models[device]
        return model, bank.bind(model, sample_index)


def materialize(value):
    return value.numpy() if hasattr(value, 'numpy') else np.asarray(value)


def strict_float32_execution(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        previous = tf.config.experimental.tensor_float_32_execution_enabled()
        tf.config.experimental.enable_tensor_float_32_execution(False)
        try:
            return function(*args, **kwargs)
        finally:
            tf.config.experimental.enable_tensor_float_32_execution(previous)
    return wrapped


def verify_predictions(predictions, expected, rtol, atol, scope):
    predictions, expected = np.asarray(predictions), np.asarray(expected)
    if predictions.shape != expected.shape or predictions.ndim != 2:
        raise AssertionError(f'{scope}: prediction shapes do not match: '
                             f'{predictions.shape} versus {expected.shape}')
    if not np.isfinite(predictions).all() or not np.isfinite(expected).all():
        raise AssertionError(f'{scope}: non-finite inference output')
    if scope == 'detector_to_gate':
        np.testing.assert_array_equal(predictions[:, 1], expected[:, 1],
                                      err_msg=f'{scope}: CPU/GPU acceptance decisions differ')
    np.testing.assert_array_equal(predictions[:, 0] > .5, expected[:, 0] > .5,
                                  err_msg=f'{scope}: CPU/GPU logical predictions differ')
    np.testing.assert_allclose(predictions, expected, rtol=rtol, atol=atol,
                               err_msg=f'{scope}: CPU reference mismatch with TF32 disabled; '
                                       'check the GPU/compiler environment before changing tolerances')


def latency_statistics(elapsed):
    elapsed = np.asarray(elapsed, np.float64)
    return {'unit': 'microseconds_per_complete_shot', 'repetitions': len(elapsed),
            'mean': float(elapsed.mean()), 'median': float(np.median(elapsed)),
            'minimum': float(elapsed.min()), 'p95': float(np.percentile(elapsed, 95)),
            'p99': float(np.percentile(elapsed, 99)), 'samples_us': elapsed.tolist()}


def measure(infer, sample, warmup=100, repetitions=1000):
    if warmup < 1 or repetitions < 1:
        raise ValueError('Warm-up and timed repetitions must be positive')
    samples = sample if isinstance(sample, list) else [sample]
    if not samples:
        raise ValueError('At least one input is required')
    for index in range(warmup):
        materialize(infer(samples[index % len(samples)]))
    elapsed = []
    for index in range(repetitions):
        start = time.perf_counter_ns()
        materialize(infer(samples[index % len(samples)]))
        elapsed.append((time.perf_counter_ns() - start) / 1000)
    return latency_statistics(elapsed)


def memory_measurement(call, device, repetitions=3):
    process = psutil.Process()
    baseline = process.memory_info().rss
    rss = [baseline]
    stop = threading.Event()
    def sample():
        while not stop.wait(.005):
            rss.append(process.memory_info().rss)
    backend_reason = None
    try:
        tf.config.experimental.reset_memory_stats(device.lstrip('/'))
        backend_before = tf.config.experimental.get_memory_info(device.lstrip('/'))
    except (ValueError, tf.errors.OpError) as error:
        backend_before = None
        backend_reason = str(error)
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        for _ in range(repetitions):
            materialize(call())
    finally:
        rss.append(process.memory_info().rss)
        stop.set()
        sampler.join()
    backend = tf.config.experimental.get_memory_info(device.lstrip('/')) if backend_before else None
    return {'rss_baseline_bytes': baseline, 'sampled_peak_rss_bytes': max(rss),
            'sampled_rss_increase_bytes': max(rss) - baseline, 'rss_sample_interval_seconds': .005,
            'rss_scope': 'whole process, with model and input buffers already loaded; sampled lower bound',
            'tensorflow_allocator_before': backend_before, 'tensorflow_allocator_after': backend,
            'allocator_unavailable_reason': backend_reason}


def make_inference(model, mapper, num_detectors, num_nodes, device, gate=None, dynamic_batch=False):
    batch = None if dynamic_batch else 1
    @tf.function(input_signature=[tf.TensorSpec((batch, num_nodes, 13), tf.float32)])
    def model_only(features):
        with tf.device(device):
            return model(features, training=False)
    @tf.function(input_signature=[tf.TensorSpec((batch, num_detectors), tf.float32)])
    def detector_to_prediction(events):
        with tf.device(device):
            features, _ = mapper(events, tf.zeros((tf.shape(events)[0], 1), tf.float32))
            return model(features, training=False)
    result = {'model_only': model_only, 'detector_to_prediction': detector_to_prediction}
    if gate is not None:
        @tf.function(input_signature=[tf.TensorSpec((batch, num_detectors), tf.float32)])
        def detector_to_gate(events):
            with tf.device(device):
                q = calibrate_tensor(detector_to_prediction(events),
                                     tf.constant(gate['temperature'], tf.float32))
                accepted = (tf.maximum(q, 1-q) >= gate['threshold']) if gate['enabled'] else tf.zeros_like(q, tf.bool)
                return tf.concat([q, tf.cast(accepted, tf.float32)], axis=1)
        result['detector_to_gate'] = detector_to_gate
    return result


@strict_float32_execution
def benchmark(run_directory, data_path, device='cpu', placement='host',
              warmup=50, repetitions=500, shot_id=None, rtol=1e-5, atol=1e-6,
              checkpoint='restored.weights.h5', num_shots=32, seed=73,
              throughput_batch_size=8, throughput_repetitions=5, gate_path=None,
              live_dem=None, subset='test', _context=None, _sample_index=None):
    run_directory = Path(run_directory)
    if min(warmup, repetitions, num_shots, throughput_batch_size, throughput_repetitions) < 1:
        raise ValueError('Counts must be positive')
    if not np.isfinite([rtol, atol]).all() or min(rtol, atol) < 0:
        raise ValueError('Tolerances must be finite and nonnegative')
    if device not in ('cpu', 'gpu', 'both') or placement not in ('host', 'device', 'both'):
        raise ValueError('Unsupported device or placement')
    gpus = tf.config.list_physical_devices('GPU')
    if device in ('gpu', 'both') and not gpus:
        raise ValueError('GPU benchmark requested, but TensorFlow reports no GPU devices')
    if gate_path and checkpoint != 'restored.weights.h5':
        raise ValueError('The gate is tied to the default evaluated weights')
    if _context is None:
        from core.GCN.run_io import read_configuration
        initial_config = read_configuration(run_directory)
        if initial_config.get('collection'):
            with tf.device('/CPU:0'):
                context_model, context_bank, initial_config = load_run(run_directory, checkpoint)
                context_events, context_labels = read_records(data_path, initial_config)
                context_indices = read_indices(run_directory, len(context_labels))
            context = CollectionBenchmark(context_model, context_bank, initial_config,
                                          context_events, context_labels, context_indices)
            reports = []
            for index, sample in enumerate(initial_config['collection']['samples']):
                if shot_id is not None and not sample['offset'] <= shot_id < sample['offset']+sample['shots']:
                    continue
                announce(f'Benchmark sample {index+1}/{initial_config["sample_count"]}: {sample["name"]}')
                reports.append(benchmark(run_directory, data_path, device=device, placement=placement,
                    warmup=warmup, repetitions=repetitions, shot_id=shot_id, rtol=rtol, atol=atol,
                    checkpoint=checkpoint, num_shots=num_shots, seed=seed,
                    throughput_batch_size=throughput_batch_size, throughput_repetitions=throughput_repetitions,
                    gate_path=gate_path, live_dem=live_dem, subset=subset, _context=context, _sample_index=index))
            if not reports:
                raise ValueError('Requested shot does not belong to the collection')
            combined = dict(reports[0])
            combined['shot_indices'] = [i for report in reports for i in report['shot_indices']]
            combined['measurements'] = [row for report in reports for row in report['measurements']]
            combined['sample_count'] = len(reports)
            combined['sampling_rule'] = 'up to num_shots per sample, without replacement; sample graphs timed separately'
            combined['graphs_loaded_in_process'] = initial_config['sample_count']
            combined['model_loading'] = 'one model and graph bank per device; acquisition-specific tracing'
            return combined
    old_soft = tf.config.get_soft_device_placement()
    tf.config.set_soft_device_placement(False)
    try:
        with tf.device('/CPU:0'):
            if _context is None:
                cpu_model, cpu_mapper, config = load_run(run_directory, checkpoint)
                events, labels = read_records(data_path, config)
                indices = read_indices(run_directory, len(labels))
            else:
                cpu_model, bank, config = _context.model, _context.bank, _context.config
                events, labels, indices = _context.events, _context.labels, _context.indices
                sample_info = config['collection']['samples'][_sample_index]
                cpu_mapper = bank.bind(cpu_model, _sample_index)
                config = {**config, **{key: sample_info[key] for key in
                          ('number_of_detectors', 'number_of_nodes', 'number_of_edges')}}
            if subset not in ('validation', 'test'):
                raise ValueError('Benchmark subset must be validation or test')
            pool = indices[subset]
            if _context is not None:
                pool = pool[(pool >= sample_info['offset']) & (pool < sample_info['offset']+sample_info['shots'])]
            if shot_id is not None:
                if shot_id not in pool:
                    raise ValueError('Requested shot must belong to the selected saved subset')
                ids = np.asarray([shot_id], np.int64)
            else:
                ids = np.random.default_rng(seed).choice(pool, min(num_shots, len(pool)), replace=False)
            raw_events = events.read(_sample_index, ids) if _context is not None else events[ids]
            host_events = tf.constant(raw_events, tf.float32)
            host_features, _ = cpu_mapper(host_events, tf.zeros((len(ids), 1)))
            cpu_reference = np.concatenate([cpu_model(host_features[i:i+1], training=False).numpy()
                                            for i in range(len(ids))])
        gate = json.loads(Path(gate_path).read_text()) if gate_path else None
        if gate:
            verify_gate(gate, run_directory, indices)
        if live_dem and _context is not None:
            from core.GCN.reference import collection_path
            live_dem = collection_path(live_dem, data_path, sample_info)
        decoder = MatchingDecoder(live_dem, config['number_of_detectors']) if live_dem else None
        records = []
        devices = ['/CPU:0', '/GPU:0'] if device == 'both' else [f'/{device.upper()}:0']
        for execution_device in devices:
            with tf.device(execution_device):
                if execution_device == '/CPU:0':
                    model, mapper = cpu_model, cpu_mapper
                else:
                    if _context is not None:
                        model, mapper = _context.on_device(execution_device, run_directory, checkpoint, _sample_index)
                    else:
                        model, mapper, _ = load_run(run_directory, checkpoint)
                inference = make_inference(model, mapper, config['number_of_detectors'],
                                           config['number_of_nodes'], execution_device, gate)
                batched = make_inference(model, mapper, config['number_of_detectors'],
                                         config['number_of_nodes'], execution_device, gate, True)
            placements = ['host'] if execution_device == '/CPU:0' else (
                ['host', 'device'] if placement == 'both' else [placement])
            for input_placement in placements:
                input_device = '/CPU:0' if input_placement == 'host' else execution_device
                with tf.device(input_device):
                    features = tf.identity(host_features)
                    compact = tf.identity(host_events)
                    feature_samples = [tf.identity(features[i:i+1]) for i in range(len(ids))]
                    event_samples = [tf.identity(compact[i:i+1]) for i in range(len(ids))]
                for scope, infer in inference.items():
                    announce(f'Benchmark {execution_device} / {input_placement} / {scope}')
                    samples = feature_samples if scope == 'model_only' else event_samples
                    outputs = [infer(sample) for sample in samples]
                    if any(not value.device.upper().endswith(execution_device[1:]) for value in outputs):
                        raise RuntimeError('Inference output was placed on an unexpected device')
                    predictions = np.concatenate([value.numpy() for value in outputs])
                    expected = cpu_reference
                    if scope == 'detector_to_gate':
                        q, accepted = gate_decisions(cpu_reference[:, 0], gate)
                        expected = np.column_stack([q, accepted])
                    verify_predictions(predictions, expected, rtol, atol, scope)
                    with stage(f'Latency ({warmup} warm-up, {repetitions} timed calls)'):
                        latency = measure(infer, samples, warmup, repetitions)
                    memory = memory_measurement(lambda: infer(samples[0]), execution_device)
                    with tf.device(input_device):
                        source = features if scope == 'model_only' else compact
                        batches = [tf.identity(source[i:i+throughput_batch_size])
                                   for i in range(0, len(ids), throughput_batch_size)]
                    batch_infer = batched[scope]
                    for batch in batches:
                        batch_infer(batch).numpy()
                    announce(f'Throughput: {throughput_repetitions} passes')
                    start = time.perf_counter_ns()
                    completed = 0
                    for _ in range(throughput_repetitions):
                        for batch in batches:
                            batch_infer(batch).numpy()
                            completed += int(batch.shape[0])
                    seconds = (time.perf_counter_ns() - start) / 1e9
                    batch_memory = memory_measurement(lambda: batch_infer(batches[0]), execution_device)
                    records.append({'execution_device': execution_device, 'input_placement': input_placement,
                                    'input_device': samples[0].device, 'output_device': outputs[0].device,
                                    'variable_devices': sorted({v.value.device for v in model.variables}),
                                    'scope': scope, 'latency': latency, 'memory': memory,
                                    'max_absolute_difference_from_cpu': float(np.max(np.abs(predictions-expected))),
                                    'throughput': {'completed_shots': completed, 'elapsed_seconds': seconds,
                                                   'shots_per_second': completed / seconds,
                                                   'requested_batch_size': throughput_batch_size,
                                                   'actual_batch_sizes': [int(x.shape[0]) for x in batches],
                                                   'memory': batch_memory}})
                if gate and decoder:
                    announce(f'Benchmark {execution_device} / {input_placement} / live_hybrid')
                    infer_gate = inference['detector_to_gate']
                    samples = list(zip(event_samples, raw_events))
                    def hybrid(sample):
                        neural_input, original_events = sample
                        q, accepted = infer_gate(neural_input).numpy()[0]
                        return int(q > .5) if accepted else decoder.decode_one(original_events)
                    records.append({'execution_device': execution_device, 'input_placement': input_placement,
                                    'scope': 'live_hybrid', 'latency': measure(hybrid, samples, warmup, repetitions),
                                    'fallback': decoder.metadata})
        if decoder:
            announce('Benchmark /CPU:0 / live_reference')
            records.append({'execution_device': '/CPU:0', 'input_placement': 'host',
                            'scope': 'live_reference',
                            'latency': measure(decoder.decode_one, list(raw_events), warmup, repetitions),
                            'fallback': decoder.metadata})
        if _context is not None:
            for record in records:
                record.update({'sample': sample_info['name'], 'number_of_nodes': sample_info['number_of_nodes'],
                               'number_of_edges': sample_info['number_of_edges'], 'shot_indices': ids.tolist()})
        return {'schema_version': 2, 'run_directory': str(run_directory.resolve()),
                'run_identity': run_identity(run_directory, checkpoint), 'checkpoint': checkpoint,
                'gate_sha256': file_sha256(gate_path) if gate_path else None,
                'shot_indices': ids.tolist(), 'subset': subset, 'sampling_seed': seed,
                'sampling_rule': 'uniform without replacement; timed calls cycle through saved IDs',
                'batch_size': 1, 'warmup': warmup, 'precision': config.get('precision', 'float32'),
                'rtol': rtol, 'atol': atol, 'soft_device_placement': False,
                'gate_included': gate is not None, 'output_materialized_on_host': True,
                'hardware': {'platform': platform.platform(), 'processor': platform.processor(),
                             'machine': platform.machine(), 'logical_cpus': psutil.cpu_count(),
                             'physical_cpus': psutil.cpu_count(logical=False),
                             'system_memory_bytes': psutil.virtual_memory().total,
                             'devices': [d.name for d in tf.config.list_logical_devices()],
                             'gpu_details': [tf.config.experimental.get_device_details(d) for d in gpus]},
                'software': {'python': platform.python_version(), 'tensorflow': tf.__version__,
                             'numpy': np.__version__, 'tensorflow_build': tf.sysconfig.get_build_info()},
                'runtime': {'intra_op_threads': tf.config.threading.get_intra_op_parallelism_threads(),
                            'inter_op_threads': tf.config.threading.get_inter_op_parallelism_threads(),
                            'tensor_float_32_execution_enabled':
                                tf.config.experimental.tensor_float_32_execution_enabled(),
                            'float32_math_policy': 'TF32 disabled for reference checks and all timed execution'},
                'measurements': records}
    finally:
        tf.config.set_soft_device_placement(old_soft)

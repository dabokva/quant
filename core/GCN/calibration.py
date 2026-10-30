import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logit
import tensorflow as tf

from core.GCN.metrics import (binary_vector, probabilities_vector, probability_metrics,
                             upper_error_bound, classification_metrics)
from core.GCN.run_io import (load_run, read_records, read_indices, predict_records,
                             run_identity, write_json, select_device)


@tf.function(reduce_retracing=True)
def calibrate_tensor(probabilities, temperature):
    p = tf.clip_by_value(tf.cast(probabilities, tf.float32), 1e-7, 1 - 1e-7)
    return tf.sigmoid((tf.math.log(p) - tf.math.log1p(-p)) / tf.cast(temperature, tf.float32))


def calibrate_probabilities(probabilities, temperature):
    p = probabilities_vector(probabilities)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    with tf.device('/CPU:0'):
        return calibrate_tensor(tf.constant(p, tf.float32), tf.constant(temperature, tf.float32)).numpy()


def fit_temperature(labels, probabilities):
    labels = binary_vector(labels)
    p = probabilities_vector(probabilities, len(labels))
    if len(np.unique(labels)) != 2:
        raise ValueError('Temperature fitting needs both classes; reserve more calibration shots or use method none')
    logits = logit(np.clip(p, 1e-7, 1 - 1e-7))
    def loss(log_temperature):
        scaled = logits / np.exp(log_temperature)
        return np.mean(np.logaddexp(0, scaled) - labels * scaled)
    result = minimize_scalar(loss, bounds=(np.log(0.05), np.log(20.0)), method='bounded')
    if not result.success:
        raise ValueError('Temperature fitting did not converge')
    return float(np.exp(result.x))


def select_threshold(labels, calibrated, target_error, alpha=0.05, min_accepted=20,
                     thresholds=None):
    labels = binary_vector(labels)
    p = probabilities_vector(calibrated, len(labels)).astype(np.float32)
    if not 0 < target_error < 1 or not 0 < alpha < 1 or min_accepted < 1:
        raise ValueError('Invalid gate target, alpha or minimum accepted count')
    thresholds = np.linspace(0.5, 1.0, 101) if thresholds is None else np.asarray(thresholds)
    if thresholds.ndim != 1 or len(thresholds) == 0 or not np.isfinite(thresholds).all() or np.any((thresholds < .5) | (thresholds > 1)):
        raise ValueError('Thresholds must be a fixed grid in [0.5, 1]')
    thresholds = np.unique(np.asarray(thresholds, dtype=np.float32))
    rows = []
    for threshold in thresholds:
        mask = np.maximum(p, 1 - p) >= threshold
        count = int(mask.sum())
        errors = int(np.sum((p[mask] > .5) != labels[mask]))
        upper = upper_error_bound(errors, count, alpha / len(thresholds))
        rows.append({'threshold': float(threshold), 'accepted': count, 'incorrect': errors,
                     'coverage': count / len(labels) if len(labels) else 0.0,
                     'error_rate': errors / count if count else None,
                     'simultaneous_upper': upper,
                     'eligible': bool(count >= min_accepted and upper is not None and upper <= target_error)})
    eligible = [row for row in rows if row['eligible']]
    selected = max(eligible, key=lambda row: (row['accepted'], -row['threshold'])) if eligible else None
    return {'enabled': selected is not None, 'threshold': selected['threshold'] if selected else None,
            'target_error': target_error, 'alpha': alpha, 'min_accepted': min_accepted,
            'selection_method': 'one-sided Clopper-Pearson with Bonferroni over fixed grid',
            'selection': selected, 'calibration_curve': rows,
            'status': 'target_met_on_calibration_selection' if selected else 'no_supported_threshold'}


def gate_decisions(probabilities, gate):
    q = calibrate_probabilities(probabilities, gate['temperature'])
    accepted = ((np.maximum(q, 1 - q) >= np.float32(gate['threshold'])) if gate['enabled']
                else np.zeros(len(q), dtype=bool))
    return q, accepted


def gate_metrics(labels, probabilities, gate, fallback=None):
    labels = binary_vector(labels)
    q, accepted = gate_decisions(probabilities, gate)
    result = {'coverage': float(accepted.mean()) if len(labels) else None,
              'accepted': classification_metrics(labels[accepted], q[accepted] > .5),
              'rejected_shots': int((~accepted).sum()),
              'calibrated': probability_metrics(labels, q),
              'target_error': gate['target_error'], 'enabled': gate['enabled'],
              'threshold': gate['threshold']}
    if fallback is not None:
        fallback = binary_vector(fallback, 'fallback predictions')
        if fallback.shape != labels.shape:
            raise ValueError('Fallback predictions are misaligned')
        hybrid = np.where(accepted, q > .5, fallback)
        result['fallback_on_rejected'] = classification_metrics(labels[~accepted], fallback[~accepted])
        result['hybrid'] = classification_metrics(labels, hybrid)
    return result


def fit_gate(run_directory, data_path, target_error, method='temperature', alpha=.05,
             min_accepted=20, seed=73, batch_size=32, device='cpu', output=None):
    import tensorflow as tf
    if method not in ('temperature', 'none'):
        raise ValueError('Unsupported calibration method')
    with tf.device(select_device(device)):
        model, mapper, config = load_run(run_directory)
        events, labels = read_records(data_path, config)
        ids = read_indices(run_directory, len(labels))['calibration']
        if len(ids) < 4:
            raise ValueError('Reserve calibration shots during training before fitting a gate')
        order = np.random.default_rng(seed).permutation(ids)
        fit_ids, selection_ids = np.array_split(order, 2)
        p = predict_records(model, mapper, events[order], batch_size)
        split = len(fit_ids)
        temperature = fit_temperature(labels[fit_ids], p[:split]) if method == 'temperature' else 1.0
        q = calibrate_probabilities(p[split:], temperature)
        gate = select_threshold(labels[selection_ids], q, target_error, alpha, min_accepted)
    gate.update({'schema_version': 1, 'run_identity': run_identity(run_directory),
                 'temperature': temperature, 'method': method, 'seed': seed,
                 'fit_shot_indices': fit_ids.tolist(), 'selection_shot_indices': selection_ids.tolist(),
                 'selection_metrics_before': probability_metrics(labels[selection_ids], p[split:]),
                 'selection_metrics_after': probability_metrics(labels[selection_ids], q)})
    write_json(output or Path(run_directory) / 'gate.json', gate)
    return gate


def verify_gate(gate, run_directory, indices):
    if gate['run_identity'] != run_identity(run_directory):
        raise ValueError('Gate was fitted to different weights, precision, graph or partitions')
    fit_ids = np.asarray(gate['fit_shot_indices'])
    select_ids = np.asarray(gate['selection_shot_indices'])
    combined = np.concatenate([fit_ids, select_ids])
    if combined.dtype.kind not in 'iu' or len(set(combined.tolist())) != len(combined):
        raise ValueError('Gate calibration fit/selection identifiers overlap or are invalid')
    if set(combined.tolist()) != set(indices['calibration'].tolist()):
        raise ValueError('Gate must use only the reserved calibration subset')
    if gate['enabled'] and (gate['threshold'] is None or not .5 <= gate['threshold'] <= 1):
        raise ValueError('Invalid saved gate threshold')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--target-error', type=float, required=True)
    parser.add_argument('--method', choices=['temperature', 'none'], default='temperature')
    parser.add_argument('--alpha', type=float, default=.05)
    parser.add_argument('--min-accepted', type=int, default=20)
    parser.add_argument('--seed', type=int, default=73)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', choices=['cpu', 'gpu', 'auto'], default='cpu')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = fit_gate(args.run_dir, args.data_path, args.target_error, args.method,
                      args.alpha, args.min_accepted, args.seed, args.batch_size, args.device, args.output)
    print(f"Gate: {result['status']}; temperature={result['temperature']:.6g}; threshold={result['threshold']}")


if __name__ == '__main__':
    main()

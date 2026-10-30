import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import stim


def read_bits(path):
    return stim.read_shot_data_file(
        path=str(path), format='b8', num_observables=1,
    )[:, 0]


def check_ids(ids, number_of_shots, name):
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(f'{name} must contain integer shot indices')
    if not len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError(f'{name} must be nonempty and contain no duplicate indices')
    if np.any(ids < 0) or np.any(ids >= number_of_shots):
        raise ValueError(f'{name} contains indices outside this dataset')


def report(name, predictions, truth):
    errors = int(np.count_nonzero(predictions != truth))
    error_rate = errors / len(truth)
    print(f'{name:<32} {1 - error_rate:>10.4%} {error_rate:>12.4%} {errors:>9}')


def main():
    parser = argparse.ArgumentParser(description='Compare reference decoders on a saved test split')
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--prior', default='dem_correlations',
                        help='Subdirectory under obs_flips_predicted')
    args = parser.parse_args()

    with (args.run_dir / 'config.json').open() as handle:
        configuration = json.load(handle)
    if configuration.get('overfit_shots', 0):
        raise ValueError('Choose a full-training run, not an overfit-shots run')
    circuit_path = args.data_path / 'circuit_ideal.stim'
    circuit = stim.Circuit.from_file(str(circuit_path))
    if circuit.num_observables != 1:
        raise ValueError('This comparison expects exactly one logical observable')
    expected_hash = configuration.get('circuit_sha256')
    if expected_hash and hashlib.sha256(circuit_path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError('This circuit does not match the saved training run')

    labels = read_bits(args.data_path / 'obs_flips_actual.b8')
    with np.load(args.run_dir / 'splits.npz') as splits:
        train_ids, test_ids = splits['train'], splits['test']
    check_ids(train_ids, len(labels), 'Training split')
    check_ids(test_ids, len(labels), 'Test split')
    if np.intersect1d(train_ids, test_ids).size:
        raise ValueError('Training and test splits overlap')
    truth = labels[test_ids]

    prediction_directory = args.data_path / 'obs_flips_predicted' / args.prior
    prediction_paths = sorted(prediction_directory.glob('*.b8'))
    if not prediction_paths:
        raise FileNotFoundError(f'No reference prediction files found in {prediction_directory}')

    print(f'Total shots: {len(labels)}; held-out test shots: {len(test_ids)}')
    print(f'Test positive-label rate: {truth.mean():.4%}')
    print(f'Reference prior: {args.prior}\n')
    print(f'{"Predictor":<32} {"Accuracy":>10} {"Error rate":>12} {"Errors":>9}')
    constant = labels[train_ids].mean() >= 0.5
    report(f'Constant prediction: {int(constant)}', np.full(truth.shape, constant), truth)

    for path in prediction_paths:
        predictions = read_bits(path)
        if len(predictions) != len(labels):
            raise ValueError(f'{path.name}: {len(predictions)} predictions for {len(labels)} shots')
        report(path.stem, predictions[test_ids], truth)

    saved_predictions = args.run_dir / 'predictions.npz'
    if saved_predictions.exists():
        with np.load(saved_predictions) as saved:
            ids = saved['shot_ids']
            check_ids(ids, len(labels), 'Saved neural predictions')
            if not np.array_equal(np.sort(ids), np.sort(test_ids)):
                raise ValueError('Saved neural predictions use different test shots')
            saved_labels = saved['labels'].reshape(-1)
            if not np.array_equal(saved_labels, labels[ids]):
                raise ValueError('Actual labels differ from the labels saved during training')
            probabilities = saved['probabilities'].reshape(-1)
            if probabilities.shape != ids.shape or not np.all(np.isfinite(probabilities)):
                raise ValueError('Invalid saved neural probabilities')
            report('Neural predecoder', probabilities > 0.5, labels[ids])
    else:
        print('\nNo saved neural predictions yet; reference results are still available.')


if __name__ == '__main__':
    main()

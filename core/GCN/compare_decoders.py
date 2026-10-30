import argparse
import json
from pathlib import Path

import numpy as np
import stim

from core.GCN.data_loader import PARTITION_NAMES, validate_shot_indices
from core.GCN.run_io import read_records, training_ids


def prediction_metrics(labels, predictions):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if labels.shape != predictions.shape or labels.ndim != 1 or len(labels) == 0:
        raise ValueError('Labels and predictions must be aligned nonempty vectors')
    if not np.isin(labels, (0, 1)).all() or not np.isin(predictions, (0, 1)).all():
        raise ValueError('Labels and predictions must be binary')
    incorrect = int(np.count_nonzero(labels != predictions))
    error_rate = incorrect / len(labels)
    return {'shots': len(labels), 'incorrect': incorrect,
            'accuracy': 1.0 - error_rate, 'error_rate': error_rate}


def load_neural_predictions(path, shot_indices, labels):
    with np.load(path, allow_pickle=False) as saved:
        ids = saved['shot_indices']
        if ids.dtype.kind not in 'iu' or not np.array_equal(ids, shot_indices):
            raise ValueError('Neural shot identifiers do not match the saved test order')
        if not np.array_equal(saved['labels'], labels):
            raise ValueError('Neural labels do not match the experimental labels')
        probabilities = saved['probabilities']
    if probabilities.shape != labels.shape:
        raise ValueError('Neural probability shape does not match test labels')
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('Neural probabilities must be finite and in [0, 1]')
    return probabilities > 0.5


def compare(data_path, run_directory, references=None, reference_format='b8', neural_path=None):
    data_path, run_directory = Path(data_path), Path(run_directory)
    config_path = run_directory / 'configuration.json'
    if config_path.exists():
        config = json.loads(config_path.read_text())
    else:
        config = json.loads((run_directory / 'history.json').read_text())['configuration']
    _, actual = read_records(data_path, config)
    with np.load(run_directory / 'shot_indices.npz', allow_pickle=False) as archive:
        indices = {name: archive[name] for name in PARTITION_NAMES}
    validate_shot_indices(indices, len(actual))
    test_ids = indices['test']
    test_labels = actual[test_ids]
    constant_class = int(np.mean(actual[training_ids(run_directory, indices)]) > 0.5)
    results = {
        'constant': {
            **prediction_metrics(test_labels, np.full(len(test_labels), constant_class)),
            'class_selected_on_training': constant_class,
        },
    }
    from core.GCN.reference import stored_predictions
    reference_values, _ = stored_predictions(references or {}, data_path, len(actual), reference_format, config.get('collection'))
    for name, predictions in reference_values.items():
        if name in ('constant', 'neural'):
            raise ValueError(f'Reserved predictor name: {name}')
        results[name] = prediction_metrics(test_labels, predictions[test_ids])
    if neural_path is None:
        candidate = run_directory / 'test_predictions.npz'
        if candidate.exists():
            neural_path = candidate
    if neural_path is not None:
        predictions = load_neural_predictions(neural_path, test_ids, test_labels)
        results['neural'] = prediction_metrics(test_labels, predictions)
    return {'error_rate_unit': 'per_complete_shot', 'results': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--reference', action='append', default=[], metavar='NAME=PATH',
                        help='Saved predictions in original shot order; repeat per decoder')
    parser.add_argument('--reference-format', choices=['b8', '01'], default='b8')
    parser.add_argument('--neural-predictions', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    references = {}
    for specification in args.reference:
        name, separator, path = specification.partition('=')
        if not separator or not name or not path or name in references:
            parser.error('--reference requires a unique NAME=PATH')
        references[name] = path
    report = compare(args.data_path, args.run_dir, references,
                     args.reference_format, args.neural_predictions)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == '__main__':
    main()

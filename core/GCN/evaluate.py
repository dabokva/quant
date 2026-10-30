import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from core.GCN.calibration import verify_gate, gate_metrics, gate_decisions
from core.GCN.metrics import classification_metrics, probability_metrics, paired_difference
from core.GCN.reference import parse_references, stored_predictions, MatchingDecoder, CollectionDecoder
from core.GCN.run_io import (load_run, read_records, read_indices, predict_records,
                             save_predictions, write_json, run_identity, select_device, file_sha256, training_ids)

def evaluate(run_directory, data_path, subset='test', references=None, reference_format='b8',
             gate_path=None, live_dem=None, batch_size=32, device='cpu', bootstrap=2000,
             seed=73, output=None):
    if subset not in ('validation', 'test'):
        raise ValueError('Evaluation subset must be validation or test')
    run_directory = Path(run_directory)
    with tf.device(select_device(device)):
        model, mapper, config = load_run(run_directory)
        events, labels = read_records(data_path, config)
        indices = read_indices(run_directory, len(labels))
        ids = indices[subset]
        probabilities = predict_records(model, mapper, events[ids], batch_size)
    actual = labels[ids]
    collection = config.get('collection')
    predictions, metadata = stored_predictions(references or {}, data_path, len(labels), reference_format, collection)
    if live_dem:
        decoder = (CollectionDecoder(live_dem, data_path, collection) if collection
                   else MatchingDecoder(live_dem, config['number_of_detectors']))
        predictions['pymatching'] = decoder.decode_batch(events[ids])
        metadata['pymatching'] = decoder.metadata
    predictions = {name: values if name == 'pymatching' and live_dem else values[ids]
                   for name, values in predictions.items()}
    used = training_ids(run_directory, indices)
    constant = int(labels[used].mean() > .5)
    results = {'neural': probability_metrics(actual, probabilities),
               'constant': {**classification_metrics(actual, np.full(len(ids), constant)),
                            'class_selected_on_training': constant}}
    comparisons = {}
    for name, values in predictions.items():
        results[name] = classification_metrics(actual, values)
        comparisons[name] = paired_difference(actual, probabilities > .5, values, bootstrap, seed)
    comparisons['constant'] = paired_difference(actual, probabilities > .5,
                                                np.full(len(ids), constant), bootstrap, seed)
    report = {'schema_version': 1, 'run_directory': str(run_directory.resolve()),
              'run_identity': run_identity(run_directory), 'subset': subset,
              'error_rate_unit': 'per_complete_shot', 'configuration': config,
              'training_positive_prevalence': float(labels[used].mean()),
              'training_shots_used': len(used),
              'results': results, 'paired_neural_minus_reference': comparisons,
              'references': metadata, 'gate_sha256': file_sha256(gate_path) if gate_path else None}
    if gate_path:
        gate = json.loads(Path(gate_path).read_text())
        verify_gate(gate, run_directory, indices)
        report['gate_configuration'] = gate
        report['gate'] = gate_metrics(actual, probabilities, gate)
        report['hybrids'] = {name: gate_metrics(actual, probabilities, gate, values)
                            for name, values in predictions.items()}
        # Diagnostic curve only. The operational threshold stays locked to calibration.
        q, _ = gate_decisions(probabilities, gate)
        curve = []
        for threshold in np.linspace(.5, 1, 101):
            accepted = np.maximum(q, 1-q) >= np.float32(threshold)
            curve.append({'threshold': float(threshold), 'coverage': float(accepted.mean()),
                          **classification_metrics(actual[accepted], q[accepted] > .5)})
        report['diagnostic_risk_coverage'] = curve
    output = Path(output) if output else run_directory / f'evaluation_{subset}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    if collection:
        from core.GCN.collection import provenance
        origins = provenance(collection, ids)
        report['per_sample'] = {}
        for sample in collection['samples']:
            selected = origins['sample_names'] == sample['name']
            values = {'neural': probability_metrics(actual[selected], probabilities[selected]),
                      'constant': classification_metrics(actual[selected], np.full(int(selected.sum()), constant)),
                      **{name: classification_metrics(actual[selected], prediction[selected]) for name, prediction in predictions.items()}}
            item = {'results': values}
            if gate_path:
                item['gate'] = gate_metrics(actual[selected], probabilities[selected], gate)
                item['hybrids'] = {name: gate_metrics(actual[selected], probabilities[selected], gate, prediction[selected])
                                   for name, prediction in predictions.items()}
            report['per_sample'][sample['name']] = item
    save_predictions(output.with_suffix('.predictions.npz'), ids, actual, probabilities, collection)
    if subset == 'test':
        save_predictions(run_directory / 'test_predictions.npz', ids, actual, probabilities, collection)
    if predictions:
        names = sorted(predictions)
        np.savez_compressed(output.with_suffix('.references.npz'), shot_indices=ids,
                            reference_names=np.asarray(names),
                            predictions=np.stack([predictions[name] for name in names]))
    report['prediction_file'] = output.with_suffix('.predictions.npz').name
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--subset', choices=['validation', 'test'], default='test')
    parser.add_argument('--reference', action='append', default=[], metavar='NAME=PATH')
    parser.add_argument('--reference-format', choices=['b8', '01'], default='b8')
    parser.add_argument('--gate', type=Path)
    parser.add_argument('--live-dem', type=Path, help='Optional decomposed DEM for live PyMatching MWPM')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', choices=['cpu', 'gpu', 'auto'], default='cpu')
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=73)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = evaluate(args.run_dir, args.data_path, args.subset, parse_references(args.reference),
                      args.reference_format, args.gate, args.live_dem, args.batch_size,
                      args.device, args.bootstrap, args.seed, args.output)
    print(json.dumps(report['results'], indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

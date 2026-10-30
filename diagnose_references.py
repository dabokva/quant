from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import stim

from core.GCN.metrics import classification_metrics, probability_metrics


SETTINGS = {
    'diagnostic_directory': '/work/diagnostics/generalization/sample_00_20261013_162810_675250',
    'output_root': 'diagnostics/references',
    'data_root': None,
    'format_overrides': {},
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def shot_ids(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in 'iu':
        raise ValueError(f'{name} must contain integer shot indices')
    if len(np.unique(values)) != len(values):
        raise ValueError(f'{name} contains duplicate shot indices')
    return values


def compare(diagnostic_directory, output_root, data_root=None, format_overrides=None):
    diagnostic = Path(diagnostic_directory).expanduser().resolve()
    saved_report = json.loads((diagnostic/'generalization_report.json').read_text())
    run = Path(saved_report['source_run']).expanduser().resolve()
    configuration = run/'configuration.json'
    config = (json.loads(configuration.read_text()) if configuration.exists()
              else json.loads((run/'history.json').read_text())['configuration'])
    samples = [s for s in config['collection']['samples'] if s['name'] == saved_report['sample']]
    if len(samples) != 1:
        raise ValueError('Diagnostic sample does not uniquely match the source collection')
    sample = samples[0]
    root = Path(data_root if data_root is not None else config['data_root']).expanduser().resolve()
    acquisition = root/sample['relative_path']
    predictions_directory = acquisition/'obs_flips_predicted'
    output_root = Path(output_root).expanduser().resolve()
    for protected in (run, diagnostic, root):
        if output_root == protected or protected in output_root.parents:
            raise ValueError('Output must be outside the source run, diagnostic, and dataset directories')

    print(f'Checking saved acquisition and validation indices for {sample["name"]}...', flush=True)
    for name in ('detection_events.b8', 'obs_flips_actual.b8'):
        if file_sha256(acquisition/name) != sample['sha256'][name]:
            raise ValueError(f'Acquisition file changed since training: {acquisition/name}')
    all_labels = stim.read_shot_data_file(path=acquisition/'obs_flips_actual.b8',
                                         format='b8', num_observables=1)[:, 0]
    if len(all_labels) != sample['shots']:
        raise ValueError('Actual-label count differs from the saved acquisition')
    with np.load(diagnostic/'diagnostic_shot_indices.npz', allow_pickle=False) as saved:
        ids = shot_ids(saved['validation'], 'Diagnostic validation')
        train_ids = shot_ids(saved['train'], 'Diagnostic training')
    if not len(ids) or np.intersect1d(ids, train_ids).size:
        raise ValueError('Validation must be nonempty and disjoint from diagnostic training')
    with np.load(run/'shot_indices.npz', allow_pickle=False) as saved:
        original = shot_ids(saved['validation'], 'Source validation')
        expected = original[(original >= sample['offset']) &
                            (original < sample['offset'] + sample['shots'])]
        if not np.array_equal(ids, expected):
            raise ValueError('Diagnostic validation differs from the original acquisition validation split')
        if not np.isin(train_ids, saved['train']).all():
            raise ValueError('Diagnostic training includes shots outside the source training split')
        for name in ('train', 'calibration', 'test'):
            if name in saved and np.intersect1d(ids, saved[name]).size:
                raise ValueError(f'Validation overlaps the source {name} split')
    local_ids = ids - sample['offset']
    labels = all_labels[local_ids]
    with np.load(diagnostic/'validation_predictions.npz', allow_pickle=False) as saved:
        if not np.array_equal(saved['shot_indices'], ids) or not np.array_equal(saved['labels'], labels):
            raise ValueError('Saved neural predictions do not align with original validation shots/labels')
        probabilities = saved['probabilities']
        epoch = int(saved['epoch'])
    if epoch != saved_report['history'][-1]['epoch']:
        raise ValueError('Neural predictions and diagnostic report describe different epochs')

    overrides = dict(format_overrides or {})
    if any(fmt not in ('b8', '01') for fmt in overrides.values()):
        raise ValueError('Format overrides currently support only b8 and 01')
    for name in overrides:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Override paths must be relative to obs_flips_predicted')
    files = sorted(p for p in predictions_directory.rglob('*') if p.is_file())
    report = {
        'source_run': str(run), 'diagnostic_directory': str(diagnostic),
        'sample': sample['name'], 'subset': 'validation', 'shots': len(ids),
        'prediction_directory': str(predictions_directory),
        'validation_indices_sha256': file_sha256(diagnostic/'diagnostic_shot_indices.npz'),
        'actual_labels_sha256': sample['sha256']['obs_flips_actual.b8'],
        'neural_epoch': epoch, 'neural': probability_metrics(labels, probabilities),
        'always_zero': classification_metrics(labels, np.zeros(len(labels), dtype=np.uint8)),
        'always_one': classification_metrics(labels, np.ones(len(labels), dtype=np.uint8)),
        'references': {}, 'rejected': {}, 'skipped': {},
        'note': 'Stored predictions only; no decoding latency measured. Reference rows are assumed '
                'to use original acquisition order. File counts alone do not prove alignment. '
                'This is an exploratory validation comparison, not a final test evaluation.',
    }
    print(f'{len(ids):,} validation shots; neural predictions from epoch {epoch}.', flush=True)
    print(f'Reference directory: {predictions_directory}', flush=True)

    def print_metrics(name, metrics):
        print(f'{name}: accuracy={metrics["accuracy"]:.4f} '
              f'errors={metrics["incorrect"]}/{metrics["shots"]} '
              f'error_rate={metrics["error_rate"]:.4f}', flush=True)

    for name in ('neural', 'always_zero', 'always_one'):
        print_metrics(name, report[name])
    selected = {}
    for path in files:
        name = path.relative_to(predictions_directory).as_posix()
        fmt = overrides.get(name, {'.b8': 'b8', '.01': '01'}.get(path.suffix.lower()))
        if fmt is None:
            report['skipped'][name] = 'Unknown format; specify b8 or 01 in format_overrides if applicable'
            print(f'SKIPPED {name}: unknown format', flush=True)
            continue
        try:
            values = stim.read_shot_data_file(path=path, format=fmt, num_observables=1)[:, 0]
            if len(values) != sample['shots']:
                raise ValueError(f'Expected {sample["shots"]} original acquisition shots, found {len(values)}')
            predictions = values[local_ids]
            metrics = classification_metrics(labels, predictions)
            neural_errors = (probabilities > .5) != labels
            reference_errors = predictions != labels
            report['references'][name] = {
                'path': str(path), 'format': fmt, 'sha256': file_sha256(path),
                'metrics': metrics,
                'neural_only_wrong': int(np.count_nonzero(neural_errors & ~reference_errors)),
                'reference_only_wrong': int(np.count_nonzero(reference_errors & ~neural_errors)),
            }
            selected[name] = predictions
            print_metrics(name, metrics)
        except (OSError, ValueError, RuntimeError) as error:
            report['rejected'][name] = str(error)
            print(f'REJECTED {name}: {error}', flush=True)
    found_names = {p.relative_to(predictions_directory).as_posix() for p in files}
    for name in overrides.keys() - found_names:
        report['rejected'][name] = 'Override file does not exist'
        print(f'REJECTED {name}: override file does not exist', flush=True)
    report['status'] = ('no_valid_references' if not selected else
                        'partial' if report['rejected'] or report['skipped'] else 'complete')
    if not files:
        print('No reference files found. Check the printed directory and data_root setting.', flush=True)
    output = output_root/(sample['name']+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    names = list(selected)
    np.savez_compressed(output/'validation_comparison.npz', shot_indices=ids, labels=labels,
                        neural_probabilities=probabilities, reference_names=np.asarray(names, dtype=str),
                        reference_predictions=(np.stack(list(selected.values())) if selected
                                               else np.empty((0, len(ids)), dtype=np.uint8)))
    report_path = output/'reference_report.json'
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'{report["status"]}: {report_path}', flush=True)
    return output, report


if __name__ == '__main__':
    options = dict(SETTINGS)
    for key in ('diagnostic_directory', 'output_root', 'data_root'):
        if options[key] is not None:
            path = Path(options[key]).expanduser()
            options[key] = path if path.is_absolute() else Path(__file__).resolve().parent/path
    compare(**options)

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from core.GCN.progress import track
from core.GCN.run_io import run_identity, write_json


def tex_escape(value):
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
                    '#': r'\#', '_': r'\_\allowbreak ', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
                    '^': r'\textasciicircum{}'}
    return ''.join(replacements.get(char, char) for char in str(value))


def number(value):
    return '--' if value is None else f'{value:.5g}' if isinstance(value, float) else str(value)


def seed_group(configuration):
    architecture = {key: configuration.get(key) for key in
                    ('hidden_units', 'number_of_gcn_layers', 'dense_units', 'dropout_rate')}
    architecture['dense2_units'] = configuration.get('dense2_units', configuration['dense_units'] // 2)
    architecture['precision'] = configuration.get('precision', 'float32')
    architecture['training_mode'] = configuration.get('training_mode', 'fixed_dataset')
    architecture['fine_tuning_mode'] = configuration.get('fine_tuning_mode')
    architecture['batch_size'] = configuration.get('batch_size')
    variant = configuration.get('experiment_label')
    if not variant:
        variant = (f"{architecture['training_mode']}_{configuration.get('kind', 'baseline')}_"
                   f"h{architecture['hidden_units']}_l{architecture['number_of_gcn_layers']}_"
                   f"d{architecture['dense_units']}_{architecture['dense2_units']}_"
                   f"{architecture['precision']}")
    return variant, architecture


def aggregate_measurements(measurements):
    from core.GCN.benchmark import latency_statistics
    groups = {}
    for item in measurements:
        key = (item['execution_device'], item['input_placement'], item['scope'])
        groups.setdefault(key, []).append(item)
    rows = []
    for items in groups.values():
        row = dict(items[0])
        if len(items) > 1:
            row['latency'] = latency_statistics(np.concatenate([item['latency']['samples_us'] for item in items]))
            row['sample'] = 'all'
            throughputs = [item['throughput'] for item in items if item.get('throughput')]
            if throughputs:
                seconds = sum(item['elapsed_seconds'] for item in throughputs)
                completed = sum(item['completed_shots'] for item in throughputs)
                row['throughput'] = {'shots_per_second': completed/seconds,
                                     'actual_batch_sizes': [n for item in throughputs for n in item['actual_batch_sizes']]}
            for container, memories in [(row, [item.get('memory', {}) for item in items]),
                                         (row.get('throughput', {}), [item.get('memory', {}) for item in throughputs])]:
                rss = [m['sampled_peak_rss_bytes'] for m in memories if m.get('sampled_peak_rss_bytes') is not None]
                peaks = [(m.get('tensorflow_allocator_after') or {}).get('peak') for m in memories]
                peaks = [v for v in peaks if v is not None]
                container['memory'] = {'sampled_peak_rss_bytes': max(rss) if rss else None,
                                       'tensorflow_allocator_after': {'peak': max(peaks)} if peaks else None}
        rows.append(row)
    return rows


def csv_file(path, rows):
    if not rows:
        path.write_text('')
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as output:
        writer = csv.DictWriter(output, fields)
        writer.writeheader()
        writer.writerows(rows)


def table(path, caption, label, headers, rows):
    columns = 'p{0.30\\linewidth}' + 'r' * (len(headers) - 1)
    lines = [r'\begingroup\small', r'\setlength{\tabcolsep}{4pt}',
             r'\begin{longtable}{' + columns + '}',
             r'\caption{' + caption + r'}\label{' + label + r'}\\', r'\hline',
             ' & '.join(headers) + r' \\', r'\hline\endfirsthead',
             ' & '.join(headers) + r' \\', r'\hline\endhead']
    lines += [' & '.join((r'\raggedright ' if index == 0 else '') + tex_escape(number(value)) for index, value in enumerate(row)) + r' \\' for row in rows]
    if not rows:
        lines += [r'\multicolumn{' + str(len(headers)) + r'}{c}{No measurements available.} \\']
    lines += [r'\hline', r'\end{longtable}', r'\endgroup', '']
    path.write_text('\n'.join(lines))


def export_reports(evaluation_paths, benchmark_paths, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    prediction_rows, gate_rows, latency_rows, resource_rows, paired_rows = [], [], [], [], []
    sources, summaries, figures, setup_rows = [], [], [], []
    graph_rows, per_sample_rows, per_sample_gates, detailed_latency = {}, [], [], []
    identity_by_run = {}
    membership_by_variant, seen_seeds = {}, set()
    for index, path in enumerate(track(evaluation_paths, 'Export evaluations', unit='run')):
        path = Path(path)
        report = json.loads(path.read_text())
        if report['subset'] != 'test':
            raise ValueError('Thesis final-result export accepts test reports only')
        run = Path(report['run_directory'])
        if run_identity(run) != report['run_identity']:
            raise ValueError(f'Stale evaluation report: {path}')
        config = report['configuration']
        name = config.get('experiment_label', run.name)
        seed = config['random_seed']
        identity_by_run[str(run)] = (report['run_identity'], report.get('gate_sha256'))
        variant, architecture = seed_group(config)
        if (variant, seed) in seen_seeds:
            raise ValueError('Seed aggregation requires unique seeds per variant; duplicate report or mislabeled variant')
        seen_seeds.add((variant, seed))
        with np.load(path.with_suffix('.predictions.npz'), allow_pickle=False) as saved:
            shot_ids = saved['shot_indices']
        membership = (config['dataset_sha256'], config.get('dem_sha256'), architecture,
                      report['run_identity']['graph_structure.npz'],
                      report['run_identity']['shot_indices.npz'], shot_ids.tolist())
        if variant in membership_by_variant and membership != membership_by_variant[variant]:
            raise ValueError('Seed aggregation requires identical datasets, graphs, architectures and partitions per variant')
        membership_by_variant[variant] = membership
        display = name if config.get('study_protocol') == 'single_seed_batch_comparison' else f'{name} (seed {seed})'
        setup_rows.append({'run': display, **{key: config.get(key) for key in
                          ('num_train', 'num_validation', 'num_calibration', 'num_test',
                           'number_of_detectors', 'number_of_errors', 'number_of_nodes', 'number_of_edges')}})
        setup_rows[-1].update({'training_mode': config.get('training_mode', 'fixed_dataset'),
                              'training_batch_size': config.get('batch_size'),
                              'fixed_training_shots': config.get('fixed_training_shots'),
                              'completed_epochs': config.get('completed_epochs'),
                              'sample_count': config.get('sample_count', 1),
                              'unique_training_shots_used': report.get('training_shots_used', config['num_train'])})
        if config.get('collection'):
            for sample in config['collection']['samples']:
                graph_rows[(sample['name'], sample['graph_sha256'])] = {'run': sample['name'], **sample}
        else:
            graph_rows[display] = {'run': display, **config}
        for sample, item in report.get('per_sample', {}).items():
            for predictor, values in item['results'].items():
                per_sample_rows.append({'run': display, 'sample': sample, 'predictor': predictor,
                                        **{key: values.get(key) for key in ('shots', 'incorrect', 'error_rate', 'accuracy', 'bce', 'roc_auc', 'brier', 'ece')},
                                        'ci_lower': values['error_rate_ci'][0], 'ci_upper': values['error_rate_ci'][1]})
            if item.get('gate'):
                gate = item['gate']
                per_sample_gates.append({'run': display, 'sample': sample, 'coverage': gate['coverage'],
                                         'accepted_shots': gate['accepted']['shots'], 'accepted_error': gate['accepted']['error_rate'],
                                         'accepted_upper': gate['accepted']['error_rate_upper'],
                                         **{f'{name}_hybrid_error': h['hybrid']['error_rate'] for name, h in item.get('hybrids', {}).items()}})
        for predictor, values in report['results'].items():
            prediction_rows.append({'run': display, 'predictor': predictor, 'seed': seed,
                                    'shots': values['shots'], 'incorrect': values['incorrect'],
                                    'error_rate': values['error_rate'], 'accuracy': values['accuracy'],
                                    'ci_lower': values['error_rate_ci'][0], 'ci_upper': values['error_rate_ci'][1],
                                    **{key: values.get(key) for key in ('bce', 'roc_auc', 'brier', 'ece')},
                                    'test_positive_prevalence': values['positive_prevalence'],
                                    'train_positive_prevalence': report['training_positive_prevalence']})
        for reference, values in report['paired_neural_minus_reference'].items():
            paired_rows.append({'run': display, 'reference': reference, **values})
        gate = report.get('gate')
        if gate:
            gate_rows.append({'run': display, 'fallback': 'none', 'coverage': gate['coverage'],
                              'accepted_shots': gate['accepted']['shots'],
                              'accepted_error': gate['accepted']['error_rate'],
                              'accepted_upper': gate['accepted']['error_rate_upper'],
                              'target_error': gate['target_error'], 'threshold': gate['threshold'],
                              'hybrid_error': None})
            for name_ref, hybrid in report.get('hybrids', {}).items():
                gate_rows.append({**gate_rows[-1], 'fallback': name_ref,
                                  'hybrid_error': hybrid['hybrid']['error_rate']})
        resource_rows.append({'run': display, 'precision': config.get('precision', 'float32'),
                              'parameters': config.get('parameter_count'),
                              'parameter_bytes': config.get('parameter_bytes'),
                              'resident_parameter_bytes': config.get('resident_parameter_bytes'),
                              'export_file_bytes': config.get('export_file_bytes'),
                              'hidden_units': config['hidden_units'], 'dense_units': config['dense_units'],
                              'dense2_units': config.get('dense2_units', config['dense_units']//2)})
        neural = report['results']['neural']
        summaries.append({'variant': variant,
                          'seed': seed, 'error_rate': neural['error_rate'], 'bce': neural['bce']})
        bins = neural['reliability']
        fig, axes = plt.subplots(1, 2 if gate else 1, figsize=(10 if gate else 5, 4), squeeze=False)
        axes[0, 0].plot([0, 1], [0, 1], '--', color='gray')
        nonempty = [row for row in bins if row['shots']]
        axes[0, 0].plot([r['mean_probability'] for r in nonempty],
                        [r['positive_fraction'] for r in nonempty], 'o-', label='Raw')
        if gate:
            bins = [r for r in gate['calibrated']['reliability'] if r['shots']]
            axes[0, 0].plot([r['mean_probability'] for r in bins], [r['positive_fraction'] for r in bins], 'o-', label='Calibrated')
            curve = report['diagnostic_risk_coverage']
            valid = [row for row in curve if row['error_rate'] is not None]
            axes[0, 1].plot([r['coverage'] for r in valid], [r['error_rate'] for r in valid])
            axes[0, 1].set(xlabel='Coverage', ylabel='Accepted-shot error', title='Diagnostic test curve')
        axes[0, 0].set(xlabel='Mean predicted flip probability', ylabel='Observed positive fraction', title='Reliability')
        axes[0, 0].legend()
        fig.suptitle(display)
        fig.tight_layout()
        fig.savefig(output / f'diagnostics_{index + 1}.pdf')
        plt.close(fig)
        figures.append(r'\begin{figure}[htbp]\centering' + '\n' +
                       r'\includegraphics[width=0.95\linewidth]{\EvaluationResultsPath/diagnostics_' + str(index + 1) + '.pdf}' + '\n' +
                       r'\caption{Probability diagnostics for ' + tex_escape(display) + '.}' + '\n' + r'\end{figure}')
        sources.append({'path': str(path.resolve()), 'run_identity': report['run_identity']})
    for path in track(benchmark_paths, 'Export benchmarks', unit='run'):
        report = json.loads(Path(path).read_text())
        run = Path(report['run_directory'])
        if run_identity(run, report.get('checkpoint', 'restored.weights.h5')) != report['run_identity']:
            raise ValueError(f'Stale benchmark: {path}')
        if str(run) in identity_by_run and identity_by_run[str(run)] != (report['run_identity'], report.get('gate_sha256')):
            raise ValueError('Prediction and benchmark reports use different weights or gates')
        from core.GCN.run_io import read_configuration
        config = read_configuration(run)
        name = config.get('experiment_label', run.name)
        display = name if config.get('study_protocol') == 'single_seed_batch_comparison' else f"{name} (seed {config['random_seed']})"
        for item in report['measurements']:
            detailed_latency.append({'run': display, 'sample': item.get('sample'),
                                      'device': item['execution_device'], 'placement': item['input_placement'],
                                      'scope': item['scope'], **{key: value for key, value in item['latency'].items() if key != 'samples_us'}})
        for item in aggregate_measurements(report['measurements']):
            values = item['latency']
            memory = item.get('memory', {})
            allocator = memory.get('tensorflow_allocator_after') or {}
            latency_rows.append({'run': display, 'device': item['execution_device'],
                                 'placement': item['input_placement'], 'scope': item['scope'],
                                 **{key: values[key] for key in ('mean', 'median', 'minimum', 'p95', 'p99')},
                                 'repetitions': values['repetitions'],
                                 'throughput_shots_per_second': item.get('throughput', {}).get('shots_per_second'),
                                 'throughput_max_actual_batch': max(item.get('throughput', {}).get('actual_batch_sizes', [0])) or None,
                                 'sampled_peak_rss_bytes': memory.get('sampled_peak_rss_bytes'),
                                 'tensorflow_allocator_peak_bytes': allocator.get('peak'),
                                 'throughput_sampled_peak_rss_bytes': item.get('throughput', {}).get('memory', {}).get('sampled_peak_rss_bytes'),
                                 'throughput_allocator_peak_bytes': (item.get('throughput', {}).get('memory', {}).get('tensorflow_allocator_after') or {}).get('peak')})
        sources.append({'path': str(Path(path).resolve()), 'run_identity': report['run_identity']})
    for filename, rows in [('configuration', setup_rows), ('predictions', prediction_rows), ('gates', gate_rows), ('latency', latency_rows),
                            ('resources', resource_rows), ('paired_comparisons', paired_rows),
                            ('per_sample_predictions', per_sample_rows), ('per_sample_gates', per_sample_gates),
                            ('per_sample_latency', detailed_latency), ('graphs', list(graph_rows.values()))]:
        csv_file(output / f'{filename}.csv', rows)
    table(output / 'configuration_table.tex', 'Actual saved partition sizes for the reported experiments.',
          'tab:evaluation_actual_partitions', ['Model', 'Train', 'Validation', 'Calibration', 'Test'],
          [[r['run'], r['num_train'], r['num_validation'], r['num_calibration'], r['num_test']] for r in setup_rows])
    table(output / 'graph_table.tex', 'Graph dimensions per acquisition (or model for single-directory runs); directed edges include self-loops.',
          'tab:evaluation_actual_graph', ['Acquisition / model', 'Detectors', 'Errors', 'Nodes', 'Edges'],
          [[r['run'], r['number_of_detectors'], r['number_of_errors'], r['number_of_nodes'], r['number_of_edges']] for r in graph_rows.values()])
    table(output / 'prediction_table.tex', 'Held-out prediction results. Errors are per complete shot.',
          'tab:evaluation_prediction_results', ['Model / predictor', 'Errors', r'$P_{\rm fail}$', 'BCE', 'AUC'],
          [[f"{r['run']} / {r['predictor']}", r['incorrect'], r['error_rate'], r['bce'], r['roc_auc']] for r in prediction_rows])
    table(output / 'gate_table.tex', 'Frozen gate results. The upper bound is one-sided 95 percent on accepted test shots.',
          'tab:evaluation_gate_results', ['Model / fallback', 'Coverage', 'Accepted error', 'Upper', 'Hybrid error'],
          [[f"{r['run']} / {r['fallback']}", r['coverage'], r['accepted_error'], r['accepted_upper'], r['hybrid_error']] for r in gate_rows])
    table(output / 'latency_table.tex', 'Batch-one latency in microseconds. Collection results pool raw timings across acquisitions; see per-sample CSV for variation.',
          'tab:evaluation_latency', ['Model / device / scope', 'Median', 'P95', 'P99'],
          [[f"{r['run']} / {r['device']} / {r['placement']} / {r['scope']}", r['median'], r['p95'], r['p99']] for r in latency_rows])
    table(output / 'throughput_memory_table.tex', 'Throughput batch size and rate; memory columns refer to batch-one execution. RSS is sampled whole-process memory; allocator peaks are backend-specific. Memory units are MiB.',
          'tab:evaluation_throughput_memory', ['Model / device / scope', 'Batch', 'Shots/s', 'RSS', 'Allocator'],
          [[f"{r['run']} / {r['device']} / {r['placement']} / {r['scope']}",
            r['throughput_max_actual_batch'], r['throughput_shots_per_second'],
            r['sampled_peak_rss_bytes']/2**20 if r['sampled_peak_rss_bytes'] is not None else None,
            r['tensorflow_allocator_peak_bytes']/2**20 if r['tensorflow_allocator_peak_bytes'] is not None else None]
           for r in latency_rows])
    table(output / 'resource_table.tex', 'Model size. Exported files exclude optimizer state; runtime memory is reported separately.',
          'tab:evaluation_resources', ['Model', 'Parameters', 'Parameter bytes', 'File bytes'],
          [[r['run'], r['parameters'], r['parameter_bytes'], r['export_file_bytes']] for r in resource_rows])
    (output / 'diagnostic_figures.tex').write_text('\n\n'.join(figures) + '\n')
    aggregates = []
    for variant in sorted({row['variant'] for row in summaries}):
        rows = [row for row in summaries if row['variant'] == variant]
        aggregates.append({'variant': variant, 'runs': len(rows),
                           **{f'{key}_mean': float(np.mean([r[key] for r in rows])) for key in ('error_rate', 'bce')},
                           **{f'{key}_seed_sd': float(np.std([r[key] for r in rows], ddof=1)) if len(rows) > 1 else None
                              for key in ('error_rate', 'bce')}})
    if summaries and all(row['runs'] == 1 for row in aggregates):
        single = [{'variant': row['variant'], 'error_rate': row['error_rate_mean'], 'bce': row['bce_mean']}
                  for row in aggregates]
        csv_file(output / 'variant_summary.csv', single)
        write_json(output / 'sources.json', {'sources': sources, 'variant_summary': single,
                    'note': 'One run per variant. No across-seed uncertainty estimate.'})
    else:
        csv_file(output / 'seed_summary.csv', aggregates)
        write_json(output / 'sources.json', {'sources': sources, 'seed_summary': aggregates,
                                             'note': 'Seed SD describes initialization variation, not independent test samples.'})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, action='append', default=[])
    parser.add_argument('--benchmark', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.evaluation:
        parser.error('At least one --evaluation report is required')
    print(f'Results exported to: {export_reports(args.evaluation, args.benchmark, args.output)}')


if __name__ == '__main__':
    main()

import argparse
import json
from pathlib import Path
import re

import numpy as np
import stim
import tensorflow as tf

from core.GCN.graph_class import Graph
from core.GCN.graph_data import FEATURE_NAMES, make_gcn_structure, make_node_feature_mapper, make_feature_mapper_from_static


RECORD_FILES = ('circuit_ideal.stim', 'detection_events.b8', 'obs_flips_actual.b8', 'dems/correlations.dem')


def discover(root, configuration='d5_at_q6_5/X/r25', expected_samples=None):
    root = Path(root).expanduser().resolve()
    relative = Path(configuration)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Configuration must be a relative path under each sample folder')
    samples = sorted((p for p in root.glob('sample_*') if p.is_dir() and re.fullmatch(r'sample_\d+', p.name)),
                     key=lambda p: int(p.name.split('_')[1]))
    if not samples or (expected_samples is not None and len(samples) != expected_samples):
        raise ValueError(f'Found {len(samples)} sample folders in {root}; expected {expected_samples or "at least one"}')
    for sample in samples:
        for name in RECORD_FILES:
            if not (sample/relative/name).is_file():
                raise ValueError(f'Missing required collection file: {sample/relative/name}')
    return [(p.name, p/relative) for p in samples]


def inspect_collection(root, configuration='d5_at_q6_5/X/r25', expected_samples=None):
    from core.GCN.run_io import file_sha256
    samples, offset = [], 0
    for name, path in discover(root, configuration, expected_samples):
        circuit = stim.Circuit.from_file(str(path/'circuit_ideal.stim'))
        if circuit.num_observables != 1 or circuit.num_detectors < 1:
            raise ValueError(f'{name}: expected detectors and exactly one logical observable')
        width = (circuit.num_detectors + 7)//8
        size = (path/'detection_events.b8').stat().st_size
        count, remainder = divmod(size, width)
        if remainder or count == 0 or (path/'obs_flips_actual.b8').stat().st_size != count:
            raise ValueError(f'{name}: incomplete b8 records or misaligned event/label shot counts')
        samples.append({'name': name, 'relative_path': str(path.relative_to(Path(root).expanduser().resolve())),
                        'offset': offset, 'shots': count, 'number_of_detectors': circuit.num_detectors,
                        'sha256': {file: file_sha256(path/file) for file in RECORD_FILES},
                        'graph_file': f'graphs/{name}.npz'})
        offset += count
    return {'schema_version': 1, 'configuration': configuration, 'sample_count': len(samples),
            'total_shots': offset, 'ordering': 'numeric sample suffix, then original b8 row', 'samples': samples}


def save_graphs(root, manifest, run):
    from core.GCN.run_io import file_sha256, write_json
    run = Path(run)
    (run/'graphs').mkdir(parents=True, exist_ok=True)
    for sample in manifest['samples']:
        path = Path(root)/sample['relative_path']
        graph = Graph(path/'circuit_ideal.stim', path/'dems/correlations.dem')
        edges, weights = make_gcn_structure(graph)
        mapper, _ = make_node_feature_mapper(graph)
        features, _ = mapper(np.zeros(graph.circuit.num_detectors), np.zeros(1))
        np.savez_compressed(run/sample['graph_file'], edge_index=edges, edge_weights=weights,
                            static_node_features=features.numpy()[:, 1:], feature_names=np.asarray(FEATURE_NAMES),
                            number_of_detectors=graph.circuit.num_detectors)
        sample.update({'number_of_nodes': len(graph.graph_nodes), 'number_of_errors': len(graph.error_nodes),
                       'number_of_edges': edges.shape[1], 'number_of_incidences': len(graph.edges),
                       'graph_sha256': file_sha256(run/sample['graph_file'])})
    import shutil
    shutil.copy2(run/manifest['samples'][0]['graph_file'], run/'graph_structure.npz')
    write_json(run/'collection.json', manifest)


class CollectionEvents:
    def __init__(self, root, manifest, ids=None, source=None):
        self.root, self.manifest = Path(root), manifest
        self.samples = manifest['samples']
        self.offsets = np.array([s['offset'] for s in self.samples], np.int64)
        self.ids = np.arange(manifest['total_shots'], dtype=np.int64) if ids is None else np.asarray(ids, np.int64)
        if source is None:
            self.packed = [np.memmap(self.root/s['relative_path']/'detection_events.b8', mode='r', dtype=np.uint8,
                                     shape=(s['shots'], (s['number_of_detectors']+7)//8)) for s in self.samples]
        else:
            self.packed = source.packed

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ids):
        return CollectionEvents(self.root, self.manifest, np.atleast_1d(self.ids[ids]), self)

    def locate(self, ids=None):
        ids = self.ids if ids is None else np.asarray(ids, np.int64)
        if np.any(ids < 0) or np.any(ids >= self.manifest['total_shots']):
            raise ValueError('Collection shot identifier out of range')
        sample_ids = np.searchsorted(self.offsets, ids, side='right')-1
        return sample_ids, ids-self.offsets[sample_ids]

    def read(self, sample_index, global_ids):
        sample = self.samples[sample_index]
        local = np.asarray(global_ids)-sample['offset']
        if np.any(local < 0) or np.any(local >= sample['shots']):
            raise ValueError('Shot identifiers do not belong to the selected sample')
        packed = self.packed[sample_index][local]
        return np.unpackbits(packed, axis=1, count=sample['number_of_detectors'], bitorder='little').astype(bool)

    def batches(self, batch_size=32, shuffle_seed=None):
        if batch_size < 1:
            raise ValueError('Batch size must be positive')
        sample_ids, _ = self.locate()
        groups = [np.flatnonzero(sample_ids == i) for i in range(len(self.samples))]
        rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
        if rng is not None:
            for group in groups:
                rng.shuffle(group)
        descriptors = [(i, start) for i, group in enumerate(groups) for start in range(0, len(group), batch_size)]
        if rng is not None:
            rng.shuffle(descriptors)
        for i, start in descriptors:
            positions = groups[i][start:start+batch_size]
            yield positions, i, self.read(i, self.ids[positions])


def read_collection(root, config):
    from core.GCN.run_io import file_sha256
    manifest = config['collection']
    labels = []
    for sample in manifest['samples']:
        path = Path(root)/sample['relative_path']
        for file, expected in sample['sha256'].items():
            if file_sha256(path/file) != expected:
                raise ValueError(f'Collection file changed: {sample["name"]}/{file}')
        values = stim.read_shot_data_file(path=path/'obs_flips_actual.b8', format='b8', num_observables=1)[:, 0]
        if len(values) != sample['shots']:
            raise ValueError('Collection label count changed')
        labels.append(values)
    return CollectionEvents(root, manifest), np.concatenate(labels)


class GraphBank:
    def __init__(self, run, manifest):
        self.manifest = manifest
        self.graphs = []
        for sample in manifest['samples']:
            with np.load(Path(run)/sample['graph_file'], allow_pickle=False) as saved:
                if saved['feature_names'].tolist() != FEATURE_NAMES:
                    raise ValueError('Collection graph uses a different feature layout')
                self.graphs.append({name: saved[name] for name in saved.files})
        self.mappers = [make_feature_mapper_from_static(g['static_node_features'], int(g['number_of_detectors']))
                        for g in self.graphs]

    def inputs(self, sample_index, events):
        features, _ = self.mappers[sample_index](events, np.zeros((len(events), 1), np.float32))
        graph = self.graphs[sample_index]
        return {'features': features, 'edge_index': tf.convert_to_tensor(graph['edge_index'], tf.int32),
                'edge_weights': tf.convert_to_tensor(graph['edge_weights'], tf.float32)}

    def predict(self, model, events, batch_size):
        probabilities = np.empty(len(events), np.float32)
        for positions, sample_index, batch in events.batches(batch_size):
            probabilities[positions] = model(self.inputs(sample_index, batch), training=False).numpy().reshape(-1)
        return probabilities

    def bind(self, model, index):
        graph = self.graphs[index]
        model.edge_index = tf.convert_to_tensor(graph['edge_index'], tf.int32)
        model.edge_weights = tf.convert_to_tensor(graph['edge_weights'], tf.float32)
        return self.mappers[index]


def split_collection(manifest, seed=73, calibration_fraction=.05):
    from core.GCN.data_loader import partition_indices, PARTITION_NAMES, validate_shot_indices
    parts = {name: [] for name in PARTITION_NAMES}
    for sample in manifest['samples']:
        local = partition_indices(sample['shots'], seed + int(sample['name'].split('_')[-1]), calibration_fraction)
        for name in parts:
            parts[name].append(local[name]+sample['offset'])
    indices = {name: np.concatenate(values) for name, values in parts.items()}
    validate_shot_indices(indices, manifest['total_shots'])
    return indices


def provenance(manifest, ids):
    offsets = np.array([s['offset'] for s in manifest['samples']], np.int64)
    sample_ids = np.searchsorted(offsets, ids, side='right')-1
    names = np.array([s['name'] for s in manifest['samples']])
    return {'sample_names': names[sample_ids], 'local_shot_indices': np.asarray(ids)-offsets[sample_ids]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--configuration', default='d5_at_q6_5/X/r25')
    parser.add_argument('--expected-samples', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    manifest = inspect_collection(args.data_root, args.configuration, args.expected_samples)
    if args.output:
        from core.GCN.run_io import write_json
        write_json(args.output, manifest)
    print(json.dumps({'samples': manifest['sample_count'], 'shots': manifest['total_shots'],
                      'configuration': manifest['configuration'],
                      'per_sample': {s['name']: s['shots'] for s in manifest['samples']}}, indent=2))


if __name__ == '__main__':
    main()

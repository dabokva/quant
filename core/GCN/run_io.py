import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from core.GCN.GCN import GCNPredecoder
from core.GCN.graph_data import FEATURE_NAMES, make_feature_mapper_from_static


DATA_FILES = ('circuit_ideal.stim', 'detection_events.b8', 'obs_flips_actual.b8')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dataset_fingerprints(data_path):
    return {name: file_sha256(Path(data_path) / name) for name in DATA_FILES}


def verify_dataset(data_path, configuration):
    if configuration.get('collection'):
        from core.GCN.collection import read_collection
        read_collection(data_path, configuration)
        return
    expected = configuration.get('dataset_sha256')
    if expected is None:
        raise ValueError('Run has no dataset fingerprints; retrain with the updated script')
    if dataset_fingerprints(data_path) != expected:
        raise ValueError('Dataset files differ from the collection used for this run')


def select_device(device):
    gpus = tf.config.list_physical_devices('GPU')
    if device == 'gpu' and not gpus:
        raise ValueError('GPU requested, but TensorFlow reports no GPU devices')
    if device == 'cpu' or not gpus:
        return '/CPU:0'
    return '/GPU:0'


def load_run(run_directory, checkpoint='restored.weights.h5', precision=None):
    run_directory = Path(run_directory)
    config = read_configuration(run_directory)
    with np.load(run_directory / 'graph_structure.npz', allow_pickle=False) as archive:
        if archive['feature_names'].tolist() != FEATURE_NAMES:
            raise ValueError('Run uses a different node-feature layout')
        static = archive['static_node_features']
        num_detectors = int(archive['number_of_detectors'])
        mapper = make_feature_mapper_from_static(static, num_detectors)
        if static.shape[0] != config['number_of_nodes']:
            raise ValueError('Graph archive and configuration disagree on node count')
        if num_detectors != config['number_of_detectors']:
            raise ValueError('Graph archive and configuration disagree on detector count')
        model = GCNPredecoder(
            edge_index=archive['edge_index'], edge_weights=archive['edge_weights'],
            hidden_units=config['hidden_units'],
            number_of_gcn_layers=config['number_of_gcn_layers'],
            dense_units=config['dense_units'], dropout_rate=config['dropout_rate'],
            dense2_units=config.get('dense2_units'), precision=precision or config.get('precision', 'float32'),
        )
    model(tf.zeros((1, config['number_of_nodes'], len(FEATURE_NAMES)), tf.float32))
    if checkpoint == 'restored.weights.h5' and config.get('weights_file'):
        checkpoint = config['weights_file']
    if str(checkpoint).endswith('.npz'):
        with np.load(run_directory / checkpoint, allow_pickle=False) as saved_weights:
            model.set_weights([saved_weights[f'weight_{i}'].astype('float32') for i in range(len(model.weights))])
    else:
        model.load_weights(str(run_directory / checkpoint))
    if config.get('collection'):
        from core.GCN.collection import GraphBank
        if json.loads((run_directory/'collection.json').read_text()) != config['collection']:
            raise ValueError('Configuration and collection manifest disagree')
        mapper = GraphBank(run_directory, config['collection'])
    return model, mapper, config


def read_configuration(run_directory):
    run_directory = Path(run_directory)
    path = run_directory / 'configuration.json'
    if path.exists():
        return json.loads(path.read_text())
    return json.loads((run_directory / 'history.json').read_text())['configuration']


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, allow_nan=False, default=str)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text + '\n')
    temporary.replace(path)


def run_identity(run_directory, checkpoint='restored.weights.h5'):
    run_directory = Path(run_directory)
    config = read_configuration(run_directory)
    weights_file = config.get('weights_file', 'restored.weights.h5') if checkpoint == 'restored.weights.h5' else checkpoint
    names = ['graph_structure.npz', 'shot_indices.npz', weights_file]
    if config.get('collection'):
        names += ['collection.json', 'training_schedule.npz', 'training_used.npz']
        names += [sample['graph_file'] for sample in config['collection']['samples']]
    return {name: file_sha256(run_directory/name) for name in names} | {'precision': config.get('precision', 'float32')}


def read_records(data_path, config):
    import stim
    if config.get('collection'):
        from core.GCN.collection import read_collection
        return read_collection(data_path, config)
    verify_dataset(data_path, config)
    events = stim.read_shot_data_file(path=Path(data_path) / 'detection_events.b8', format='b8',
                                      num_detectors=config['number_of_detectors'])
    labels = stim.read_shot_data_file(path=Path(data_path) / 'obs_flips_actual.b8', format='b8',
                                      num_observables=1)[:, 0]
    if len(events) != len(labels):
        raise ValueError('Events and labels are misaligned')
    return events, labels


def read_indices(run_directory, count):
    from core.GCN.data_loader import PARTITION_NAMES, validate_shot_indices
    with np.load(Path(run_directory) / 'shot_indices.npz', allow_pickle=False) as saved:
        indices = {name: saved[name] for name in PARTITION_NAMES}
    validate_shot_indices(indices, count)
    return indices


def training_ids(run_directory, indices):
    path = Path(run_directory)/'training_used.npz'
    if not path.exists():
        return indices['train']
    with np.load(path, allow_pickle=False) as saved:
        used = saved['shot_indices']
    if not len(used) or len(np.unique(used)) != len(used) or not np.isin(used, indices['train']).all():
        raise ValueError('Recorded training exposure must be a nonempty subset of reserved training shots')
    return used


def predict_records(model, mapper, events, batch_size=32):
    if batch_size < 1:
        raise ValueError('Batch size must be positive')
    if hasattr(mapper, 'predict'):
        return mapper.predict(model, events, batch_size)
    output = []
    for start in range(0, len(events), batch_size):
        batch = events[start:start + batch_size]
        features, _ = mapper(batch, np.zeros((len(batch), 1), np.float32))
        output.append(model(features, training=False).numpy().reshape(-1))
    return np.concatenate(output) if output else np.empty(0, np.float32)


def save_predictions(path, ids, labels, probabilities, collection=None):
    extra = {}
    if collection:
        from core.GCN.collection import provenance
        extra = provenance(collection, ids)
    np.savez_compressed(path, shot_indices=np.asarray(ids, np.int64),
                        labels=np.asarray(labels, np.uint8), probabilities=np.asarray(probabilities), **extra)

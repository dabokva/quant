from pathlib import Path
import numpy as np
import stim

from core.GCN.run_io import file_sha256


def parse_references(specifications):
    result = {}
    for specification in specifications:
        name, separator, path = specification.partition('=')
        if not separator or not name or not path or name in result or name in ('constant', 'neural', 'pymatching'):
            raise ValueError('References require unique, nonreserved NAME=PATH values')
        result[name] = path
    return result


def stored_predictions(references, data_path, count, format='b8', collection=None):
    predictions, metadata = {}, {}
    for name, path in references.items():
        if collection:
            chunks, records = [], []
            for sample in collection['samples']:
                resolved = collection_path(path, data_path, sample)
                values = stim.read_shot_data_file(path=resolved, format=format, num_observables=1)[:, 0]
                if len(values) != sample['shots']:
                    raise ValueError(f'{name}/{sample["name"]}: prediction count differs from original shots')
                chunks.append(values)
                records.append({'sample': sample['name'], 'path': str(resolved), 'sha256': file_sha256(resolved)})
            predictions[name] = np.concatenate(chunks)
            metadata[name] = {'samples': records, 'format': format, 'execution': 'stored_predictions_only'}
            continue
        path = Path(path)
        if not path.is_absolute():
            path = Path(data_path) / path
        values = stim.read_shot_data_file(path=path, format=format, num_observables=1)[:, 0]
        if len(values) != count:
            raise ValueError(f'{name} requires one prediction per original shot, in original order')
        predictions[name] = values
        metadata[name] = {'path': str(path.resolve()), 'sha256': file_sha256(path),
                          'format': format, 'execution': 'stored_predictions_only'}
    return predictions, metadata


def collection_path(specification, root, sample):
    text = str(specification)
    path = Path(text.format(sample=sample['name']))
    if path.is_absolute():
        if '{sample}' not in text:
            raise ValueError('Absolute collection reference/DEM paths must contain {sample}')
        return path
    return Path(root)/sample['relative_path']/path


class CollectionDecoder:
    def __init__(self, specification, root, manifest):
        self.decoders = [MatchingDecoder(collection_path(specification, root, s), s['number_of_detectors'])
                         for s in manifest['samples']]
        self.metadata = {'execution': 'live_uncorrelated_MWPM_per_sample',
                         'samples': {s['name']: d.metadata for s, d in zip(manifest['samples'], self.decoders)}}

    def decode_batch(self, events):
        result = np.empty(len(events), np.uint8)
        for positions, sample_index, batch in events.batches(4096):
            result[positions] = self.decoders[sample_index].decode_batch(batch)
        return result


class MatchingDecoder:
    def __init__(self, dem_path, num_detectors):
        import pymatching
        dem = stim.DetectorErrorModel.from_file(str(dem_path)).flattened()
        if dem.num_detectors != num_detectors or dem.num_observables != 1:
            raise ValueError('Live decoder DEM dimensions do not match the experimental circuit')
        for instruction in dem:
            if instruction.type != 'error':
                continue
            component = set()
            for target in [*instruction.targets_copy(), stim.DemTarget.separator()]:
                if target.is_separator():
                    if len(component) > 2:
                        raise ValueError('Live MWPM needs graphlike DEM components; supply a decomposed DEM')
                    component = set()
                elif target.is_relative_detector_id():
                    component.symmetric_difference_update({target.val})
        self.matching = pymatching.Matching.from_detector_error_model(dem)
        self.matching.ensure_num_fault_ids(1)
        self.metadata = {'name': 'pymatching', 'version': pymatching.__version__,
                         'dem_path': str(Path(dem_path).resolve()), 'dem_sha256': file_sha256(dem_path),
                         'execution': 'live_uncorrelated_MWPM'}

    def decode_one(self, events):
        return int(self.matching.decode(np.asarray(events, np.uint8))[0])

    def decode_batch(self, events):
        if len(events) == 0:
            return np.empty(0, np.uint8)
        return self.matching.decode_batch(np.asarray(events, np.uint8))[:, 0]

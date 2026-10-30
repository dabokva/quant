from pathlib import Path
import stim

class Graph:
    def __init__(self, circuit_path, dem_path):
        self.circuit_path = Path(circuit_path)
        self.dem_path = Path(dem_path)

        self.circuit = None
        self.dem = None

        self.detector_coordinates = {}
        self.centered_detector_coordinates = {}
        self.final_qubit_coordinates = {}
        self.node_weights = {}

        self.detector_nodes = {}
        self.error_nodes = {}
        self.graph_nodes = {}

        self.slices = {}
        self.edges = {}

        self.load_circuit()
        self.load_dem()
        self.extract_coordinates()
        self.compute_weights()
        self.compute_coordinate_centroids()
        self.build_nodes()
        self.populate_slices()
        self.find_parity_measurements()
        self.build_error_nodes()
        self.compute_edges()

    def load_circuit(self):
        self.circuit = stim.Circuit.from_file(str(self.circuit_path))

    def load_dem(self):
        self.dem = stim.DetectorErrorModel.from_file(
            str(self.dem_path)
        ).flattened()

        if self.dem.num_detectors != self.circuit.num_detectors:
            raise ValueError(
                f'Circuit has {self.circuit.num_detectors} detectors, '
                f'but DEM has {self.dem.num_detectors}'
            )

        if self.dem.num_observables != self.circuit.num_observables:
            raise ValueError(
                'Circuit and DEM have different observable counts'
            )

    def extract_coordinates(self):
        self.detector_coordinates = self.circuit.get_detector_coordinates()
        self.final_qubit_coordinates = (
            self.circuit.get_final_qubit_coordinates()
        )

    def compute_weights(self):
        for detector_id, record in self.detector_coordinates.items():
            if not record or len(record) % 3 != 0:
                raise ValueError(
                    f'Detector {detector_id} has an unexpected coordinate record'
                )

            self.node_weights[detector_id] = max(1, len(record) // 3 - 1)

    def compute_coordinate_centroids(self):
        for detector_id, record in self.detector_coordinates.items():
            current_record = record if len(record) == 3 else record[:-3]

            points = [
                current_record[start:start + 3]
                for start in range(0, len(current_record), 3)
            ]

            times = {point[2] for point in points}
            if len(times) != 1:
                raise ValueError(
                    f'Detector {detector_id} has multiple current times'
                )

            x = sum(point[0] for point in points) / len(points)
            y = sum(point[1] for point in points) / len(points)
            t = points[0][2]

            self.centered_detector_coordinates[detector_id] = (x, y, t)

    def build_nodes(self):
        times = [
            coordinates[2]
            for coordinates in self.centered_detector_coordinates.values()
        ]

        if not times:
            raise ValueError('The circuit contains no detector coordinates')

        first_time = min(times)
        last_time = max(times)

        for detector_id, coordinates in (
            self.centered_detector_coordinates.items()
        ):
            node = Node(detector_id, coordinates)
            node.weight = self.node_weights[detector_id]
            node.coordinate_record = tuple(self.detector_coordinates[detector_id])

            t = coordinates[2]
            node.init_boundary = t == first_time
            node.final_boundary = t == last_time
            node.bulk = not (node.init_boundary or node.final_boundary)

            self.detector_nodes[detector_id] = node
            self.graph_nodes[detector_id] = node

    def populate_slices(self):
        for detector_id, node in self.detector_nodes.items():
            t = node.coordinates[2]
            self.slices.setdefault(t, []).append(
                (detector_id, node.coordinates)
            )

    def find_parity_measurements(self):
        for detector_id, node in self.detector_nodes.items():
            record = self.detector_coordinates[detector_id]

            if node.weight > 1:
                current_record = record[:-3]

                node.parity_measurements = [
                    tuple(current_record[start:start + 3])
                    for start in range(0, len(current_record), 3)
                ]

            if len(record) > 3:
                node.previous_measurement_coordinates = tuple(record[-3:])

    def build_error_nodes(self):
        first_error_node_id = self.circuit.num_detectors

        for instruction in self.dem:
            if instruction.type != 'error':
                continue

            probability = float(instruction.args_copy()[0])

            detector_ids = set()
            observable_ids = set()
            components = [[]]

            for target in instruction.targets_copy():
                if target.is_separator():
                    components.append([])
                    continue

                components[-1].append(str(target))

                if target.is_relative_detector_id():
                    detector_ids.symmetric_difference_update(
                        {int(target.val)}
                    )

                elif target.is_logical_observable_id():
                    observable_ids.symmetric_difference_update(
                        {int(target.val)}
                    )

            mechanism_id = len(self.error_nodes)
            node_id = first_error_node_id + mechanism_id

            node = ErrorNode(
                node_id=node_id,
                mechanism_id=mechanism_id,
                probability=probability,
                detector_ids=sorted(detector_ids),
                observable_ids=sorted(observable_ids),
                components=[
                    tuple(component)
                    for component in components
                    if component
                ],
            )

            self.error_nodes[node_id] = node
            self.graph_nodes[node_id] = node

    def compute_edges(self):
        for error_id, error_node in self.error_nodes.items():
            for detector_id in error_node.detector_ids:
                if detector_id not in self.detector_nodes:
                    raise ValueError(
                        f'DEM references unknown detector D{detector_id}'
                    )

                detector_node = self.detector_nodes[detector_id]

                self.edges[(detector_id, error_id)] = {
                    'detector_id': detector_id,
                    'error_id': error_id,
                }

                detector_node.neighbors.append(error_id)
                error_node.neighbors.append(detector_id)

    def make_edge_index(self):
        sources = []
        targets = []

        for detector_id, error_id in self.edges:
            sources.extend([detector_id, error_id])
            targets.extend([error_id, detector_id])

        return [sources, targets]


class Node:
    def __init__(self, node_id, coordinates):
        self.node_id = node_id
        self.node_type = 'detector'
        self.coordinates = tuple(coordinates)
        self.coordinate_record = ()

        self.weight = None

        self.init_boundary = False
        self.bulk = False
        self.final_boundary = False

        self.parity_measurements = []
        self.previous_measurement_coordinates = None

        self.neighbors = []


class ErrorNode:
    def __init__(
        self,
        node_id,
        mechanism_id,
        probability,
        detector_ids,
        observable_ids,
        components,
    ):
        self.node_id = node_id
        self.node_type = 'error'
        self.mechanism_id = mechanism_id

        self.probability = probability
        self.detector_ids = tuple(detector_ids)
        self.observable_ids = tuple(observable_ids)
        self.components = tuple(components)

        self.is_boundary = len(detector_ids) == 1
        self.is_undetectable = len(detector_ids) == 0

        self.neighbors = []

import numpy as np
import tensorflow as tf


FEATURE_NAMES = [
    'detection_event', 'is_detector', 'is_error', 'x', 'y', 't',
    'initial_slice', 'final_slice', 'bulk', 'error_probability',
    'matching_boundary', 'undetectable', 'flips_L0',
]


def validate_graph(graph):
    number_of_nodes = len(graph.graph_nodes)
    number_of_detectors = graph.circuit.num_detectors
    if set(graph.graph_nodes) != set(range(number_of_nodes)):
        raise ValueError('Graph node IDs must be contiguous and start at zero')
    if set(graph.detector_nodes) != set(range(number_of_detectors)):
        raise ValueError('Detector node IDs must match the circuit detector IDs')
    if set(graph.error_nodes) != set(range(number_of_detectors, number_of_nodes)):
        raise ValueError('Error node IDs must follow all detector node IDs')
    if graph.circuit.num_observables != 1:
        raise ValueError('This binary predecoder expects exactly one observable')


def make_gcn_structure(graph):
    validate_graph(graph)
    number_of_nodes = len(graph.graph_nodes)
    pairs = set()

    for detector_id, error_id in graph.edges:
        if detector_id not in graph.detector_nodes or error_id not in graph.error_nodes:
            raise ValueError('Each incidence must connect a detector to an error node')
        pairs.add((detector_id, error_id))
        pairs.add((error_id, detector_id))

    pairs.update((node_id, node_id) for node_id in range(number_of_nodes))
    edge_index = np.asarray(sorted(pairs), dtype=np.int32).T
    sources, targets = edge_index
    degree = np.bincount(targets, minlength=number_of_nodes).astype(np.float32)

    edge_weights = 1.0 / np.sqrt(degree[sources] * degree[targets])
    return edge_index, edge_weights.astype(np.float32)


def make_node_feature_mapper(graph):
    validate_graph(graph)
    number_of_nodes = len(graph.graph_nodes)
    number_of_detectors = graph.circuit.num_detectors
    number_of_features = len(FEATURE_NAMES)

    static = np.zeros((number_of_nodes, number_of_features - 1), dtype=np.float32)
    coordinates = np.asarray([
        graph.detector_nodes[index].coordinates
        for index in range(number_of_detectors)
    ], dtype=np.float32)
    minimum = coordinates.min(axis=0)
    span = np.maximum(coordinates.max(axis=0) - minimum, 1.0)
    coordinates = (coordinates - minimum) / span

    for detector_id, node in graph.detector_nodes.items():
        static[detector_id, 0] = 1.0  # is_detector
        static[detector_id, 2:5] = coordinates[detector_id]
        static[detector_id, 5:8] = [
            node.init_boundary, node.final_boundary, node.bulk,
        ]

    for error_id, node in graph.error_nodes.items():
        static[error_id, 1] = 1.0  # is_error
        static[error_id, 8:12] = [
            node.probability, node.is_boundary, node.is_undetectable,
            0 in node.observable_ids,
        ]

    return make_feature_mapper_from_static(static, number_of_detectors), number_of_features


def make_feature_mapper_from_static(static, number_of_detectors):
    static = np.asarray(static, dtype=np.float32)
    if static.ndim != 2 or static.shape[1] != len(FEATURE_NAMES) - 1:
        raise ValueError('Static features must have shape (N, 12)')
    number_of_nodes = static.shape[0]
    if not 0 < number_of_detectors <= number_of_nodes:
        raise ValueError('Invalid detector count for static features')
    number_of_errors = number_of_nodes - number_of_detectors
    number_of_features = len(FEATURE_NAMES)
    static_features = tf.constant(static, dtype=tf.float32)

    def map_to_graph(events, labels):
        events = tf.cast(events, tf.float32)
        rank = events.shape.rank
        message = 'Detector-event width does not match the circuit'
        if rank is not None and rank not in (1, 2):
            raise ValueError('Detector events must have rank 1 or 2')
        if rank is not None and events.shape[-1] is not None:
            if events.shape[-1] != number_of_detectors:
                raise tf.errors.InvalidArgumentError(None, None, message)
        else:
            with tf.device('/CPU:0'):
                rank_check = tf.debugging.assert_rank_in(events, (1, 2))
                with tf.control_dependencies([rank_check]):
                    assertion = tf.debugging.assert_equal(
                        tf.shape(events)[-1], number_of_detectors, message=message)
            with tf.control_dependencies([assertion]):
                events = tf.identity(events)
        leading_shape = tf.shape(events)[:-1]

        unknown_error_inputs = tf.zeros(
            tf.concat([leading_shape, [number_of_errors, 1]], axis=0),
            dtype=tf.float32,
        )
        event_features = tf.concat([events[..., None], unknown_error_inputs], axis=-2)
        repeated_static = tf.broadcast_to(
            static_features,
            tf.concat([leading_shape, [number_of_nodes, number_of_features - 1]], axis=0),
        )
        features = tf.concat([event_features, repeated_static], axis=-1)
        if rank is not None:
            features.set_shape(events.shape[:-1].concatenate([number_of_nodes, number_of_features]))
        return features, tf.cast(labels, tf.float32)

    return map_to_graph

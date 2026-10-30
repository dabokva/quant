import stim
import networkx as nx
from core.GCN.graph_class import Graph
from pathlib import Path
import plotly.graph_objects as go

circuit_path = './data/sycamore_surface_code_d3_d5/sample_00/d5_at_q6_5/X/r25/circuit_ideal.stim'
simple_dem_path = './data/sycamore_surface_code_d3_d5/sample_00/d5_at_q6_5/X/r25/dems/simple.dem'
simple_dem = stim.DetectorErrorModel.from_file(str(simple_dem_path))
correlations_dem_path = './data/sycamore_surface_code_d3_d5/sample_00/d5_at_q6_5/X/r25/dems/correlations.dem'
correlations_dem = stim.DetectorErrorModel.from_file(str(correlations_dem_path))

def to_networkx(graph):
    nx_graph = nx.MultiDiGraph()

    coordinate_to_detector = {
        tuple(float(value) for value in node.coordinates): detector_id
        for detector_id, node in graph.detector_nodes.items()
    }

    for detector_id, node in graph.detector_nodes.items():
        x, y, time = node.coordinates
        record = graph.detector_coordinates[detector_id]

        if node.init_boundary:
            boundary = 'initial'
        elif node.final_boundary:
            boundary = 'final'
        else:
            boundary = 'bulk'

        previous_coordinate = None

        if node.previous_measurement_coordinates is not None:
            previous_coordinate = node.previous_measurement_coordinates

        nx_graph.add_node(
            detector_id,
            x=float(x),
            y=float(y),
            t=float(time),
            boundary=boundary,
            weight=int(node.weight),
            parity_measurements=[
                tuple(float(value) for value in coordinate)
                for coordinate in node.parity_measurements
            ],
            previous_coordinate=previous_coordinate,
        )

    for detector_id, attributes in nx_graph.nodes(data=True):
        previous_coordinate = attributes['previous_coordinate']

        if previous_coordinate is None:
            continue

        previous_detector = coordinate_to_detector.get(
            previous_coordinate
        )

        if previous_detector is not None:
            nx_graph.add_edge(
                previous_detector,
                detector_id,
                edge_type='temporal',
            )

    return nx_graph

def visualize_graph_3d(nx_graph, output_path=None):
    temporal_x = []
    temporal_y = []
    temporal_t = []

    for source, target, attributes in nx_graph.edges(data=True):
        if attributes.get('edge_type') != 'temporal':
            continue

        source_node = nx_graph.nodes[source]
        target_node = nx_graph.nodes[target]

        temporal_x.extend([source_node['x'], target_node['x'], None])
        temporal_y.extend([source_node['y'], target_node['y'], None])
        temporal_t.extend([source_node['t'], target_node['t'], None])

    temporal_trace = go.Scatter3d(
        x=temporal_x,
        y=temporal_y,
        z=temporal_t,
        mode='lines',
        line=dict(color='#6A6A6A', width=2),
        hoverinfo='none',
        name='Temporal detector connections',
    )

    boundary_data = {
        'initial': {
            'x': [],
            'y': [],
            't': [],
            'text': [],
            'color': '#A2AD00',
            'name': 'Initial-boundary detectors',
        },
        'bulk': {
            'x': [],
            'y': [],
            't': [],
            'text': [],
            'color': '#3070B3',
            'name': 'Bulk detectors',
        },
        'final': {
            'x': [],
            'y': [],
            't': [],
            'text': [],
            'color': '#C4071B',
            'name': 'Final-boundary detectors',
        },
    }

    multi_x = []
    multi_y = []
    multi_t = []
    multi_text = []

    membership_x = []
    membership_y = []
    membership_t = []

    measurement_owners = {}

    for detector_id, attributes in nx_graph.nodes(data=True):
        x = attributes['x']
        y = attributes['y']
        time = attributes['t']
        weight = attributes['weight']
        boundary = attributes['boundary']
        measurements = attributes.get('parity_measurements', [])

        hover_text = (
            f'Detector: {detector_id}<br>'
            f'Position: ({x:.1f}, {y:.1f}, {time:.1f})<br>'
            f'Type: {boundary}<br>'
            f'Weight: {weight}<br>'
            f'Measurements in parity: {len(measurements)}'
        )

        group = boundary_data[boundary]
        group['x'].append(x)
        group['y'].append(y)
        group['t'].append(time)
        group['text'].append(hover_text)

        if weight > 1:
            multi_x.append(x)
            multi_y.append(y)
            multi_t.append(time)
            multi_text.append(hover_text)

            for measurement in measurements:
                measurement_owners.setdefault(
                    measurement,
                    [],
                ).append(detector_id)

                measurement_x, measurement_y, measurement_t = measurement

                membership_x.extend([x, measurement_x, None])
                membership_y.extend([y, measurement_y, None])
                membership_t.extend([time, measurement_t, None])

    membership_trace = go.Scatter3d(
        x=membership_x,
        y=membership_y,
        z=membership_t,
        mode='lines',
        line=dict(
            color='#E37222',
            width=2,
            dash='dot',
        ),
        hoverinfo='none',
        name='Measurements included in detector parity',
    )

    multi_detector_trace = go.Scatter3d(
        x=multi_x,
        y=multi_y,
        z=multi_t,
        mode='markers',
        marker=dict(
            size=10,
            color='#E37222',
            symbol='diamond',
        ),
        text=multi_text,
        hoverinfo='text',
        name='Detectors combining multiple measurements',
    )

    detector_traces = []

    for group in boundary_data.values():
        detector_traces.append(
            go.Scatter3d(
                x=group['x'],
                y=group['y'],
                z=group['t'],
                mode='markers',
                marker=dict(
                    size=6,
                    color=group['color'],
                    symbol='circle',
                    line=dict(color='black', width=0.4),
                ),
                text=group['text'],
                hoverinfo='text',
                name=group['name'],
            )
        )

    measurement_x = []
    measurement_y = []
    measurement_t = []
    measurement_text = []

    for coordinate, detector_ids in measurement_owners.items():
        x, y, time = coordinate

        measurement_x.append(x)
        measurement_y.append(y)
        measurement_t.append(time)
        measurement_text.append(
            f'Measurement location: ({x:.1f}, {y:.1f}, {time:.1f})<br>'
            f'Used by detectors: {detector_ids}'
        )

    measurement_trace = go.Scatter3d(
        x=measurement_x,
        y=measurement_y,
        z=measurement_t,
        mode='markers',
        marker=dict(
            size=4,
            color='#F9BA00',
            symbol='circle',
            line=dict(color='black', width=0.4),
        ),
        text=measurement_text,
        hoverinfo='text',
        name='Measurements used in multi-measurement parity',
    )

    figure = go.Figure(
        data=[
            temporal_trace,
            membership_trace,
            multi_detector_trace,
            *detector_traces,
            measurement_trace,
        ]
    )

    figure.update_layout(
        title='QEC detector spacetime graph',
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='time',
            aspectmode='data',
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0.01, y=0.99),
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.write_html(str(output_path))
        print(f'Graph saved to: {output_path.resolve()}')

    figure.show()

data_path = Path(
    './data/sycamore_surface_code_d3_d5/'
    'sample_00/d5_at_q6_5/X/r25'
)

graph = Graph(
    circuit_path=data_path / 'circuit_ideal.stim',
    dem_path=data_path / 'dems/correlations.dem',
)

print(f'Detector nodes: {len(graph.detector_nodes)}')
print(f'Error nodes: {len(graph.error_nodes)}')
print(f'Total nodes: {len(graph.graph_nodes)}')
print(f'Incidence edges: {len(graph.edges)}')

for error_node in list(graph.error_nodes.values())[:5]:
    print(
        f'E{error_node.mechanism_id}: '
        f'p={error_node.probability:.6g}, '
        f'detectors={error_node.detector_ids}, '
        f'observables={error_node.observable_ids}'
    )

edge_index = graph.make_edge_index()
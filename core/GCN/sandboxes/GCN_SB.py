from pathlib import Path

from core.GCN.graph_class import Graph

data_path = Path('./data/sycamore_surface_code_d3_d5/sample_00/d5_at_q6_5/X/r25')

graph = Graph(data_path / 'circuit_ideal.stim', data_path / 'dems/correlations.dem')

for node_id, node in graph.detector_nodes.items():
    print(
        f'Node id: {node_id}\n'
        f'\tOriginal detector record: {graph.detector_coordinates[node_id]}\n'
        f'\tWeight: {node.weight}\n'
        f'\tCentroid coordinates: {node.coordinates}\n'
        f'\tInitial boundary: {node.init_boundary}\n'
        f'\tBulk: {node.bulk}\n'
        f'\tFinal boundary: {node.final_boundary}\n'
        f'\tParity locations: {node.parity_measurements}\n'
        f'\tPrevious location: {node.previous_measurement_coordinates}\n'
        f'\tAdjacent error nodes: {node.neighbors}'
    )
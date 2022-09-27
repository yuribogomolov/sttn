from sttn.network import SpatioTemporalNetwork
from sttn.utils import get_edges_with_centroids
from keplergl import KeplerGl
from typing import Optional, List

from .keplergl.map import MapConfig

DEFAULT_MAP_HEIGHT = 600


def choropleth(data: SpatioTemporalNetwork, node_layers: Optional[List[str]] = None, include_edges: bool = False,
               edge_size: Optional[str] = None) -> KeplerGl:
    map_conf = MapConfig(data_id='nodes')
    nodes = data.nodes

    node_layers = _get_node_layers(node_layers, list(nodes.columns))
    for column in node_layers:
        map_conf.add_geo_layer(label=f"node: {column}", color_column=column)
    map_data = {'nodes': nodes.copy()}

    if include_edges:
        edges = get_edges_with_centroids(data)
        map_conf.with_data_id('edges').add_arc_layer(label=f"edge: {edge_size}", origin_lat="lat_from",
                                                     origin_lng="long_from",
                                                     destination_lat="lat_to", destination_lng="long_to",
                                                     size_column=edge_size)
        map_data['edges'] = edges

    output_map = KeplerGl(height=DEFAULT_MAP_HEIGHT, data=map_data, config=map_conf.to_dict())
    return output_map


def _get_node_layers(node_layers: Optional[List[str]], node_columns: List[str]) -> List[str]:
    if node_layers is None:
        node_columns.remove('geometry')
        return node_columns
    else:
        for layer in node_layers:
            if layer not in node_columns:
                raise ValueError(f"{layer} is not one of the data node data layers: {node_layers}")

        return node_layers

from sttn.network import SpatioTemporalNetwork
from keplergl import KeplerGl
from typing import Optional, List

from .keplergl.map import MapConfig

DEFAULT_MAP_HEIGHT = 600


def choropleth(data: SpatioTemporalNetwork, node_layers: Optional[List[str]] = None) -> KeplerGl:
    map_conf = MapConfig(data_id='nodes')
    nodes = data.nodes

    node_layers = _get_node_layers(node_layers, list(nodes.columns))
    print(node_layers)
    for column in node_layers:
        map_conf.add_geo_layer(label=f"node: {column}", color_column=column)

    output_map = KeplerGl(height=DEFAULT_MAP_HEIGHT, data={'nodes': nodes.copy()}, config=map_conf.to_dict())
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

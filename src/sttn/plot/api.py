from typing import Optional, List, Tuple

import pandas as pd
from keplergl import KeplerGl

from sttn.network import SpatioTemporalNetwork
from sttn.utils import get_edges_with_centroids
from .keplergl.map import MapConfig

DEFAULT_MAP_HEIGHT = 600


def choropleth(data: SpatioTemporalNetwork, node_layers: Optional[List[str]] = None, include_edges: bool = False,
               edge_size_column: Optional[str] = None) -> KeplerGl:
    map_conf = MapConfig(data_id='nodes')
    nodes = data.nodes

    node_layers = _get_node_layers(node_layers, list(nodes.columns))
    for column in node_layers:
        map_conf.add_geo_layer(label=f"node: {column}", color_column=column)
    map_data = {'nodes': nodes.copy()}

    if include_edges:
        edges = get_edges_with_centroids(data)
        map_conf.with_data_id('edges').add_arc_layer(label=f"edge: {edge_size_column}", origin_lat="lat_from",
                                                     origin_lng="long_from",
                                                     destination_lat="lat_to", destination_lng="long_to",
                                                     size_column=edge_size_column)
        map_data['edges'] = edges

    output_map = KeplerGl(height=DEFAULT_MAP_HEIGHT, data=map_data, config=map_conf.to_dict())
    return output_map


def transaction_heatmap(data: SpatioTemporalNetwork, weight_column: str, transaction_node: str = "origin",
                        time_motion_column: Optional[str] = None) -> KeplerGl:
    map_conf = MapConfig(data_id='edges')
    edges = get_edges_with_centroids(data)

    if transaction_node == "origin":
        lat = "lat_from"
        lng = "long_from"
    elif transaction_node == "destination":
        lat = "lat_to"
        lng = "long_to"
    else:
        raise ValueError(f"{transaction_node} is not one of supported values [origin, destination]")

    map_conf.add_heatmap_layer(label=f"Edge {transaction_node} {weight_column}", lat_column=lat, lng_column=lng,
                               weight_column=weight_column)
    if time_motion_column is not None:
        window = _get_time_range(data.edges[time_motion_column])
        map_conf.add_time_range_filter(time_column=time_motion_column, window=window, y_axis_column=weight_column)

    map_data = {'edges': edges}
    output_map = KeplerGl(height=DEFAULT_MAP_HEIGHT, data=map_data, config=map_conf.to_dict())
    return output_map


def transaction(data: SpatioTemporalNetwork, edge_size_column: str,
                time_motion_column: Optional[str] = None) -> KeplerGl:
    map_conf = MapConfig(data_id='edges')
    edges = get_edges_with_centroids(data)
    map_conf.add_arc_layer(label=f"edge: {edge_size_column}", origin_lat="lat_from",
                           origin_lng="long_from",
                           destination_lat="lat_to", destination_lng="long_to",
                           size_column=edge_size_column)

    if time_motion_column is not None:
        window = _get_time_range(data.edges[time_motion_column])
        map_conf.add_time_range_filter(time_column=time_motion_column, window=window, y_axis_column=edge_size_column)

    map_data = {'edges': edges}
    output_map = KeplerGl(height=DEFAULT_MAP_HEIGHT, data=map_data, config=map_conf.to_dict())
    return output_map


def _get_time_range(time_column: pd.Series) -> Tuple[int, int]:
    window_start = time_column.min()
    window_end = time_column.max()
    window = (window_start.value // 1000000, window_end.value // 1000000)
    return window


def _get_node_layers(node_layers: Optional[List[str]], node_columns: List[str]) -> List[str]:
    if node_layers is None:
        node_columns.remove('geometry')
        return node_columns
    else:
        for layer in node_layers:
            if layer not in node_columns:
                raise ValueError(f"{layer} is not one of the data node data layers: {node_layers}")

        return node_layers

import pytest
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point
from sttn.network import SpatioTemporalNetwork

edges = {'origin': [1, 2], 'destination': [3, 4]}
edges_pd = pd.DataFrame(data=edges)

nodes = {'id': [1, 2], 'geometry': [Point(1, 2), Point(2, 1)]}
nodes_gpd = gpd.GeoDataFrame(nodes, crs="EPSG:4326")
nodes_indexed = nodes_gpd.set_index('id')


def test_data_type_errors():
    # invalid nodes type
    with pytest.raises(TypeError):
        SpatioTemporalNetwork(nodes=edges_pd, edges=edges_pd)
    # invalid edges type
    with pytest.raises(TypeError):
        SpatioTemporalNetwork(nodes=nodes_gpd, edges=edges)

    # inconsistent origin/destination types
    origin_casted = edges_pd.astype({'origin': 'int32'})
    with pytest.raises(TypeError):
        SpatioTemporalNetwork(nodes=nodes_indexed, edges=origin_casted)

    # inconsistent index types
    od_casted = edges_pd.astype({'origin': 'int32', 'destination': 'int32'})
    with pytest.raises(TypeError):
        SpatioTemporalNetwork(nodes=nodes_indexed, edges=od_casted)


def test_column_name_errors():
    with pytest.raises(KeyError):
        SpatioTemporalNetwork(nodes=nodes_gpd, edges=edges_pd, origin='origin_2')
    with pytest.raises(KeyError):
        SpatioTemporalNetwork(nodes=nodes_gpd, edges=edges_pd, destination='destination_2')


def test_index_check_errors():
    # no node index
    with pytest.raises(ValueError):
        SpatioTemporalNetwork(nodes=nodes_gpd, edges=edges_pd)


def test_index_values():
    # destination ids are not in the node index
    with pytest.raises(KeyError):
        SpatioTemporalNetwork(nodes=nodes_indexed, edges=edges_pd)

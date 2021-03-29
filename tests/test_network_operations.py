import pytest
import pandas as pd
import geopandas as gpd

from pandas.testing import assert_frame_equal

from shapely.geometry import Point
from sttn.network import SpatioTemporalNetwork

edges = {'origin': [1, 1, 2, 1, 2], 'destination': [2, 2, 1, 2, 2], 'value': [1, 2, 4, 8, 16], 'key': [1, 2, 1, 1, 1]}
edges_pd = pd.DataFrame(data=edges)

nodes = {'id': [1, 2, 3], 'geometry': [Point(1, 2), Point(2, 1), Point(3, 1)]}
nodes_gpd = gpd.GeoDataFrame(nodes, crs="EPSG:4326").set_index('id')
stn = SpatioTemporalNetwork(nodes=nodes_gpd, edges=edges_pd)


def test_agg_parallel_edges_no_key():
    aggregated = stn.agg_parallel_edges(column_aggs={'value': 'sum'})
    assert aggregated.edges.shape[0] == 3
    assert set(aggregated.edges.columns) == {'origin', 'destination', 'value'}

    expected = pd.DataFrame(data={'origin': [1, 2, 2], 'destination': [2, 1, 2], 'value': [11, 4, 16]})
    assert_frame_equal(aggregated.edges, expected)


def test_agg_parallel_edges_with_key():
    aggregated = stn.agg_parallel_edges(column_aggs={'value': 'sum'}, key='key')
    assert aggregated.edges.shape[0] == 4
    assert set(aggregated.edges.columns) == {'origin', 'destination', 'value', 'key'}

    expected = pd.DataFrame(data={'origin': [1, 1, 2, 2], 'destination': [2, 2, 1, 2], 'key': [1, 2, 1, 1], 'value': [9, 2, 4, 16]})
    assert_frame_equal(aggregated.edges, expected)

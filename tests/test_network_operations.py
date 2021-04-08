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
    assert_frame_equal(aggregated.nodes, nodes_gpd)


def test_agg_parallel_edges_with_key():
    aggregated = stn.agg_parallel_edges(column_aggs={'value': 'sum'}, key='key')
    assert aggregated.edges.shape[0] == 4
    assert set(aggregated.edges.columns) == {'origin', 'destination', 'value', 'key'}

    expected = pd.DataFrame(data={'origin': [1, 1, 2, 2], 'destination': [2, 2, 1, 2], 'key': [1, 2, 1, 1], 'value': [9, 2, 4, 16]})
    assert_frame_equal(aggregated.edges, expected)


def test_agg_adjacent_edges():
    # outgoing = True, include_cycles = True
    edges_aggregated = stn.agg_adjacent_edges(aggs={'value': 'sum'})
    expected_outgoing = pd.DataFrame(data={'origin': [1, 2], 'value': [11, 20]}).set_index('origin')
    assert_frame_equal(edges_aggregated, expected_outgoing)

    # outgoing = False, include_cycles = True
    incoming_aggregated = stn.agg_adjacent_edges(aggs={'value': 'sum'}, outgoing=False)
    expected_incoming = pd.DataFrame(data={'destination': [1, 2], 'value': [4, 27]}).set_index('destination')
    assert_frame_equal(incoming_aggregated, expected_incoming)

    # outgoing = True, include_cycles = False
    aggregated_no_cycle = stn.agg_adjacent_edges(aggs={'value': 'sum'}, include_cycles=False)
    expected_no_cycle = pd.DataFrame(data={'origin': [1, 2], 'value': [11, 4]}).set_index('origin')
    assert_frame_equal(aggregated_no_cycle, expected_no_cycle)

import pandas as pd
from haversine import haversine_vector, Unit
from networkx.algorithms import community

from sttn.network import SpatioTemporalNetwork


def get_edges_with_centroids(network: SpatioTemporalNetwork) -> pd.DataFrame:
    import warnings
    warnings.filterwarnings('ignore', message=".*geographic CRS. Results from 'centroid' are likely incorrect.*")

    centroid = network.nodes.centroid
    centroid_long = centroid.x
    centroid_long.name = 'long'
    centroid_lat = centroid.y
    centroid_lat.name = 'lat'
    centroids = pd.concat([centroid_long, centroid_lat], axis=1)
    centroid_from = network.edges.join(centroids, on=network._origin).rename(
        columns={'long': 'long_from', 'lat': 'lat_from'})
    centroid_all = centroid_from.join(centroids, on=network._destination).rename(
        columns={'long': 'long_to', 'lat': 'lat_to'})
    return centroid_all


def add_distance(network: SpatioTemporalNetwork) -> SpatioTemporalNetwork:
    """Add distance in km between area centroids."""
    centroid_all = get_edges_with_centroids(network)
    from_points = list(zip(centroid_all.lat_from, centroid_all.long_from))
    to_points = list(zip(centroid_all.lat_to, centroid_all.long_to))
    centroid_all['distance'] = haversine_vector(from_points, to_points, Unit.KILOMETERS)
    centroid_all.drop(['long_from', 'lat_from', 'long_to', 'lat_to'], axis=1, inplace=True)
    return SpatioTemporalNetwork(nodes=network.nodes, edges=centroid_all)


def detect_communities(network: SpatioTemporalNetwork, algo, **kwargs):
    if algo == 'fluid':
        comm_iter = community.asyn_fluidc(network.to_multigraph().to_undirected(), **kwargs)
        return list(comm_iter)
    if algo == 'clm':
        comm_iter = community.greedy_modularity_communities(network.to_multigraph().to_undirected(), **kwargs)
        return list(comm_iter)

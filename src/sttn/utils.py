import pandas as pd
from sttn.network import SpatioTemporalNetwork

from haversine import haversine_vector, Unit


def add_distance(network: SpatioTemporalNetwork) -> SpatioTemporalNetwork:
    """Add distance in km between area centroids."""
    centroid = network.node_labels.centroid
    centroid_long = centroid.x
    centroid_long.name = 'long'
    centroid_lat = centroid.y
    centroid_lat.name = 'lat'
    centroids = pd.concat([centroid_long, centroid_lat], axis=1)
    centroid_from = network.edges_df.join(centroids, on='from').rename(columns={'long': 'long_from', 'lat': 'lat_from'})
    centroid_all = centroid_from.join(centroids, on='to').rename(columns={'long': 'long_to', 'lat': 'lat_to'})

    from_points = list(zip(centroid_all.lat_from, centroid_all.long_from))
    to_points = list(zip(centroid_all.lat_to, centroid_all.long_to))
    centroid_all['distance'] = haversine_vector(from_points, to_points, Unit.KILOMETERS)
    centroid_all.drop(['long_from', 'lat_from', 'long_to', 'lat_to'], axis=1, inplace=True)
    return SpatioTemporalNetwork(centroid_all, network.node_labels)

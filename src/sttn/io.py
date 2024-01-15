import geopandas as gpd
import pandas as pd

from sttn.network import SpatioTemporalNetwork


def read_parquet(path: str) -> SpatioTemporalNetwork:
    """Read STTN nodes and edges to the Parquet format.
    """
    node_path = f"{path}-nodes.parquet"
    edge_path = f"{path}-edges.parquet"
    nodes = gpd.read_parquet(node_path)
    edges = pd.read_parquet(edge_path)
    return SpatioTemporalNetwork(nodes=nodes, edges=edges)

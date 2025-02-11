import geopandas as gpd
import networkx as nx
import pandas as pd
import skmob

from sttn import constants


class SpatioTemporalNetwork:
    def __init__(self, nodes: gpd.GeoDataFrame, edges: pd.DataFrame, origin: str = constants.ORIGIN,
                 destination: str = constants.DESTINATION, node_id: str = constants.NODE_ID):
        if not isinstance(nodes, gpd.GeoDataFrame):
            raise TypeError('Incompatible nodes data type: {e}'.format(e=type(edges)))

        if not isinstance(edges, pd.DataFrame):
            raise TypeError('Incompatible edges data type: {e}'.format(e=type(edges)))

        if origin not in edges:
            raise KeyError('Origin column name: {orig} is not found in the list: {columns}'
                           .format(orig=origin, columns=list(edges.columns)))

        if destination not in edges:
            raise KeyError('Destination column name: {dest} is not found in the list: {columns}'
                           .format(dest=destination, columns=list(edges.columns)))

        if not nodes.index.name == node_id:
            raise ValueError('Nodes dataframe must be indexed on {id}'.format(id=node_id))

        if edges[origin].dtype != edges[destination].dtype:
            raise TypeError('Origin dtype {o} does not match destination dtype {d}'
                            .format(o=edges[origin].dtype, d=edges[destination].dtype))

        if edges[origin].dtype != nodes.index.dtype:
            raise TypeError('Origin dtype {o} does not match node index dtype {d}'
                            .format(o=edges[origin].dtype, d=nodes.index.dtype))

        SpatioTemporalNetwork._validate_ids(edges[origin], nodes.index)
        SpatioTemporalNetwork._validate_ids(edges[destination], nodes.index)

        self._nodes = nodes
        self._edges = edges
        self._origin = origin
        self._destination = destination
        self._node_id = node_id

    @staticmethod
    def _validate_ids(edge_ids: pd.Series, node_index: pd.Index):
        not_in_index = edge_ids[~edge_ids.isin(node_index)]
        if not_in_index.size > 0:
            samples = not_in_index.unique()[:5]
            raise KeyError('Edge ids {ids} are not in the node index'.format(ids=samples))

    @property
    def nodes(self) -> gpd.GeoDataFrame:
        return self._nodes

    @property
    def edges(self) -> pd.DataFrame:
        return self._edges

    def agg_parallel_edges(self, column_aggs: dict, key: str = None):
        grouping = [self._origin, self._destination]
        if key:
            grouping.append(key)
        new_edges = self._edges.groupby(by=grouping, as_index=False).agg(column_aggs)
        return SpatioTemporalNetwork(nodes=self._nodes, edges=new_edges, origin=self._origin,
                                     destination=self._destination, node_id=self._node_id)

    def to_multigraph(self):
        return nx.from_pandas_edgelist(self._edges, source=self._origin, target=self._destination,
                                       edge_attr=True, create_using=nx.MultiDiGraph)

    def to_flow_date_frame(self, flow: str) -> skmob.FlowDataFrame:
        return skmob.FlowDataFrame(self._edges, origin=self._origin, destination=self._destination, flow=flow,
                                   tile_id=self._node_id, tessellation=self._nodes.reset_index())

    def shape(self) -> (int, int):
        return self._nodes.shape[0], self._edges.shape[0]

    def group_nodes(self, node_label):
        nodes = self._nodes

        if isinstance(node_label, list):
            node_to_label = [(item, ind) for ind, sublist in enumerate(node_label) for item in sublist]
            community_df = pd.DataFrame(node_to_label, columns=[self._node_id, 'community'])
            community_df = community_df.set_index(self._node_id)
            nodes = self._nodes.join(community_df)
            node_label = 'community'

        dissolved = nodes.dissolve(by=node_label, as_index=False).rename(columns={node_label: self._node_id})
        dissolved = dissolved.set_index(self._node_id)

        node_mapping = nodes[[node_label]]
        mapped_from = self._edges.join(node_mapping, on=self._origin) \
            .drop(self._origin, axis=1) \
            .rename(columns={node_label: self._origin})
        mapped_to = mapped_from.join(node_mapping, on=self._destination) \
            .drop(self._destination, axis=1) \
            .rename(columns={node_label: self._destination})
        return SpatioTemporalNetwork(nodes=dissolved, edges=mapped_to, origin=self._origin,
                                     destination=self._destination, node_id=self._node_id)

    def agg_adjacent_edges(self, aggs: dict, outgoing: bool = True, include_cycles: bool = True) -> pd.DataFrame:
        grouping_column = self._origin if outgoing else self._destination
        edges = self._edges if include_cycles else self._edges[
            self._edges[self._origin] != self._edges[self._destination]]
        grouped = edges.groupby(grouping_column).agg(aggs)
        return grouped.rename(columns={grouping_column: self._node_id})

    def join_node_labels(self, extra_columns):
        new_nodes = self._nodes.join(extra_columns)
        return SpatioTemporalNetwork(nodes=new_nodes, edges=self._edges, origin=self._origin,
                                     destination=self._destination, node_id=self._node_id)

    def filter_nodes(self, condition: pd.Series):
        if self._nodes.shape[0] != condition.shape[0]:
            msg = 'Number of nodes {nodes} is different from the length of the condition array {condition}'.format(
                nodes=self._nodes.shape[0], condition=condition.shape[0])
            raise ValueError(msg)

        ids_to_keep = self._nodes[condition].index
        filtered_edges = self._edges[
            self._edges[self._origin].isin(ids_to_keep) & self._edges[self._destination].isin(ids_to_keep)]
        return SpatioTemporalNetwork(nodes=self._nodes[condition], edges=filtered_edges, origin=self._origin,
                                     destination=self._destination, node_id=self._node_id)

    def filter_edges(self, condition: pd.Series):
        if self._edges.shape[0] != condition.shape[0]:
            msg = 'Number of edges {edges} is different from the length of the condition array {condition}'.format(
                edges=self._edges.shape[0], condition=condition.shape[0])
            raise ValueError(msg)

        filtered_edges = self._edges[condition]
        return SpatioTemporalNetwork(nodes=self._nodes, edges=filtered_edges, origin=self._origin,
                                     destination=self._destination, node_id=self._node_id)

    def to_parquet(self, path: str) -> None:
        """Write a STTN to the Parquet format.
        """
        # ignore GeoPandas Parquet warnings
        import warnings
        warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')

        node_path = f"{path}-nodes.parquet"
        edge_path = f"{path}-edges.parquet"
        self._nodes.to_parquet(node_path)
        self._edges.to_parquet(edge_path)

import networkx as nx
import pandas as pd
import skmob
from networkx.algorithms import community

class SpatioTemporalNetwork:
    def __init__(self, edges_df, node_labels):
        self.edges_df = edges_df
        self.node_labels = node_labels
        self.directed = True

    def agg_parallel_edges(self, column_aggs, key=None):
        grouping = ['from', 'to']
        if key:
            grouping.append(key)
        new_edges = self.edges_df.groupby(by=grouping, as_index=False).agg(column_aggs)
        return SpatioTemporalNetwork(new_edges, self.node_labels)

    def to_multigraph(self):
        return nx.from_pandas_edgelist(self.edges_df, source='from', target='to', edge_attr=True, create_using=nx.MultiDiGraph)

    def to_flow_date_frame(self, flow: str) -> skmob.FlowDataFrame:
        return skmob.FlowDataFrame(self.edges_df, origin='from', destination='to', flow=flow, tile_id='id',
                                   tessellation=self.node_labels.reset_index())

    def shape(self):
        G = self.to_multigraph()
        return (G.number_of_nodes(), G.number_of_edges())

    def group_nodes(self, node_label):
        node_labels = self.node_labels

        if isinstance(node_label, list):
            node_to_label = [(item, ind) for ind, sublist in enumerate(node_label) for item in sublist]
            community_df = pd.DataFrame(node_to_label, columns=['id', 'community'])
            community_df = community_df.set_index('id')
            node_labels = self.node_labels.join(community_df)
            node_label = 'community'

        dissolved = node_labels.dissolve(by=node_label, as_index=False).rename(columns={node_label: 'id'})
        dissolved = dissolved.set_index('id')

        node_mapping = node_labels[[node_label]]
        mapped_from = self.edges_df.join(node_mapping, on='from').drop('from', axis=1).rename(columns={node_label: 'from'})
        mapped_to = mapped_from.join(node_mapping, on='to').drop('to', axis=1).rename(columns={node_label: 'to'})
        return SpatioTemporalNetwork(mapped_to, dissolved)

    def agg_adjacent_edges(self, aggs, outgoing: bool = True, include_cycles: bool = True):
        grouping_column = 'from' if outgoing else 'to'
        edges = self.edges_df if include_cycles else self.edges_df[self.edges_df['from'] != self.edges_df['to']]
        grouped = edges.groupby(grouping_column).agg(aggs)
        return grouped.rename(columns={grouping_column: 'id'})

    def join_node_labels(self, extra_columns):
        new_labels = self.node_labels.join(extra_columns)
        return SpatioTemporalNetwork(self.edges_df, new_labels)

    def filter_nodes(self, condition: pd.Series):
        if self.node_labels.shape[0] != condition.count():
            msg = 'Number of nodes {nodes} is different from the length of the condition array {condition}'.format(
                nodes=self.node_labels.shape[0], condition=condition.count())
            raise ValueError(msg)

        ids_to_keep = self.node_labels[condition].index
        filtered_edges = self.edges_df[self.edges_df['from'].isin(ids_to_keep) & self.edges_df.to.isin(ids_to_keep)]
        return SpatioTemporalNetwork(filtered_edges, self.node_labels[condition])

    def detect_communities(self, algo, **kwargs):
        if algo == 'fluid':
            comm_iter = community.asyn_fluidc(self.to_multigraph().to_undirected(), **kwargs)
            return list(comm_iter)
        if algo == 'clm':
            comm_iter = community.greedy_modularity_communities(self.to_multigraph().to_undirected(), **kwargs)
            return list(comm_iter)

    def plot(self, **kwargs):
        return self.node_labels.plot(**kwargs)

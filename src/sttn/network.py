import networkx as nx

class SpatioTemporalNetwork:
    def __init__(self, edges_df, node_labels = {}):
        self.edges_df = edges_df
        self.node_labels = node_labels
        self.directed = True

    def agg_parallel_edges(self, column_aggs):
        new_edges = self.edges_df.groupby(by=['from', 'to'], as_index=False).agg(column_aggs)
        return SpatioTemporalNetwork(new_edges, self.node_labels)

    def to_multigraph(self):
        G = nx.MultiDiGraph()
        G.add_nodes_from(self.edges_df['from'].tolist())
        G.add_nodes_from(self.edges_df['to'].tolist())
        G.add_edges_from(list(zip(self.edges_df['from'], self.edges_df['to'])))
        return G

    def shape(self):
        G = self.to_multigraph()
        return (G.number_of_nodes(), G.number_of_edges())

    def group_nodes(self, node_label):
        dissolved = self.node_labels.dissolve(by=node_label, as_index=False).rename(columns={node_label: 'id'})
        dissolved = dissolved.set_index('id')

        node_mapping = self.node_labels[[node_label]]
        mapped_from = self.edges_df.join(node_mapping, on='from').drop('from', axis=1).rename(columns={node_label: 'from'})
        mapped_to = mapped_from.join(node_mapping, on='to').drop('to', axis=1).rename(columns={node_label: 'to'})
        return SpatioTemporalNetwork(mapped_to, dissolved)

    def plot(self, figsize=None):
        return self.node_labels.plot(figsize=figsize)

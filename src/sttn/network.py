import networkx as nx

class SpatioTemporalNetwork:
    def __init__(self, edges_df, column_from, column_to, time_column, node_labels = {}):
        self.edges_df = edges_df
        self.column_from = column_from
        self.column_to = column_to
        self.time_column = time_column
        self.node_labels = node_labels
        self.directed = True

    def agg_parallel_edges(self, column_aggs):
        new_edges = self.edges_df.groupby(by=[self.column_from, self.column_to], as_index=False).agg(column_aggs)
        return SpatioTemporalNetwork(new_edges, self.column_from, self.column_to, self.time_column, self.node_labels)

    def to_multigraph(self):
        G = nx.MultiDiGraph()
        G.add_nodes_from(self.edges_df[self.column_from].tolist())
        G.add_nodes_from(self.edges_df[self.column_to].tolist())
        G.add_edges_from(list(zip(self.edges_df[self.column_from], self.edges_df[self.column_to])))
        return G

    def shape(self):
        G = self.to_multigraph()
        return (G.number_of_nodes(), G.number_of_edges())

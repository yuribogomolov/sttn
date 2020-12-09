
class SpatioTemporalNetwork:
    def __init__(self, edges_df, column_from, column_to, time_column, node_labels = {}):
        self.edges_df = edges_df
        self.column_from = column_from
        self.column_to = column_to
        self.node_labels = node_labels
        self.directed = True

    def agg_parallel_edges(self, column_aggs):
        new_edges = edges_df.groupby(column_from, column_to).agg(column_aggs)
        return SpatioTemporalNetwork(new_edges, column_from, column_to, time_column, node_labels)

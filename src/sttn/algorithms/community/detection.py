from typing import Optional

import pandas as pd
import pycombo

from sttn.network import SpatioTemporalNetwork


def combo_communities(data: SpatioTemporalNetwork, weight: Optional[str] = None, **kwargs) -> SpatioTemporalNetwork:
    graph = data.to_multigraph()
    partition_dict, _ = pycombo.execute(graph, weight=weight, **kwargs)
    partition_df = pd.DataFrame.from_dict(partition_dict, orient='index', columns=['cluster'])
    data_with_community = data.join_node_labels(partition_df)
    return data_with_community

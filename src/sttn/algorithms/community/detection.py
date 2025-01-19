from typing import Optional, Tuple

import pandas as pd
import pycombo

from sttn.network import SpatioTemporalNetwork


def combo_communities(data: SpatioTemporalNetwork, weight: Optional[str] = None, random_seed: Optional[int] = 0, **kwargs) -> Tuple[SpatioTemporalNetwork, float]:
    graph = data.to_multigraph()
    partition_dict, modularity = pycombo.execute(graph, weight=weight, random_seed=random_seed, **kwargs)
    partition_df = pd.DataFrame.from_dict(partition_dict, orient='index', columns=['cluster'])
    data_with_community = data.join_node_labels(partition_df)
    return data_with_community, modularity

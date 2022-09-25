from __future__ import annotations
from typing import Any, Dict

from sttn.plot.keplergl.layer import GeoMapLayer


class MapConfig:
    def __init__(self, data_id: str):
        self.layers = []
        self._data_id = data_id

    def add_geo_layer(self, label: str, color_column: str) -> MapConfig:
        layer = GeoMapLayer(data_id=self._data_id, label=label, color_column=color_column)
        self.layers.append(layer)
        return self

    def to_dict(self) -> Dict[str, Any]:
        layer_configs = [layer.get_config() for layer in self.layers]
        return {'version': 'v1', 'config': {'visState': {'filters': [],
                                                         'layers': layer_configs}}}

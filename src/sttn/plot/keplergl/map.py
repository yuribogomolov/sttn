from __future__ import annotations

from typing import Any, Dict, Tuple

from sttn.plot.keplergl.filter import TimeRangeFilter
from sttn.plot.keplergl.layer import ArcLayer, GeoMapLayer, HeatMapLayer


class MapConfig:
    def __init__(self, data_id: str):
        self.layers = []
        self.filters = []
        self._data_id = data_id

    def with_data_id(self, data_id) -> MapConfig:
        self._data_id = data_id
        return self

    def add_geo_layer(self, label: str, color_column: str) -> MapConfig:
        layer = GeoMapLayer(data_id=self._data_id, label=label, color_column=color_column)
        self.layers.append(layer)
        return self

    def add_arc_layer(self, label: str, origin_lat: str, origin_lng: str,
                      destination_lat: str, destination_lng: str, size_column: str) -> MapConfig:
        layer = ArcLayer(data_id=self._data_id, label=label, origin_lat=origin_lat, origin_lng=origin_lng,
                         destination_lat=destination_lat, destination_lng=destination_lng, size_column=size_column)
        self.layers.append(layer)
        return self

    def add_heatmap_layer(self, label: str, lat_column: str, lng_column: str, weight_column: str):
        layer = HeatMapLayer(data_id=self._data_id, label=label, lat_column=lat_column, lng_column=lng_column,
                             weight_column=weight_column)
        self.layers.append(layer)
        return self

    def add_time_range_filter(self, time_column: str, window: Tuple[int, int], y_axis_column: str) -> MapConfig:
        range_filter = TimeRangeFilter(data_id=self._data_id, time_column=time_column, window=window,
                                       y_axis_column=y_axis_column)
        self.filters.append(range_filter)
        return self

    def to_dict(self) -> Dict[str, Any]:
        layer_configs = [layer.get_config() for layer in self.layers]
        filter_configs = [map_filter.get_config() for map_filter in self.filters]
        return {'version': 'v1', 'config': {'visState': {'filters': filter_configs,
                                                         'layers': layer_configs}}}

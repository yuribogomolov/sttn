import random
import string
from abc import ABC, abstractmethod
from typing import Any, Dict

HIGHLIGHT_COLOR = [252, 242, 26, 255]
COLOR_RANGE = {'name': 'Global Warming',
               'type': 'sequential',
               'category': 'Uber',
               'colors': ['#5A1846',
                          '#900C3F',
                          '#C70039',
                          '#E3611C',
                          '#F1920E',
                          '#FFC300']}
DEFAULT_HEATMAP_RADIUS = 10


class MapLayer(ABC):
    _id: str

    @property
    def layer_id(self):
        return self._id

    @staticmethod
    def generate_id() -> str:
        return ''.join(random.choices(string.ascii_lowercase, k=5))

    @property
    @abstractmethod
    def layer_type(self) -> str:
        return NotImplemented

    @property
    @abstractmethod
    def config(self) -> Dict[str, Any]:
        return NotImplemented

    @property
    @abstractmethod
    def visual_channels(self) -> Dict[str, Any]:
        return NotImplemented

    def get_config(self) -> Dict[str, Any]:
        return {'id': self.layer_id,
                'type': self.layer_type,
                'config': self.config,
                'visualChannels': self.visual_channels
                }


class GeoMapLayer(MapLayer):
    def __init__(self, data_id: str, label: str, color_column: str):
        self._id = MapLayer.generate_id()
        self._data_id = data_id
        self._label = label
        self._color_column = color_column

    @property
    def config(self) -> Dict[str, Any]:
        return {'dataId': self._data_id,
                'label': self._label,
                'color': [137, 218, 193],
                'highlightColor': HIGHLIGHT_COLOR,
                'columns': {'geojson': 'geometry'},
                'isVisible': True,
                'visConfig': {'opacity': 0.6,
                              'strokeOpacity': 0.8,
                              'thickness': 0.5,
                              'strokeColor': [179, 173, 158],
                              'colorRange': COLOR_RANGE,
                              'enableElevationZoomFactor': True,
                              'stroked': True,
                              'filled': True,
                              'enable3d': False,
                              'wireframe': False},
                'hidden': False}

    @property
    def visual_channels(self) -> Dict[str, Any]:
        return {'colorField': {'name': self._color_column,
                               'type': 'real'},
                'colorScale': 'quantize',
                'strokeColorField': None,
                'strokeColorScale': 'quantile',
                'sizeField': None,
                'sizeScale': 'linear',
                'heightField': None,
                'heightScale': 'linear',
                'radiusField': None,
                'radiusScale': 'linear'}

    @property
    def layer_type(self) -> str:
        return 'geojson'


class ArcLayer(MapLayer):
    def __init__(self, data_id: str, label: str, origin_lat: str, origin_lng: str,
                 destination_lat: str, destination_lng: str, size_column: str):
        self._id = MapLayer.generate_id()
        self._data_id = data_id
        self._label = label
        self._origin_lat = origin_lat
        self._origin_lng = origin_lng
        self._destination_lat = destination_lat
        self._destination_lng = destination_lng
        self._size_column = size_column

    @property
    def layer_type(self) -> str:
        return 'arc'

    @property
    def config(self) -> Dict[str, Any]:
        return {'dataId': self._data_id,
                'label': self._label,
                'color': [34, 63, 154],
                'highlightColor': HIGHLIGHT_COLOR,
                'columns': {'lat0': self._origin_lat,
                            'lng0': self._origin_lng,
                            'lat1': self._destination_lat,
                            'lng1': self._destination_lng},
                'isVisible': True,
                'visConfig': {'opacity': 0.8,
                              'thickness': 2,
                              'colorRange': COLOR_RANGE,
                              'sizeRange': [0, 10],
                              'targetColor': None},
                'hidden': False,
                'textLabel': [{'field': None,
                               'color': [255, 255, 255],
                               'size': 18,
                               'offset': [0, 0],
                               'anchor': 'start',
                               'alignment': 'center'}]}

    @property
    def visual_channels(self) -> Dict[str, Any]:
        return {'colorField': None,
                'colorScale': 'quantile',
                'sizeField': {'name': self._size_column, 'type': 'real'},
                'sizeScale': 'linear'}


class HeatMapLayer(MapLayer):
    def __init__(self, data_id: str, label: str, lat_column: str, lng_column: str, weight_column: str):
        self._id = MapLayer.generate_id()
        self._data_id = data_id
        self._label = label
        self._lat = lat_column
        self._lng = lng_column
        self._weight_column = weight_column

    @property
    def layer_type(self) -> str:
        return 'heatmap'

    @property
    def config(self) -> Dict[str, Any]:
        return {'dataId': self._data_id,
                'label': self._label,
                'color': [130, 154, 227],
                'highlightColor': HIGHLIGHT_COLOR,
                'columns': {'lat': self._lat,
                            'lng': self._lng},
                'isVisible': True,
                'visConfig': {'opacity': 0.8,
                              'colorRange': COLOR_RANGE,
                              'radius': DEFAULT_HEATMAP_RADIUS},
                'hidden': False,
                'textLabel': [{'field': None,
                               'color': [255, 255, 255],
                               'size': 18,
                               'offset': [0, 0],
                               'anchor': 'start',
                               'alignment': 'center'}]}

    @property
    def visual_channels(self) -> Dict[str, Any]:
        return {'weightField': {'name': self._weight_column, 'type': 'real'},
                'weightScale': 'linear'}

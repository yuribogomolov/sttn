from abc import ABC, abstractmethod
from typing import Any, Dict
import random
import string


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
                'highlightColor': [252, 242, 26, 255],
                'columns': {'geojson': 'geometry'},
                'isVisible': True,
                'visConfig': {'opacity': 0.6,
                              'strokeOpacity': 0.8,
                              'thickness': 0.5,
                              'strokeColor': [179, 173, 158],
                              'colorRange': {'name': 'Global Warming',
                                             'type': 'sequential',
                                             'category': 'Uber',
                                             'colors': ['#5A1846',
                                                        '#900C3F',
                                                        '#C70039',
                                                        '#E3611C',
                                                        '#F1920E',
                                                        '#FFC300']},
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
                'colorScale': 'quantile',
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

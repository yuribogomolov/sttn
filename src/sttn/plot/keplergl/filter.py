import random
import string
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class MapFilter(ABC):
    _id: str
    _data_id: List[str]

    @property
    def filter_id(self) -> str:
        return self._id

    @property
    def data_id(self) -> List[str]:
        return self._data_id

    @staticmethod
    def generate_id() -> str:
        return ''.join(random.choices(string.ascii_lowercase, k=5))

    @property
    @abstractmethod
    def filter_type(self) -> str:
        return NotImplemented

    @abstractmethod
    def extra_config(self) -> Dict[str, Any]:
        return NotImplemented

    def get_config(self) -> Dict[str, Any]:
        config = {'dataId': self._data_id,
                  'id': self.filter_id,
                  'type': self.filter_type,
                  }
        config.update(self.extra_config())

        return config


class TimeRangeFilter(MapFilter):
    def __init__(self, data_id: str, time_column: str, window: Tuple[int, int], y_axis_column: str):
        self._id = MapFilter.generate_id()
        self._data_id = [data_id]
        self._time_column = time_column
        self._time_window = [window[0], window[1]]
        self._y_axis_column = y_axis_column

    @property
    def filter_type(self) -> str:
        return 'timeRange'

    def extra_config(self) -> Dict[str, Any]:
        return {'name': [self._time_column], 'value': self._time_window, 'enlarged': True,
                'plotType': 'lineChart',
                'animationWindow': 'free',
                'yAxis': {'name': self._y_axis_column, 'type': 'integer'},
                'speed': 1}

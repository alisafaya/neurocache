from dataclasses import dataclass, field
from typing import Union

from neurocache.config import NeurocacheConfig

from .cache_utils import CacheType


@dataclass
class OnDeviceCacheConfig(NeurocacheConfig):
    cache_type: Union[str, CacheType] = field(
        default="FIFO", metadata={"help": "Cache type [FIFO, LRU]"}
    )

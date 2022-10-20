from dataclasses import dataclass
from typing import Tuple

from summon.torch import Numeric


@dataclass
class BanishedNumeric:
    rows: Tuple[int, int]
    min: float
    max: float
    model: Numeric

import pandas as pd

from summon.models import BanishedNumeric


# ensure inputs & outputs are compatible with numeric model
# use automatic feature/output scaling/normalization


def banish_numeric(data: pd.Series) -> BanishedNumeric:
    # min/max scaling
    min = data.min()
    max = data.max()
    data = (data - min) / (max - min)

    pass

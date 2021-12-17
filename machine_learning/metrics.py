from typing import Callable, Dict

import numpy
from pandas import DataFrame, Series
from more_itertools import windowed

METRICS = {}


def co_methylation_percentage(patterns: DataFrame, n: int, m: int, coverage_cutoff: int) -> float:
    cnt = patterns.shape[0]
    if cnt < coverage_cutoff:
        return numpy.nan

    for pattern in patterns:
        for window in windowed(pattern, m):
            if sum(window == 3) >= n:
                break
        else:
            cnt -= 1

    return cnt / patterns.shape[0]


def metrics(regions_patterns: Dict[str, DataFrame], metric: Callable, **kwargs):
    data = Series(index=regions_patterns.keys())
    for region_id, region_patterns in regions_patterns.items():
        data[region_id] = metric(region_patterns, *kwargs)

    return data

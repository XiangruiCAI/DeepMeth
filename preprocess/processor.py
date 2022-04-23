from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy
from more_itertools import windowed
from pandas import DataFrame, Series



class PatternProcessor:

    @abstractmethod
    def process(self, sample_data: Dict[str, Tuple[DataFrame, Series]]) -> DataFrame:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

class MetricProcessor(PatternProcessor):

    def __init__(self, m: int, n: int, minimum_coverage: int):
        self.m = m
        self.n = n
        self.minimum_coverage = minimum_coverage

    @property
    def name(self) -> str:
        return f'{self.m}_{self.n}'

    def process(self, sample_data: Dict[str, Tuple[DataFrame, Series]]) -> Series:
        df = Series(index=sample_data.keys())

        for region_id, (reads, counts) in sample_data.items():
            if counts.sum() < self.minimum_coverage:
                data = numpy.nan
            else:
                reads = reads.applymap({'T': 0, 'N': 0, 'C': 1}.get)
                cm = 0

                read: numpy.ndarray
                for read, count in zip(reads.to_numpy(), counts):

                    if len(read) < self.n:
                        read = numpy.pad(read, (self.n - len(read)), constant_values=0)

                    for window in windowed(read, self.n):
                        if numpy.sum(window) >= self.m:
                            cm += count
                            break
                data = cm / counts.sum()
            df.loc[region_id] = data
        return df


class RegionSizeProcessor(PatternProcessor):

    def __init__(self, out_dir: Path):
        self.data = defaultdict(lambda: DataFrame(index=('height', 'width')))
        self.out_dir = out_dir

    def name(self) -> str:
        return 'regions'

    def process(self, sample_data: Dict[str, Tuple[DataFrame, Series]]) -> Series:
        df = Series(index=sample_data.keys())

        for region_id, (reads, counts) in sample_data.items():
            data = {
                'height': counts.sum(),
                'width': reads.shape[1]
            }
            df[region_id] = data

        return df

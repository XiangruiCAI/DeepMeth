import logging
from pathlib import Path
from typing import List

from pandas import Series

import utils


def create_top_regions(
        experiment_dir: Path,
        top_regions_file_pattern: str,
        top_n: int,
        region_id_label: str,
        importance_label: str,
        n_splits: List[int],
        overwrite: bool = False):
    top_regions = utils.read_top_regions(
        top_regions_file_pattern=top_regions_file_pattern,
        top_n=top_n,
        region_id_label=region_id_label,
        importance_label=importance_label,
        n_splits=n_splits)

    for i in range(*n_splits):
        top_regions_file = experiment_dir.joinpath(f'{i}').joinpath('regions.tsv')

        if overwrite or not top_regions_file.exists():

            regions = Series.sort_values(top_regions[i]).rename('region_id')
            regions.to_csv(top_regions_file, sep='\t', index=False, header=True)

        else:
            msg = f'{top_regions_file} already exists, ignore it'
            logging.warning(msg)


def create_top_regions_by_bed(experiment_dir: Path, bed_file: Path, n_splits: List[int], overwrite):
    regions = utils.read_bed_file(bed_file)
    for i in range(*n_splits):
        target_file = experiment_dir.joinpath(f'{i}', 'regions.tsv')
        if overwrite or not target_file.exists():

            regions.to_csv(target_file, sep='\t', header=True, index=False)
        else:
            msg = f'{target_file} already exists, ignore it'
            logging.warning(msg)

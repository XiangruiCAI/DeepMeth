import logging
from pathlib import Path
from typing import Dict, List

import pandas
from pandas import Series, DataFrame

import utils


def create_splits_by_single_file(
        experiment_dir: Path,
        splits_file: Path,
        info_file: Path,
        categories: Dict[str, str],
        over_write: bool = False,
        **kwargs
):
    info = utils.read_pathology_class(info_file)
    splits = utils.read_splits(splits_file)

    n_splits = len(splits.columns)

    for i in range(1, 1 + n_splits):
        split_dir = experiment_dir.joinpath(f'{i}', 'split')
        try:
            split_dir.mkdir(parents=True, exist_ok=over_write)

            split: Series = splits[i]
            for t, name in categories.items():
                sample_ids = split.loc[split == t].index
                patho_class = info.reindex(sample_ids)
                DataFrame.to_csv(patho_class,
                                 split_dir.joinpath(name).with_suffix('.tsv'),
                                 sep='\t',
                                 index_label='sample_id')
        except FileExistsError:
            msg = f'{split_dir} already exists, ignore running it'
            logging.warning(msg)


def create_splits_by_file_patterns(
        experiment_dir: Path,
        file_patterns: Dict[str, str],
        n_splits: List[int],
        overwrite: bool = False,
        **kwargs):
    sep = kwargs.pop('sep', '\t')
    index_col = kwargs.pop('index_col', 0)

    for i in range(*n_splits):
        target_dir = experiment_dir.joinpath(f'{i}', 'split')

        try:
            target_dir.mkdir(parents=True, exist_ok=overwrite)
        except FileExistsError:
            logging.warning(f'{target_dir} already exists, ignore running it.')
            continue

        for t, pattern in file_patterns.items():
            data = pandas.read_csv(pattern.format(f'{i}'), sep=sep, index_col=index_col)['pathology_class']
            data = data.map({'malignant': 1, 'benign': 0})
            data.to_csv(target_dir.joinpath(t).with_suffix('.tsv'), sep='\t', index_label='sample_id', header=True)

        pass
    pass

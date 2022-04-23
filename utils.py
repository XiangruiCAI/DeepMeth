import csv
import os
import pickle
import re
from pathlib import Path
from typing import Collection, NoReturn, List

import numpy
import numpy as np
import pandas
import pandas as pd
import yaml
from more_itertools import collapse
from pandas import DataFrame, Series
# noinspection PyUnresolvedReferences
from pandas.core.strings import StringMethods


def init_seed(seed: int):
    import random
    random.seed(seed)

    import numpy
    numpy.random.seed(seed)

    try:
        import torch
        if isinstance(seed, int):
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_sample_id(string, **kwargs):
    if 'regular' in kwargs:
        regular = kwargs['regular']
    else:
        regular = r'(LC015|EPIC|AIM)\-[0-9]+'
    try:
        sample_id = re.search(regular, string).group()
    except Exception as e:
        print("Error: ", e)
        raise RuntimeError(e)
    return sample_id


def read_proba(proba_file: Path) -> Series:
    data = pandas.read_csv(proba_file.with_suffix('.tsv'), sep='\t', index_col='sample_id', squeeze=True)
    return data


def write_proba(proba: Series, proba_file: Path) -> NoReturn:
    proba.to_csv(proba_file.with_suffix('.tsv'), sep='\t', index_label='sample_id', header=True)


def load_model(lgbm_file: Path):
    return pickle.loads(lgbm_file.read_bytes())


def dump_model(model, model_file: Path):
    return model_file.write_bytes(pickle.dumps(model))


def collect_probas(result_dir: Path, splits: Collection[int]):
    data: DataFrame = pandas.concat([
        read_proba(result_dir.joinpath(f'{split}', 'proba.tsv')).rename(split)
        for split in splits
    ], axis=1, sort=True)
    data.transpose().to_csv(result_dir.joinpath('probas.tsv'), sep='\t', index_label='split')


def collect_metrics(result_dir: Path, splits: Collection[int]):
    data: DataFrame = pandas.concat([
        pandas.read_csv(
            result_dir.joinpath(f'{split}', 'metrics.tsv'), sep='\t',
            header=None, index_col=0, names=('metric', f'{split}'))
        for split in splits
    ], axis=1)
    data.transpose().to_csv(result_dir.joinpath('metrics.tsv'), sep='\t', index_label='split')


def load_data(data_file: Path,
              sample_ids: List[str]) -> DataFrame:
    """

    Args:
        data_file: A file or a directory stores data
        sample_ids: tuple of samples list,
            ['LC015-train1', 'LC015-train2', ...]
        external_data: external data concat to


    Returns:
        tuple of data in each group
    """

    data: DataFrame = pandas.concat([
        pandas.Series(
            numpy.loadtxt(str(data_file.joinpath(sample_id).with_suffix('.txt'))).flatten(),
            name=sample_id)
        for sample_id in sample_ids
    ], axis=1)
    print("data shape : ", data.shape)
    return data.transpose()


def read_splits(split_file) -> DataFrame:
    data = pandas.read_csv(split_file, sep='\t', index_col=0)
    data.columns = range(1, 1 + data.shape[1])
    return data


def read_pathology_class(info_file) -> Series:
    data = read_info(info_file)
    class_map = {
        'malignant': 1,
        'benign': 0,
        'normal': 0
    }
    return data['pathology_class'].map(class_map)


def read_top_regions(
        top_regions_file_pattern: str,
        top_n: int,
        region_id_label: str,
        importance_label: str,
        n_splits: List[int],
        **kwargs
) -> DataFrame:
    top_regions = DataFrame()

    for i in range(*n_splits):
        top_regions_file = top_regions_file_pattern.format(i)
        data = pandas.read_csv(
            top_regions_file,
            sep=kwargs.pop('sep', ','),
            usecols=[0, 2])
        data.columns = ['feature', 'score']
        sm: StringMethods = data[region_id_label].str
        data[region_id_label] = list(collapse(sm.findall(r'\w+:\d+-\d+')))

        data.sort_values(importance_label, ascending=False, inplace=True)
        regions = data[region_id_label][:top_n]
        top_regions[i] = Series.sort_values(regions).reset_index(drop=True)

    return top_regions


def read_all_regions(all_region_file):
    return DataFrame()


def read_info(info_file: str):
    return pandas.read_csv(info_file, sep='\t', index_col='sample_ID')


def read_df_data(df_file, index_col=0):
    data = pandas.read_csv(df_file, sep='\t', index_col=index_col)

    return data


def read_bed_file(bed_file: Path) -> Series:
    data = pandas.read_csv(bed_file, sep='\t', names=('chr', 'start', 'end'), usecols=('chr', 'start', 'end'),
                           dtype=str)
    regions: Series = (data['chr'] + ':' + data['start'] + '-' + data['end']).drop_duplicates()
    regions.rename('region_id', inplace=True)
    regions.sort_values(inplace=True)
    return regions


def read_split_file(split_file: Path) -> Series:
    """read a split file and returns sample ids and pathology class

    Args:
        split_file: either `train.tsv` or `test.tsv`

    Returns:
        A list of sample ids and a list with corresponding pathology class
    """
    data = pandas.read_csv(split_file, sep='\t', index_col='sample_id')
    return data['pathology_class']


current_path = os.path.dirname(__file__)


def get_params():
    print(current_path)
    with open(current_path + '/../params.yaml', 'r', encoding='utf-8') as f:
        params = yaml.load(f)
    return params


def get_specified_files(source_dir, file_type):
    """
        Get all the specified files in a directory.
    Args:
        source_dir: directory.
        file_type: specified file type.

    Returns:

    """
    files_list = list()
    for root, directory, files in os.walk(source_dir, file_type):
        for file_name in files:
            name, suf = os.path.splitext(file_name)
            if suf == file_type:
                files_list.append(root + "/" + file_name)
    return files_list


def get_region_pon_dict(file_path):
    all_region_dict = dict()
    data = pd.read_csv(file_path, sep='\t', header=None, names=['chr', 'start', 'end', 'id', 'number', 'pon', 'sites'])
    region_key = np.array(data['chr'].map(str) + '_' + (data['start']).map(str) +
                          '_' + data['end'].map(str)).reshape(-1, 1)
    region_pon = np.array(data['chr'].map(str) + '_' + (data['start']).map(str) +
                          '_' + data['end'].map(str) + '_' + data['pon'].map(str)).reshape(-1, 1)
    region_trans = np.append(region_key, region_pon, axis=1)
    for row in region_trans:
        all_region_dict.setdefault(row[0], list()).append(row[1])
    return all_region_dict


def get_selected_region(sorted_region_file):
    """
        Get selected region from file.
    Args:
        sorted_region_file: Sorted region information, each line refers to a region.
    Returns:
        region_list: Selected top region list.
    """
    data = pd.read_csv(sorted_region_file, usecols=['region_id'], sep='\t', squeeze=True)
    region_list = data.tolist()
    return region_list


def get_samples_label(sample_info_list, id_idx=0, label_idx=3):
    """
        Get all samples` label.
    Args:
        sample_info_list: A list consists of all sample_info.list files(No Headers).
        id_idx: The sample ID index in sample_info.list file.
        label_idx: The sample label index in sample_info.list file.
    Returns:
        sample_label_dict: dict{'Sample_ID': Sample_label}
    """
    sample_label_dict = dict()
    for sample_info_path in sample_info_list:
        f = open(sample_info_path)
        f_reader = csv.reader(f, delimiter='\t')
        for row in f_reader:
            sample_label_dict[row[id_idx]] = 1 if row[label_idx] == 'malignant' else 0
        f.close()
    return sample_label_dict


def get_train_test(split_dir: Path):
    train_path = split_dir.joinpath('train.tsv')
    train_data = pd.read_csv(train_path, sep='\t')
    train_list = np.array(train_data['sample_id'])
    train_label = np.array(train_data['pathology_class']).astype(np.int32)

    test_path = split_dir.joinpath('test.tsv')
    test_data = pd.read_csv(test_path, sep='\t')
    test_list = np.array(test_data['sample_id'])
    test_label = np.array(test_data['pathology_class']).astype(np.int32)

    return train_list, train_label, test_list, test_label


def get_region_dict(regions):
    return {region: i for i, region in enumerate(regions)}


def read_sample_region(target_file):
    return pandas.read_csv(target_file, sep='\t', squeeze=True)

from argparse import ArgumentParser, ONE_OR_MORE
from pathlib import Path
from typing import Any, Callable, Dict, List

from functools import partial

import numpy
import torch
import yaml
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

import utils
from deep_learning.encode import encode
from deep_learning.train import ae_train
from machine_learning.sklearn import sklearn_test, sklearn_train, sklearn_metric
from preprocess.create_dataset import create_dataset
from preprocess.create_split import (create_splits_by_file_patterns,
                                     create_splits_by_single_file)
from preprocess.regions import create_top_regions, create_top_regions_by_bed
from utils import get_selected_region, get_train_test


def prepare(
        experiment_dir: Path,
        tnc_dir: Path,
        splits_params: Dict[str, Any],
        region_params: Dict[str, Any],
        n_splits: List[int],
        overwrite: bool = False):
    # dataset dir
    create_dataset(experiment_dir=experiment_dir, tnc_dir=Path(tnc_dir), overwrite=overwrite)

    # create splits
    others = splits_params.get('others', {})
    if 'splits_file' in splits_params and 'info_file' in splits_params and 'categories' in splits_params:

        create_splits_by_single_file(
            experiment_dir=experiment_dir,
            splits_file=splits_params.pop('splits_file'),
            info_file=splits_params.pop('info_file'),
            categories=splits_params.pop('categories'),
            over_write=overwrite,
            **others
        )
    elif 'file_patterns' in splits_params:
        file_patterns = splits_params.pop('file_patterns')

        create_splits_by_file_patterns(
            experiment_dir=experiment_dir,
            file_patterns=file_patterns,
            n_splits=n_splits,
            overwrite=overwrite,
            **others
        )
    else:
        raise RuntimeError()

    # create using regions
    if 'top_regions_file_pattern' in region_params \
            and 'top_n' in region_params \
            and 'region_id_label' in region_params \
            and 'importance_label' in region_params:

        create_top_regions(
            experiment_dir=experiment_dir,
            n_splits=n_splits,
            top_regions_file_pattern=region_params.pop('top_regions_file_pattern'),
            top_n=region_params.pop('top_n'),
            region_id_label=region_params.pop('region_id_label'),
            importance_label=region_params.pop('importance_label'),
            overwrite=overwrite)
    elif 'bed_file' in region_params:
        bed_file = Path(region_params.pop('bed_file'))
        create_top_regions_by_bed(
            experiment_dir=experiment_dir,
            bed_file=bed_file,
            n_splits=n_splits,
            overwrite=overwrite)
    else:
        raise RuntimeError()


def auto_encoder_train(experiment_dir: Path, n_splits: List[int], h_params: Dict[str, Any]):
    for i in range(*n_splits):
        split_dir = experiment_dir.joinpath(f'{i}')
        train_samples, train_labels, test_samples, test_labels = get_train_test(split_dir=split_dir.joinpath('split'))
        ae_train(
            experiment_dir.joinpath('dataset'),
            train_samples=train_samples,
            selected_regions=get_selected_region(split_dir.joinpath('regions.tsv')),
            serialization_dir=split_dir,
            params=h_params,
        )


def auto_encoder_encode(experiment_dir: Path, n_splits: List[int], encode_splits: List[str], h_params: Dict):
    dataset_dir = experiment_dir.joinpath('dataset')
    for i in range(*n_splits):
        split_dir = experiment_dir.joinpath(f'{i}')

        samples = numpy.concatenate([
            utils.read_split_file(split_dir.joinpath('split', t).with_suffix('.tsv')).index
            for t in encode_splits])

        selected_regions = get_selected_region(str(split_dir) + '/regions.tsv')

        split_dir.joinpath('encode').mkdir(exist_ok=True)
        encode(
            tnc_dir=dataset_dir,
            samples=samples,
            selected_regions=selected_regions,
            model_file=split_dir.joinpath('saved_models').joinpath('encoder_epoch_0.pkl'),
            encode_dir=split_dir.joinpath('encode'),
            device='cuda' if torch.cuda.is_available() else 'cpu',
            **h_params.get('encoder', {})
        )


def lightgbm_train(
        experiment_dir: Path,
        n_splits: List[int],
        h_param: Dict[str, Any],
        over_sample_param: Dict = None
):
    # classifier = LGBMClassifier(**h_param)
    classifier = RandomForestClassifier(**h_param)

    # external_data = [utils.read_df_data(ex) for ex in external_files]

    for i in tqdm(range(*n_splits)):
        utils.init_seed(1234)
        split_dir = experiment_dir.joinpath(f'{i}')
        train_samples = utils.read_split_file(Path(split_dir).joinpath('split', 'train.tsv'))

        sklearn_train(
            classifier=classifier,
            samples=train_samples,
            code_dir=split_dir.joinpath('encode'),
            # model_file=split_dir.joinpath('saved_models', 'lgbm').with_suffix('.pkl'),
            model_file=split_dir.joinpath('saved_models', 'rf_origin').with_suffix('.pkl'),
            over_sample_param=over_sample_param)

        # sklearn_test(
        #     sample_ids=train_samples.index,
        #     code_dir=split_dir.joinpath('encode'),
        #     model_file=split_dir.joinpath('saved_models', 'lgbm').with_suffix('.pkl'),
        #     proba_file=split_dir.joinpath('proba.tsv'),
        #     external_data=external_data
        # )


# def lightgbm_test(experiment_dir: Path, n_splits: List[int], test_splits: List[str], external_files: List[str]):
#     for split in test_splits:
#         _lightgbm_test(experiment_dir, n_splits, split, external_files)


def _lightgbm_test(experiment_dir: Path, n_splits: List[int], split_file: str, h_param: Dict[str, Any] = None):
    gps_dir = Path('/home/wangshichao/public/ngs_lung_cancer/sample_ids')
    for i in tqdm(range(*n_splits)):
        utils.init_seed(1234)
        split_dir = experiment_dir.joinpath(f'{i}')
        # test_samples = utils.read_split_file(split_dir.joinpath('split', split_file).with_suffix('.tsv'))
        test_samples = utils.read_split_file(gps_dir.joinpath(split_file).with_suffix('.txt'))
        sklearn_test(
            sample_ids=test_samples.index.tolist(),
            code_dir=split_dir.joinpath('encode'),
            # model_file=split_dir.joinpath('saved_models', 'lgbm').with_suffix('.pkl'),
            model_file=split_dir.joinpath('saved_models', 'rf_origin').with_suffix('.pkl'),
            proba_file=split_dir.joinpath('proba_rf.tsv')
        )
        sklearn_metric(samples=test_samples, proba_file=split_dir.joinpath('proba_rf.tsv'))
        pass

    pass


# deprecated see lightgbm_test above
lightgbm_test = partial(_lightgbm_test, split_file='test.tsv')
lightgbm_val = partial(_lightgbm_test, split_file='val.tsv')
lightgbm_external = partial(_lightgbm_test, split_file='external.tsv')
lightgbm_independent_test = partial(_lightgbm_test, split_file='independent_test.tsv')
lightgbm_31gps = partial(_lightgbm_test, split_file='31-GPS.txt')
lightgbm_44gps = partial(_lightgbm_test, split_file='44-GPS.txt')
lightgbm_57gps = partial(_lightgbm_test, split_file='57-GPS.txt')
lightgbm_75gps = partial(_lightgbm_test, split_file='75-GPS.txt')


def create_parser(sub_commands: Dict[str, Callable]) -> ArgumentParser:
    parser = ArgumentParser()

    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--experiment_dir', type=str)
    parent_parser.add_argument('--n_splits', type=int, nargs=ONE_OR_MORE)

    sub_parsers = parser.add_subparsers(dest='command')
    for command, func in sub_commands.items():
        sub_parser = sub_parsers.add_parser(command, parents=[parent_parser])
        sub_parser.set_defaults(func=func)

    return parser


def load_config(config_file: Path, sub) -> Dict[str, Any]:
    if config_file.suffix in ('.yaml', '.yml'):
        config = yaml.load(config_file.open(), Loader=yaml.FullLoader)
    else:
        raise NotImplemented()

    return config[sub]

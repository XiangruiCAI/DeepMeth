import shutil
from pathlib import Path
from typing import List, Collection

import numpy
import pandas
from fire import Fire
from more_itertools import flatten
from pandas import Series, DataFrame
from pandas.core.strings import StringMethods
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve

import utils

__all__ = ['update_sample_info', 'update_metric', 'collect_probas', 'preprocess']

from preprocess.pattern import process_pattern_files

from preprocess.processor import MetricProcessor


def cutoff_analyze(experiment_dir: Path, n_splits: List[int]):
    val_auc, test_auc = 0, 0
    acc, sensitivity, specificity = 0, 0, 0
    for i in range(*n_splits):
        split_dir = experiment_dir.joinpath(f'{i}')
        proba = utils.read_proba(split_dir.joinpath('proba.tsv'))
        val_samples = utils.read_split_file(Path(split_dir).joinpath('split', 'val.tsv'))
        val_proba = proba[val_samples.index]
        val_auc += roc_auc_score(val_samples, val_proba)

        cutoff = get_cutoff_by_sensitivity(val_samples, val_proba, sensitivity=0.95)

        test_samples = utils.read_split_file(split_dir.joinpath('split', 'test.tsv'))
        test_proba = proba[test_samples.index]
        test_predict = (test_proba >= cutoff)

        test_auc += roc_auc_score(test_samples, test_proba)
        acc += accuracy_score(test_samples, test_predict)

        cm = confusion_matrix(test_samples, test_predict)
        sensitivity += (cm[1][1] / cm[1].sum())
        specificity += (cm[0][0] / cm[0].sum())

        print(cm)
        pass
    print(val_auc)
    print(test_auc)
    print(acc)
    print(sensitivity)
    print(specificity)


def get_cutoff_by_sensitivity(target: Series, proba: Series, sensitivity: float):
    fprs, tprs, thresholds = roc_curve(target, proba, drop_intermediate=False)
    index = numpy.argmax(tprs >= sensitivity)
    threshold = thresholds[index]
    print(1 - fprs[index])
    return threshold


def copy_split(from_where: str, to_where: str, split_name: str, splits: Collection[int]):
    for split in splits:
        src = Path(from_where).joinpath(f'{split}', 'split', split_name).with_suffix('.tsv')
        trg = Path(to_where).joinpath(f'{split}', 'split')
        if src.exists() and trg.exists():
            shutil.copy2(str(src), str(trg))
    pass


def update_sample_info(old_info_file: str, new_info_file: str):
    old_info = pandas.read_csv(old_info_file, sep='\t')
    new_info = pandas.read_csv(new_info_file, sep='\t').rename(columns={'sample': 'sample_id'})
    updated_info: DataFrame = pandas.concat((old_info, new_info), axis=0, join='inner')
    updated_info.drop_duplicates(subset='sample_id', keep='last', inplace=True)

    updated_info.to_csv(
        Path(old_info_file).parent.joinpath(f'LC_sample_info_{len(updated_info)}.tsv'),
        sep='\t', index=False)


def update_metric(old_metric_file: str, new_metric_file: str):
    old_metric_file = Path(old_metric_file)
    new_metric_file = Path(new_metric_file)
    old_metric = pandas.read_csv(old_metric_file, sep='\t', index_col=0)
    new_metric = pandas.read_csv(new_metric_file, sep='\t', index_col=0)
    sm: StringMethods = new_metric.columns.str
    new_metric.columns = flatten(sm.findall(r'\w+-\d+'))
    to_update = new_metric.columns.difference(old_metric.columns)
    updated_metrics = old_metric.join(new_metric[to_update], sort=True)
    updated_metrics.to_csv(old_metric_file.with_name(f'{old_metric_file.name}_{updated_metrics.shape[1]}'), sep='\t')


def collect_probas(experiment_dir: str, splits: Collection[int]):
    utils.collect_probas(Path(experiment_dir), splits)


def preprocess(pattern_dir: str):
    Path.home().joinpath('tmp').mkdir(parents=True, exist_ok=True)
    it = process_pattern_files(
        *Path(pattern_dir).glob('*.pattern'),
        processors=[MetricProcessor(2, 3, 30),
                    MetricProcessor(3, 5, 30)],
        out_dir=Path.home().joinpath('tmp'),
        n_threads=4
    )
    return list(it)

    pass


if __name__ == '__main__':
    Fire()

import logging
from copy import deepcopy
from pathlib import Path
from typing import Collection, List, Dict

import numpy
import pandas
from pandas import DataFrame, Series
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import utils


def train_lightgbm(
        classifier,
        samples_file: Path,
        code_dir: Path,
        model_file: Path,
        over_sample_param: Dict,
        external_data: Collection[DataFrame]):
    samples = utils.read_split_file(samples_file)
    sample_ids = samples.index
    train_x = utils.load_data(code_dir, sample_ids, external_data).values
    train_y = samples.values
    clf = deepcopy(classifier)

    if over_sample_param and len(over_sample_param) > 0:
        from machine_learning.imbalance import over_sample
        train_x, train_y = over_sample(over_sample_param, train_x, train_y)

    clf.fit(train_x, train_y)

    # write model
    utils.dump_model(clf, model_file)


def sklearn_train(classifier,
                  samples: Series,
                  code_dir: Path,
                  model_file: Path,
                  *,
                  over_sample_param: Dict):
    sample_ids = samples.index
    train_x = utils.load_data(code_dir, sample_ids).values
    train_y = samples.values
    clf = deepcopy(classifier)

    if over_sample_param is not None and len(over_sample_param) > 0:
        from machine_learning.imbalance import over_sample
        train_x, train_y = over_sample(over_sample_param, train_x, train_y)

    clf.fit(train_x, train_y)
    # write model
    utils.dump_model(clf, model_file)


def sklearn_test(
        sample_ids: List[str],
        code_dir: Path,
        model_file: Path,
        proba_file: Path):
    classifier = utils.load_model(model_file)

    test_x = utils.load_data(code_dir, sample_ids)

    test_proba = classifier.predict_proba(test_x.values)[:, 1]
    try:
        proba = utils.read_proba(proba_file)
    except FileNotFoundError:
        proba = pandas.Series()

    proba = proba.append(pandas.Series(data=test_proba, index=sample_ids), verify_integrity=False)
    proba = proba[~proba.index.duplicated()]
    utils.write_proba(proba, proba_file)


def sklearn_metric(samples: Series,
                   proba_file: Path):
    proba = utils.read_proba(proba_file)[samples.index]

    auc = roc_auc_score(samples.values, proba)
    tqdm.write(f'{auc}')
    logging.info(f'{auc}')

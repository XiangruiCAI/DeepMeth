import pandas
from lightgbm import LGBMClassifier
from pathlib import Path
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
import utils
from machine_learning.sklearn import sklearn_metric

def load_data(dir_path):
    files = ['GPS_scanorama_corrected.txt',
             'LC015_scanorama_corrected.txt',
             'LC021_scanorama_corrected.txt',
             'LC022_scanorama_corrected.txt',
             'Normal_1_scanorama_corrected.txt',
             'Normal_2_scanorama_corrected.txt']

    data: DataFrame = pandas.concat([
        pandas.read_csv((dir_path + '/' + file), sep='\t').transpose()
        for file in files
    ], axis=0)
    # for file in files:
    #     a = pandas.read_csv((dir_path + '/' + file), sep='\t')
    #     pandas.concat([data, a.transpose()], axis=0)
    return data


corrected_base_dir = '/home/limei/corrected data /Model6_AE/split_'
origin_base_dir = Path('/home/wangjiaxian/experiment/Model6_top200/')

def ml_train():
    for i in range(1, 11):
        dir_path = corrected_base_dir + str(i)
        data = load_data(dir_path)

        origin_split_dir = origin_base_dir.joinpath(f'{i}')
        train_samples = utils.read_split_file(Path(origin_split_dir).joinpath('split', 'train.tsv'))
        sample_ids = train_samples.index
        train_x = data[sample_ids]
        train_y = train_samples.values
        clf = LGBMClassifier(n_estimators=500, n_jobs=30)
        clf = RandomForestClassifier(n_estimators=500, n_jobs=30)
        clf.fit(train_x, train_y)
        # write model
        model_file = origin_split_dir.joinpath('saved_models', 'lgbm_corrected').with_suffix('.pkl')
        utils.dump_model(clf, model_file)


def ml_test(split_file):
    for i in range(1, 11):
        dir_path = corrected_base_dir + str(i)
        data = load_data(dir_path)

        origin_split_dir = origin_base_dir.joinpath(f'{i}')
        test_samples = utils.read_split_file(origin_split_dir.joinpath('split', split_file).with_suffix('.tsv'))
        sample_ids = test_samples.index
        test_x = data[sample_ids]
        test_y = test_samples.values

        model_file = origin_split_dir.joinpath('saved_models', 'lgbm_corrected').with_suffix('.pkl')
        classifier = utils.load_model(model_file)
        test_proba = classifier.predict_proba(test_x.values)[:, 1]
        proba_file = origin_split_dir.joinpath('proba_lgbm_corrected.tsv')
        try:
            proba = utils.read_proba(proba_file)
        except FileNotFoundError:
            proba = pandas.Series()

        proba = proba.append(pandas.Series(data=test_proba, index=sample_ids), verify_integrity=False)
        proba = proba[~proba.index.duplicated()]
        utils.write_proba(proba, proba_file)

        sklearn_metric(samples=test_samples, proba_file=origin_split_dir.joinpath('proba_lgbm_corrected.tsv'))


ml_train()
ml_test('test')
ml_test('independent_test')
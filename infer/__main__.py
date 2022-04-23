import logging
from argparse import ArgumentParser, ONE_OR_MORE, ZERO_OR_MORE
from pathlib import Path
from typing import List

import pandas

import utils
from deep_learning.encode import encode
from machine_learning.sklearn import sklearn_test
from preprocess.tnc_encode import tnc_encode


def path_type(p: str):
    return Path(p)


def infer(pattern_files: List[Path], regions_file: Path, model_dir: Path, out_dir: Path, external_files: List[Path],
          cache_dir: Path):
    tnc_dir = cache_dir.joinpath('tnc_encode')
    sample_ids = [tnc_encode(pattern_file=pattern_file, tnc_dir=tnc_dir) for pattern_file in pattern_files]

    used_regions = pandas.read_csv(regions_file, sep='\t')['region_id']
    encode_dir = out_dir
    out_dir.mkdir(exist_ok=True)
    encode(tnc_dir=tnc_dir,
           samples=sample_ids,
           selected_regions=used_regions,
           model_file=model_dir.joinpath('encoder_epoch_0.pkl'),
           encode_dir=encode_dir,
           device='cpu')

    external_data = [utils.read_df_data(external_file)[sample_ids] for external_file in external_files]

    sklearn_test(sample_ids=sample_ids,
                 code_dir=encode_dir,
                 model_file=model_dir.joinpath('lgbm.pkl'),
                 proba_file=out_dir.joinpath('proba.tsv'),
                 external_data=external_data)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument('--pattern_file', type=path_type, nargs=ONE_OR_MORE, required=True)
    parser.add_argument('--regions_file', type=path_type, required=True)
    parser.add_argument('--model_dir', type=path_type, required=True)
    parser.add_argument('--out_dir', type=path_type, required=True)
    parser.add_argument('--cache_dir', type=path_type, default=Path('/tmp'))
    parser.add_argument('--external_file', type=path_type, nargs=ZERO_OR_MORE)

    args = parser.parse_args()
    infer(pattern_files=args.pattern_file,
          regions_file=args.regions_file,
          model_dir=args.model_dir,
          out_dir=args.out_dir,
          external_files=args.external_file,
          cache_dir=args.cache_dir)

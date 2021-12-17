import logging
import re
from functools import partial
from itertools import chain
from multiprocessing.pool import Pool
from pathlib import Path
from typing import List

import pandas
from pandas import DataFrame
from tqdm import tqdm
from argparse import ArgumentParser, ONE_OR_MORE

from yaml import parse

code_map = {'T': '1', 'N': '2', 'C': '3'}


def tnc_encode(pattern_file: Path, tnc_dir: Path, overwrite: bool = True) -> str:
    tnc_dir.mkdir(exist_ok=True)
    sample_id = re.findall(r'\w+-\d+', pattern_file.name)[0]

    logging.info(f'start tnc encode {sample_id}')

    sample_dir = tnc_dir.joinpath(sample_id)
    if sample_dir.exists() and not overwrite:
        logging.warning(f'{sample_id} tnc code already exist, ignore it')
        return sample_id

    sample_dir.mkdir(exist_ok=overwrite)
    region_sizes = DataFrame(index=('height', 'width'))
    data = pandas.read_csv(pattern_file, sep='\t', low_memory=False)
    for chr_id in data['chromosome'].unique():
        one_chr_data = data.loc[data['chromosome'] == chr_id]
        for start in one_chr_data['start'].unique():
            one_chr_start_data = one_chr_data.loc[one_chr_data['start'] == start]
            for end in one_chr_start_data['end'].unique():
                one_chr_start_end_data = one_chr_start_data.loc[one_chr_start_data['end'] == end]

                region_id = f'{chr_id}:{start}-{end}'

                reads = one_chr_start_end_data['methylation_pattern']
                counts = one_chr_start_end_data['methylation_pattern_count']

                expanded_reads = reads.repeat(counts)
                split_reads: DataFrame = expanded_reads.str.split(r'', expand=True).iloc[:, 1:-1]
                mapped_reads = split_reads.applymap(code_map.get)

                mapped_reads.to_csv(sample_dir.joinpath(region_id).with_suffix('.tsv'),
                                    sep='\t',
                                    header=False,
                                    index=False)

                region_sizes[region_id] = mapped_reads.shape

    region_sizes.transpose().to_csv(
        sample_dir.joinpath('region_sizes').with_suffix('.tsv'),
        sep='\t',
        index_label='region_id'
    )

    logging.info(f'{sample_id} tnc encoded!')
    return sample_id


def tnc_encode_dirs(pattern_dirs: List[str], tnc_dir: str, n_threads: int):
    pattern_files = list(chain(*[list(Path(pattern_dir).glob(r'*.rm.MHC.chrX.pattern')) for pattern_dir in pattern_dirs]))
    with Pool(processes=n_threads) as pool:
        sample_ids = list(tqdm(
            pool.map(partial(tnc_encode, tnc_dir=Path(tnc_dir)), pattern_files),
            total=len(pattern_files)))

    logging.info('TNC encode complete !!!')
    return sample_ids


if __name__ == '__main__':
    parser = ArgumentParser(description='TNC encode')
    parser.add_argument('--pattern_dirs', type=str, help='the pattern files dir list', nargs=ONE_OR_MORE, required=True)
    parser.add_argument('--tnc_dir', type=str, help='tnc encode output dir', default='Data/tnc_encode')
    parser.add_argument('--n_threads', type=int, help='threads number', default=1)

    args = parser.parse_args()
    tnc_encode_dirs(args.pattern_dirs, args.tnc_dir, args.n_threads)

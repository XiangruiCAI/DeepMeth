import re
from pathlib import Path
from typing import List

import pandas
from pandas import DataFrame

from preprocess.processor import PatternProcessor


def process_pattern_files(
        *pattern_files: Path,
        processors: List[PatternProcessor],
        out_dir: Path,
        n_threads: int
) -> List[str]:
    for pattern_file in pattern_files:
        sample_id = re.findall(r'\w+-\d+', pattern_file.name)[0]
        process_one(pattern_file=pattern_file, sample_out_dir=out_dir.joinpath(sample_id), processors=processors)
        yield sample_id


def process_one(pattern_file: Path, sample_out_dir: Path, processors: List[PatternProcessor]):
    sample_out_dir.mkdir(exist_ok=True)
    sample_data = parse_pattern_file(pattern_file)
    for processor in processors:
        data = processor.process(sample_data)
        tgt_f = sample_out_dir.joinpath(f'{processor.name}').with_suffix('.tsv')
        data.to_csv(tgt_f, sep='\t', header=False)


def parse_pattern_file(pattern_file: Path):
    data = pandas.read_csv(pattern_file, sep='\t', low_memory=False)
    sample_data = {}
    for chr_id in data['chromosome'].unique():
        one_chr_data = data.loc[data['chromosome'] == chr_id]
        for start in one_chr_data['start'].unique():
            one_chr_start_data = one_chr_data.loc[one_chr_data['start'] == start]
            for end in one_chr_start_data['end'].unique():
                one_chr_start_end_data = one_chr_start_data.loc[one_chr_start_data['end'] == end]

                region_id = f'{chr_id}:{start}-{end}'

                reads = one_chr_start_end_data['methylation_pattern']
                counts = one_chr_start_end_data['methylation_pattern_count']
                split_reads: DataFrame = reads.str.split(r'', expand=True).iloc[:, 1:-1]

                sample_data[region_id] = (split_reads, counts)

    return sample_data

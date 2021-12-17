from argparse import ArgumentParser, ONE_OR_MORE
from pathlib import Path

from preprocess.tnc_encode import tnc_encode_dirs

parser = ArgumentParser()
parser.add_argument('--pattern_dirs', type=Path, nargs=ONE_OR_MORE, required=True)
parser.add_argument('--tnc_dir', type=Path, default=Path('/Data/tnc_encode'))
parser.add_argument('--n_threads', type=int, default=4)

args = parser.parse_args()

tnc_encode_dirs(pattern_dirs=args.pattern_dirs, tnc_dir=args.tnc_dir, n_threads=args.n_threads)

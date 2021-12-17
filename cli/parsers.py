from argparse import ArgumentParser, ONE_OR_MORE
from pathlib import Path

from preprocess import preprocess_from_args


def path_type(p: str) -> Path:
    return Path(p)


def preprocess_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument('--pattern_files', type=path_type, nargs=ONE_OR_MORE, required=True)

    parser.set_defaults(func=preprocess_from_args)
    return parser

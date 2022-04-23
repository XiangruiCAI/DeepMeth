from argparse import ArgumentParser
from pathlib import Path


def preprocess_parser():
    parser = ArgumentParser()
    parser.add_argument('--pattern_dir', type=lambda p: Path(p), required=True)

    pass

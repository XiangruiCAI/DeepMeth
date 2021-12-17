import shutil
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()


def path_type(p: str) -> Path:
    return Path(p)


def extract(experiment_dir: Path, out_dir: Path):
    experiment_name = experiment_dir.name
    target_dir = out_dir.joinpath(experiment_name + '_deploy')
    for i in range(1, 11):
        shutil.copytree(
            src=str(experiment_dir.joinpath(f'{i}', 'saved_models')),
            dst=str(target_dir.joinpath(f'{i}', 'saved_models'))
        )
        shutil.copy(
            src=str(experiment_dir.joinpath(f'{i}', 'regions.tsv')),
            dst=str(target_dir.joinpath(f'{i}', 'regions.tsv'))
        )


parser.add_argument('--experiment_dir', type=path_type, required=True)
parser.add_argument('--out_dir', type=path_type, default=Path.home())

args = parser.parse_args()
extract(args.experiment_dir, args.out_dir)

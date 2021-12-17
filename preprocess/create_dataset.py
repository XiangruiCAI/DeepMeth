from pathlib import Path


def create_dataset(experiment_dir: Path, tnc_dir: Path, overwrite: bool = False):
    dataset_dir = experiment_dir.joinpath('dataset')
    if dataset_dir.is_symlink():
        if overwrite:
            dataset_dir.unlink()
        else:
            return
    dataset_dir.symlink_to(tnc_dir, target_is_directory=True)

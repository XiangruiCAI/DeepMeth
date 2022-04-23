from pathlib import Path

from run import prepare, auto_encoder_train, auto_encoder_encode, lightgbm_train, lightgbm_test, load_config, \
    create_parser, lightgbm_independent_test, lightgbm_31gps, lightgbm_44gps, lightgbm_57gps, lightgbm_75gps

sub_commands = {
    'prepare': prepare,
    'auto_encoder_train': auto_encoder_train,
    'auto_encoder_encode': auto_encoder_encode,
    'lightgbm_train': lightgbm_train,
    'lightgbm_test': lightgbm_test,
    'lightgbm_independent_test': lightgbm_independent_test,
    'lightgbm_31gps': lightgbm_31gps,
    'lightgbm_44gps': lightgbm_44gps,
    'lightgbm_57gps': lightgbm_57gps,
    'lightgbm_75gps': lightgbm_75gps
}

parser = create_parser(sub_commands)

args = parser.parse_args()

experiment_dir = Path(args.experiment_dir)
n_splits = args.n_splits

config = load_config(experiment_dir.joinpath('config.yml'), args.command)

args.func(experiment_dir=experiment_dir, n_splits=n_splits, **config)

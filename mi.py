from pathlib import Path

from run import lightgbm_train, lightgbm_test


def create_symbol(encode_dir: Path, input_dir: Path):
    encode_dir.mkdir(exist_ok=True)
    for sample_dir in input_dir.iterdir():
        sample_id = sample_dir.name
        encode_dir.joinpath(sample_id).with_suffix('.txt') \
            .symlink_to(sample_dir.joinpath(sample_id).with_suffix('.txt'))


if __name__ == '__main__':
    # create_splits_by_single_file(
    #     experiment_dir=Path('/home/wangshichao/share'),
    #     splits_file=Path('/ehpcdata/analysis/LC015_pre529_poolA_poolB_poolC_10_splits_matrix.txt'),
    #     info_file=Path('/home/wangshichao/share/patho_info.txt'),
    #     categories={'train': 'train', 'test': 'test',
    #                 'independent_test': 'independent_test'},
    #     over_write=True
    # )
    # create_symbol(
    #     Path('/home/wangshichao/share/3/encode'),
    #     Path('/ehpcdata/analysis/LC_new_panel_all_pattern_20191223_anchordx/pre529/AE_top200/model3'))
    clinical_score = Path('/home/wangshichao/share/model_deploy/clinical_score.txt')

    lightgbm_train(
        experiment_dir=Path('/home/wangshichao/share'),
        n_splits=[3, 4],
        external_files=[
            '/home/wangshichao/share/model_deploy/comethylation_percentage_matrix_2_3_noX.txt',
            '/home/wangshichao/share/model_deploy/comethylation_percentage_matrix_3_5_noX.txt',
            clinical_score,
        ],
        h_param={},
    )

    lightgbm_test(experiment_dir=Path('/home/wangshichao/share'),
                  n_splits=[3, 4],
                  test_splits=['independent_test'],
                  external_files=[
                      '/home/wangshichao/share/model_deploy/comethylation_percentage_matrix_2_3_noX.txt',
                      '/home/wangshichao/share/model_deploy/comethylation_percentage_matrix_3_5_noX.txt',
                      clinical_score
                  ])

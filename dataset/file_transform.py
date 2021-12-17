import os
import pandas as pd
import numpy as np
from utils import get_selected_region, get_samples_label


def create_dataset_symlink(source_dirs, dst_dir, **kwargs):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for source_dir in source_dirs:
        for root, directory, files in os.walk(source_dir):
            for direct in directory:
                src = root + '/' + direct
                dst = dst_dir + '/' + direct
                os.symlink(src, dst)
                print(src, dst)
            break
    return 1


def split_transform(predefined_split, sample_info_list, split_id, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    samples_label = get_samples_label(sample_info_list)
    data = pd.read_csv(predefined_split, sep='\t')
    np_data = np.array(data)
    train_idx = np.where(np_data[:, int(split_id)] == 'train')
    test_idx = np.where(np_data[:, int(split_id)] == 'test')
    train = np_data[train_idx, 0]
    test = np_data[test_idx, 0]
    train_label = np.array([samples_label[sample] for sample in train[0]]).reshape(1, -1)
    test_label = np.array([samples_label[sample] for sample in test[0]]).reshape(1, -1)
    pd_train = pd.DataFrame(np.concatenate((train, train_label)).T, columns=['sample_id', 'label'])
    pd_test = pd.DataFrame(np.concatenate((test, test_label)).T, columns=['sample_id', 'label'])
    pd_train.to_csv(path_or_buf=out_dir + '/train.csv', index=False)
    pd_test.to_csv(path_or_buf=out_dir + '/test.csv', index=False)


def region_transform(region_importance_path, out_dir, all_region_file=None):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    region_list = get_selected_region(sorted_region_file=region_importance_path, all_region_file=all_region_file)
    pd_regions = pd.DataFrame(np.array(region_list).reshape(-1, 1), columns=['Region'])
    pd_regions.to_csv(path_or_buf=out_dir + '/region.csv', index=False)

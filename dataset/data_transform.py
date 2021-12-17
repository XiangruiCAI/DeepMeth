import re
import numpy as np
import pandas as pd
import csv
import time
import os

from tqdm import tqdm

from utils import get_sample_id


def tnc_to_code(tnc_matrix, **kwargs):
    if 'code_dict' in kwargs:
        code_dict = kwargs['code_dict']
    else:
        code_dict = {'T': '1', 'N': '2', 'C': '3'}
    res = np.array(tnc_matrix)
    h = res.shape[0]
    if h == 0:
        return 0, None
    for key, value in code_dict.items():
        res = np.char.replace(res, key, value)
    res = res.astype(np.int)
    return 1, res


def transform_single_sample(sample_path, out_dir):
    sample_file = open(sample_path)
    sample_reader = csv.reader(sample_file, delimiter='\t')
    next(sample_reader)
    cur_region = ''
    cur_region_pattern = list()
    regions_size = list()
    for line in sample_reader:
        s = '_'.join(line[0: 4])
        if s != cur_region:
            if cur_region != '':
                region_saving_path = out_dir + '/' + str(cur_region) + '.txt'
                flag, code_mat = tnc_to_code(cur_region_pattern)
                if flag:
                    try:
                        np.savetxt(region_saving_path, code_mat, fmt='%d')
                        regions_size.append([cur_region, code_mat.shape[0], code_mat.shape[1]])
                    except Exception as e:
                        print("Error: ", e)
                else:
                    print(sample_path, ' Transform Failed! ')
            cur_region = s
            cur_region_pattern = list()
            pattern_count = int(line[6])
            pattern_list = list(line[5])
            for i in range(pattern_count):
                cur_region_pattern.append(pattern_list)
        else:
            pattern_count = int(line[6])
            pattern_list = list(line[5])
            for i in range(pattern_count):
                cur_region_pattern.append(pattern_list)
    region_saving_path = out_dir + '/' + str(cur_region) + '.txt'
    flag, code_mat = tnc_to_code(cur_region_pattern)
    if flag:
        try:
            np.savetxt(region_saving_path, code_mat, fmt='%d')
            regions_size.append([cur_region, code_mat.shape[0], code_mat.shape[1]])
        except Exception as e:
            print("Error: ", e)
    else:
        print(sample_path, ' Transform Failed! ')
    sample_file.close()
    region_size_path = out_dir + '/region_size.csv'
    pd_region_size = pd.DataFrame(regions_size, columns=["region", "height", "width"])
    pd_region_size.to_csv(path_or_buf=region_size_path, index=False)


def transform_multi_sample(file_list, out_dir):
    for i in tqdm(range(len(file_list))):
        pattern_file_path = file_list[i]
        sample_id = get_sample_id(pattern_file_path.split('/')[-1])
        print(i + 1, " Start Transform Sample : ", sample_id)
        trans_out = out_dir + '/' + str(sample_id)
        if not os.path.exists(trans_out):
            os.makedirs(trans_out)
        transform_single_sample(pattern_file_path, trans_out)
        print(i, "  Finished Transform sample : ", sample_id, " | Time : ",
              time.strftime('%m-%d %H:%M:%S', time.localtime(time.time())))
    return 1

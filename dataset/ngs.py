from pathlib import Path
from typing import List

import numpy
import numpy as np
import pandas
from torch.utils.data import Dataset
import torch
import pandas as pd
import os

import utils


class TrainDataset(Dataset):

    def __init__(self, tnc_dir: Path, samples: List[str], selected_regions, **kwargs):
        self.region_height = kwargs.get('region_height', 1000)
        self.region_height_min = kwargs.get('region_height_min', 0)
        self.region_width = kwargs.get('region_width', 40)
        self.region_width_min = kwargs.get('region_width_min', 3)
        self.noise_factor = kwargs.get('noise_factor', 0)

        datalist = list()
        for sample in samples:
            csv_path = tnc_dir.joinpath(sample).joinpath('region_sizes.tsv')
            region_size = pd.read_csv(csv_path, sep='\t')
            satisfied_regions = region_size['region_id'][(region_size.height < self.region_height) &
                                                         (region_size.height > self.region_height_min) &
                                                         (region_size.width < self.region_width) &
                                                         (region_size.width > self.region_width_min)]
            used_regions = np.intersect1d(np.array(selected_regions), np.array(satisfied_regions))
            for region in used_regions:
                data_path = tnc_dir.joinpath(sample, str(region)).with_suffix('.tsv')
                datalist.append(data_path)
        self.datalist = datalist

    def __getitem__(self, index):
        path = self.datalist[index]
        data_np = pandas.read_csv(path, sep='\t', header=None).values

        input_data = data_np
        output_data = data_np
        if self.noise_factor != 0:
            data_np = data_np / 3
            data_np_noisy = data_np + self.noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data_np.shape)
            input_data = np.clip(data_np_noisy, 0., 1.)
            output_data = data_np

        if len(input_data.shape) == 1:
            if self.region_height_min == 0 and self.region_width_min > 0:
                input_data = input_data.reshape(1, -1)
            if self.region_height_min > 0 and self.region_width_min == 0:
                input_data = input_data.reshape(-1, 1)
        height, width = input_data.shape
        temp_input = np.reshape(input_data, (1, 1, height, width))
        temp_output = np.reshape(output_data, (1, 1, height, width))
        tensor_input = torch.from_numpy(temp_input)
        tensor_output = torch.from_numpy(temp_output)

        width = min(self.region_width, width)
        height = min(self.region_height, height)

        mask = torch.zeros(1, self.region_height, self.region_width)

        align_data_input = torch.zeros(1, self.region_height, self.region_width)
        align_data_output = torch.zeros(1, self.region_height, self.region_width)

        align_data_input[0, : height, :width] = tensor_input[0, 0, : height, : width]
        align_data_output[0, : height, :width] = tensor_output[0, 0, : height, : width]

        mask[0, : height, : width] = 1
        return align_data_input, align_data_output, mask

    def __len__(self):
        return len(self.datalist)


class EncodeDataset(Dataset):

    def __init__(self, sample_dir: Path, selected_regions, order_dict, **kwargs):
        self.region_height = kwargs.get('region_height', 1000)
        self.region_height_min = kwargs.get('region_height_min', 0)
        self.region_width = kwargs.get('region_width', 40)
        self.region_width_min = kwargs.get('region_width_min', 3)
        self.noise_factor = kwargs.get('noise_factor', 0)

        datalist = list()
        csv_path = sample_dir.joinpath('region_sizes.tsv')
        region_size = pd.read_csv(csv_path, sep='\t')
        satisfied_regions = region_size['region_id'][(region_size.height < self.region_height) &
                                                     (region_size.height > self.region_height_min) &
                                                     (region_size.width < self.region_width) &
                                                     (region_size.width > self.region_width_min)]
        used_regions = np.intersect1d(np.array(selected_regions), np.array(satisfied_regions))
        for region in used_regions:
            data_path = str(sample_dir) + '/' + str(region) + '.tsv'
            datalist.append(data_path)
        self.datalist = datalist
        self.order_dict = order_dict

    def __getitem__(self, index):
        path = self.datalist[index]
        region = os.path.basename(path).split('.')[0]
        region_id = self.order_dict[region]

        data_np = pandas.read_csv(path, sep='\t', header=None).values

        input_data = data_np
        output_data = data_np
        if self.noise_factor != 0:
            data_np = data_np / 3
            data_np_noisy = data_np + self.noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data_np.shape)
            input_data = np.clip(data_np_noisy, 0., 1.)
            output_data = data_np

        if len(input_data.shape) == 1:
            if self.region_height_min == 0 and self.region_width_min > 0:
                input_data = input_data.reshape(1, -1)
            if self.region_height_min > 0 and self.region_width_min == 0:
                input_data = input_data.reshape(-1, 1)

        height, width = input_data.shape
        temp_input = np.reshape(input_data, (1, 1, height, width))
        temp_output = np.reshape(output_data, (1, 1, height, width))
        tensor_input = torch.from_numpy(temp_input)
        tensor_output = torch.from_numpy(temp_output)

        width = min(self.region_width, width)
        height = min(self.region_height, height)

        mask = torch.zeros(1, self.region_height, self.region_width)

        align_data_input = torch.zeros(1, self.region_height, self.region_width)
        align_data_output = torch.zeros(1, self.region_height, self.region_width)

        align_data_input[0, : height, :width] = tensor_input[0, 0, : height, : width]
        align_data_output[0, : height, :width] = tensor_output[0, 0, : height, : width]

        mask[0, : height, : width] = 1

        return align_data_input, align_data_output, mask, region_id

    def __len__(self):
        return len(self.datalist)


class SampleRegionDataset(Dataset):

    def __init__(self,
                 sample_dir: Path,
                 regions: List[str],
                 *,
                 min_height: int = 0,
                 max_height: int = 1000,
                 min_width: int = 3,
                 max_width: int = 40
                 ):
        self.sample_dir = sample_dir

        self.min_height = min_height
        self.max_height = max_height
        self.min_width = min_width
        self.max_width = max_width

        self.regions = self.filter_regions(regions)

    def __len__(self):
        return len(self.regions)

    def filter_regions(self, regions: List[str]) -> List[str]:
        region_sizes = pandas.read_csv(self.sample_dir.joinpath('region_sizes.tsv'), sep='\t')
        satisfied_regions = region_sizes['region_id'][(region_sizes.height < self.max_height) &
                                                      (region_sizes.height > self.min_height) &
                                                      (region_sizes.width < self.max_width) &
                                                      (region_sizes.width > self.min_width)]

        return list(set.difference(set(regions), satisfied_regions))

        pass

    def __getitem__(self, item):
        region = self.regions[item]
        target_file = self.sample_dir.joinpath(region)
        region_data: np.ndarray = utils.read_sample_region(target_file)

        width, height = region_data.shape
        mask = numpy.zeros(self.max_width, self.max_height)
        mask[: width, : height] = 1

        return region_data, mask

        pass

import logging
from typing import Dict, Any

import numpy
from imblearn import over_sampling
from imblearn.over_sampling.base import BaseOverSampler
from numpy import ndarray


def over_sample(over_sample_param: Dict[str, Any], x: ndarray, y: ndarray):
    sampler_str = over_sample_param.pop('sampler')

    logging.info(f'running over sampler with {sampler_str}')
    try:
        over_sampler: BaseOverSampler = getattr(over_sampling, sampler_str)(**over_sample_param)
    except AttributeError as e:
        msg = f'support {over_sampling.__all__}'
        raise AttributeError(e, msg)

    resampled_x, resampled_y = over_sampler.fit_resample(numpy.nan_to_num(x), y)
    return resampled_x, resampled_y

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from dataset.ngs import EncodeDataset
from models.encoder import Encoder


def encode(tnc_dir: Path,
           samples,
           selected_regions,
           model_file: Path,
           encode_dir: Path,
           device: str,
           **kwargs):
    encode_batch = kwargs.pop('encode_batch', 32)
    encode_dim = kwargs.get('encode_dim', 50)
    
    print("Encode params:", kwargs)
    model = Encoder(**kwargs)
    model.encoder.load_state_dict(
        torch.load(model_file, map_location=lambda storage, loc: storage)
    )
    logging.info(device)
    model.to(device)
    logging.info("Model build success.")

    order_dict = utils.get_region_dict(selected_regions)
    samples_progress = tqdm(samples)
    for encode_sample in samples_progress:
        sample_dir = tnc_dir.joinpath(str(encode_sample))
        sample_data = EncodeDataset(sample_dir=sample_dir, selected_regions=selected_regions, order_dict=order_dict,
                                    **kwargs)
        sample_data_loader = Data.DataLoader(dataset=sample_data, batch_size=encode_batch, num_workers=8)
        final_code = np.zeros((len(order_dict), encode_dim))

        start = time.time()
        for batch_idx, (input_x, output_x, mask, region_id) in enumerate(sample_data_loader):
            input_x_var = input_x.to(device=device, dtype=torch.float)
            code = model(input_x_var)

            temp_code = code.data.cpu().numpy()
            region_id = np.array(list(region_id)).astype(np.int)
            final_code[region_id, :] = temp_code

        save_path = encode_dir.joinpath(encode_sample).with_suffix('.txt')
        try:
            np.savetxt(save_path, final_code)
        except Exception as e:
            logging.error("Error: ", e)
        end = time.time()
        samples_progress.write("ID=%s | Encode using %.4f" % (encode_sample, (end - start)))
    logging.info("Encode Done!")
    return 1

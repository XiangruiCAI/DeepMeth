import time
from pathlib import Path
from typing import List, Union

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch import optim
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from dataset.ngs import TrainDataset
from models.auto_encoder import AutoEncoder
from utils import init_seed

SHUFFLE_SEED = 1234


def ae_train(data_dir: Path, train_samples: List[str], selected_regions, serialization_dir: Path, params):
    ae_param = params['auto_encoder']
    print(ae_param)
    train_batch = ae_param.get('train_batch', 32)
    train_epoch = ae_param.get('train_epoch', 1)
    ae_lr = ae_param.get('lr', 1e-4)
    ae_weight_decay = ae_param.get('weight_decay', 1e-5)

    ae_summary_dir = serialization_dir.joinpath('loss')
    ae_summary_dir.mkdir(exist_ok=True)

    model = AutoEncoder(**ae_param)

    if torch.cuda.is_available():
        model = model.cuda()

    print("Model building success.")

    init_seed(SHUFFLE_SEED)

    ngs = TrainDataset(tnc_dir=data_dir, samples=train_samples, selected_regions=selected_regions, **ae_param)
    encoder_data_loader = DataLoader(dataset=ngs, batch_size=train_batch, shuffle=True, num_workers=8)
    print("Load data success.")

    # loss function
    loss_func = nn.MSELoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=ae_lr, weight_decay=ae_weight_decay)

    # tensorboardX
    writer = SummaryWriter(str(ae_summary_dir))

    saving_path = None

    # train
    for epoch in range(train_epoch):
        for batch_idx, (input_x, output_x, mask) in tqdm(enumerate(encoder_data_loader),
                                                         total=len(encoder_data_loader)):
            input_x_var = Variable(input_x).cuda()
            input_x_var = input_x_var.float()
            output_x_var = Variable(output_x).cuda()
            output_x_var = output_x_var.float()
            mask_var = Variable(mask).cuda()
            mask_var = mask_var.float()

            code, x_generate = model(input_x_var)
            x_generate_temp = torch.mul(x_generate, mask_var)

            loss = loss_func(output_x_var, x_generate_temp)

            del code, x_generate, x_generate_temp
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            niter = epoch * len(encoder_data_loader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), niter)
            tqdm.write(f'EPOCH : {epoch} | '
                       f'Batch : {batch_idx} | '
                       f'Train loss : {loss.item():.4f} | '
                       f'Time : {time.strftime("%m-%d %H:%M:%S", time.localtime(time.time()))}')

        saving_path = serialization_dir.joinpath('saved_models')
        Path(saving_path).mkdir(exist_ok=True)
        saving_name = 'epoch_' + str(epoch) + '.pkl'
        torch.save(model.encoder.state_dict(), saving_path.joinpath('encoder_' + saving_name))
        torch.save(model.decoder.state_dict(), saving_path.joinpath('decoder_' + saving_name))
    print("Model Training Finished ! ")
    torch.cuda.empty_cache()
    return 1, saving_path


def _train(
        model: nn.Module,
        data_loader: DataLoader,
        criterion: _Loss,
        optimizer: Optimizer,
        max_epoch: int,
        serialization_dir: Path,
        device: Union[str, torch.device]
        ):
    model = model.to(device)
    summary_dir = serialization_dir.joinpath('loss')
    summary_dir.mkdir(exist_ok=True)
    summary_writer = SummaryWriter(str(summary_dir))

    for epoch in range(max_epoch):
        input_x: torch.Tensor
        for batch_idx, (input_x, output_x, mask) in tqdm(enumerate(data_loader),
                                                         total=len(data_loader)):
            input_x_var = input_x.to(device=device, dtype=torch.float)
            output_x_var = output_x.to(device=device, dtype=torch.float)
            mask_var = mask.to(device=device, dtype=torch.float)

            code, x_generate = model(input_x_var)
            x_generate_temp = torch.mul(x_generate, mask_var)

            loss = criterion(output_x_var, x_generate_temp)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            niter = epoch * len(data_loader) + batch_idx
            summary_writer.add_scalar('Train/Loss', loss.item(), niter)
            tqdm.write(f'EPOCH : {epoch} | '
                       f'Batch : {batch_idx} | '
                       f'Train loss : {loss.item():.4f} | '
                       f'Time : {time.strftime("%m-%d %H:%M:%S", time.localtime(time.time()))}')

        saving_path = serialization_dir.joinpath('saved_models')
        Path(saving_path).mkdir(exist_ok=True)
        saving_name = 'epoch_' + str(epoch) + '.pkl'
        torch.save(model.encoder.state_dict(), saving_path.joinpath('encoder_' + saving_name))
        torch.save(model.decoder.state_dict(), saving_path.joinpath('decoder_' + saving_name))
    print("Model Training Finished ! ")


def train_auto_encoder(tnc_dir: Path, samples_file: Path, regions_file: Path, serialization_dir: Path, device, **kwargs):
    ae_param = kwargs.pop('model', {}).pop('auto_encoder', {})
    model = AutoEncoder(**ae_param)
    samples = utils.read_split_file(samples_file).index.tolist()
    selected_regions = utils.read_sample_region(regions_file)
    data_set = TrainDataset(tnc_dir=tnc_dir, samples=samples, selected_regions=selected_regions)

    optimizer_param = ae_param.pop('optimizer', {})
    _train(model=model,
           data_loader=DataLoader(data_set, batch_size=ae_param.pop('batch', 32), shuffle=True, num_workers=8),
           criterion=nn.MSELoss(),
           optimizer=optim.Adam(AutoEncoder.parameters(model),
                                lr=optimizer_param.pop('learning_rate', 1e-4),
                                weight_decay=optimizer_param.pop('weight_decay', 1e-5)),
           max_epoch=ae_param.pop('max_epoch', 1),
           serialization_dir=serialization_dir,
           device=device)
    pass

import sys

sys.path.append('/path/to/TVTS/v2')

import argparse
import collections
import torch
import data_loader.data_loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model_dist_TVTSv2_ViT_H_14 as module_arch
import utils.visualizer as module_vis
from utils.util import replace_nested_dict_item
from parse_config_dist_multi import ConfigParser
from trainer import Trainer_TVTSv2_H_14
from sacred import Experiment
from neptunecontrib.monitoring.sacred import NeptuneObserver
import transformers
import os
import torch.multiprocessing
import OpenCLIP

ex = Experiment('train')


@ex.main
def run():
    logger = config.get_logger('train')
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    os.environ['TRANSFORMERS_OFFLINE'] = "1"
    # TODO: improve Create identity (do nothing) visualiser?
    if config['visualizer']['type'] != "":
        visualizer = config.initialize(
            name='visualizer',
            module=module_vis,
            exp_name=config['name'],
            web_dir=config._web_log_dir
        )
    else:
        visualizer = None

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='tcp://{}:{}'.format(
                                             args.master_address, args.master_port),
                                         rank=args.rank, world_size=args.world_size)
    device = torch.device(f'cuda:{args.local_rank}')
    print('world_size', args.world_size, flush=True)
    print('local_rank: ', args.local_rank, flush=True)

    tokenizer = OpenCLIP.get_tokenizer('ViT-H-14')

    # setup data_loader instances
    data_loader, valid_data_loader = init_dataloaders(config, module_data)

    print('Train dataset: ', [x.n_samples for x in data_loader], ' samples')
    print('Val dataset: ', [x.n_samples for x in valid_data_loader], ' samples')

    # build model architecture, then print to console
    model = config.initialize('arch', module_arch)
    if args.local_rank == 0:
        logger.info(model)

    # get function handles of loss and metrics
    loss = config.initialize(name="loss", module=module_loss)
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # exclude all bias and LayerNorm parameters from weight decay
    no_decay_names = ['bias', 'LayerNorm', 'ln_', 'norm', 'ls_', 'LayerScale']

    text_tune_layers = ['resblocks.%d.' % i for i in range(18, 24)]

    decay_clip_params, no_decay_clip_params = [], []
    decay_new_params, no_decay_new_params = [], []

    for name, param in model.named_parameters():
        # CLIP visual branch
        if 'video_model' in name:
            if 'timeattn' in name or 'ln_3' in name or 'ls_3' in name:
                if any(nd in name for nd in no_decay_names):
                    no_decay_new_params.append((name, param))
                else:
                    decay_new_params.append((name, param))
            else:
                if any(nd in name for nd in no_decay_names):
                    no_decay_clip_params.append((name, param))
                else:
                    decay_clip_params.append((name, param))
        # CLIP text branch
        elif 'text' in name:
            if 'resblocks' in name:
                if any(tl in name for tl in text_tune_layers):
                    if any(nd in name for nd in no_decay_names):
                        no_decay_clip_params.append((name, param))
                    else:
                        decay_clip_params.append((name, param))
                else:
                    param.requires_grad = False
            else:
                if any(nd in name for nd in no_decay_names):
                    no_decay_clip_params.append((name, param))
                else:
                    decay_clip_params.append((name, param))
        # other parameters
        else:
            if any(nd in name for nd in no_decay_names):
                no_decay_new_params.append((name, param))
            else:
                decay_new_params.append((name, param))

    # for n, p in decay_new_params:
    #     print('decay_new_params: ', n)
    # for n, p in no_decay_new_params:
    #     print('no_decay_new_params: ', n)
    # for n, p in decay_clip_params:
    #     print('decay_clip_params: ', n)
    # for n, p in no_decay_clip_params:
    #     print('no_decay_clip_params: ', n)
    # exit(0)

    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_new_params], 'weight_decay': 0.05, 'lr': 1e-4},
        {'params': [p for n, p in no_decay_new_params], 'weight_decay': 0, 'lr': 1e-4},
        {'params': [p for n, p in decay_clip_params], 'weight_decay': 0.05, 'lr': 1e-7},
        {'params': [p for n, p in no_decay_clip_params], 'weight_decay': 0, 'lr': 1e-7}
    ]
    optimizer = transformers.AdamW(params=optimizer_grouped_parameters)

    lr_scheduler = None
    if 'lr_scheduler' in config._config:
        if hasattr(transformers, config._config['lr_scheduler']['type']):
            lr_scheduler = config.initialize('lr_scheduler', transformers, optimizer)
        else:
            print('lr scheduler not found')
    if config['trainer']['neptune']:
        writer = ex
    else:
        writer = None
    trainer = Trainer_TVTSv2_H_14(args, model, loss, metrics, optimizer,
                                  config=config,
                                  data_loader=data_loader,
                                  valid_data_loader=valid_data_loader,
                                  lr_scheduler=lr_scheduler,
                                  visualizer=visualizer,
                                  writer=writer,
                                  tokenizer=tokenizer,
                                  max_samples_per_epoch=config['trainer'][
                                      'max_samples_per_epoch'])
    trainer.train()


def init_dataloaders(config, module_data):
    """
    We need a way to change split from 'train' to 'val'.
    """
    if "type" in config["data_loader"] and "args" in config["data_loader"]:
        # then its a single dataloader
        data_loader = [config.initialize("data_loader", module_data)]
        config['data_loader']['args'] = replace_nested_dict_item(config['data_loader']['args'], 'split', 'val')
        valid_data_loader = [config.initialize("data_loader", module_data)]
    elif isinstance(config["data_loader"], list):
        data_loader = [config.initialize('data_loader', module_data, index=idx) for idx in
                       range(len(config['data_loader']))]
        new_cfg_li = []
        for dl_cfg in config['data_loader']:
            dl_cfg['args'] = replace_nested_dict_item(dl_cfg['args'], 'split', 'val')
            new_cfg_li.append(dl_cfg)
        config._config['data_loader'] = new_cfg_li
        valid_data_loader = [config.initialize('data_loader', module_data, index=idx) for idx in
                             range(len(config['data_loader']))]
    else:
        raise ValueError("Check data_loader config, not correct format.")

    return data_loader, valid_data_loader


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-o', '--observe', action='store_true',
                      help='Whether to observe (neptune)')
    args.add_argument('-l', '--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')

    master_address = os.environ['MASTER_ADDR']
    master_port = int(os.environ['MASTER_PORT'])
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])

    args.add_argument('-ma', '--master_address', default=master_address)
    args.add_argument('-mp', '--master_port', type=int, default=master_port)
    args.add_argument('-ws', '--world_size', type=int, default=world_size)
    args.add_argument('-rk', '--rank', type=int, default=rank)
    args.add_argument('-k', '--local_rank', type=int, default=local_rank)
    args.add_argument('-sc', '--schedule', nargs='+', type=int, default=[6, 8])

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['-lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['-bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size')),
    ]
    config = ConfigParser(args, options)
    args = args.parse_args()
    ex.add_config(config._config)

    args.local_rank = int(os.environ['LOCAL_RANK'])

    if config['trainer']['neptune']:
        # delete this error if you have added your own neptune credentials neptune.ai
        raise ValueError('Neptune credentials not set up yet.')
        ex.observers.append(NeptuneObserver(
            api_token='INSERT TOKEN',
            project_name='INSERT PROJECT NAME'))
        ex.run()
    else:
        run()

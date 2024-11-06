import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn as nn
import torch
import pdb
import logging
import copy
import argparse
import time
import random
import resource
import hydra
from hydra import initialize, compose
from omegaconf import OmegaConf
from logging.handlers import RotatingFileHandler
from torch.utils.tensorboard import SummaryWriter
from tokenizers import Tokenizer
from util import *
from models import *
from train import *
from data import *

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def worker(gpu, cfg):
    # Setting up gpu and rank within DDP
    rank = cfg.distributed.node_rank * cfg.distributed.gpus + gpu
    torch.cuda.set_device(gpu)
    device = torch.device("cuda")

    # Logger init
    logger = None
    if rank == 0 or not cfg.distributed.ddp:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        rfh = RotatingFileHandler(cfg.paths.logging_file, maxBytes=100000, backupCount=10, encoding="UTF-8")
        logger.addHandler(rfh)
    if cfg.distributed.ddp:
        dist.init_process_group(backend='nccl', init_method='env://', world_size=cfg.distributed.world_size, rank=rank)
        #dist.init_process_group(backend='gloo', init_method='file://'+args.sync_path, world_size=args.world_size, rank=rank)

    # Set the random seed manually for reproducibility.
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # Data init
    data = ASRDataset(cfg)
    data_val = None
    
    # Build tokenizers
    tokenizer = Tokenizer.from_file(cfg.paths.tokenizer_path)
    cfg.model.vocab_size = tokenizer.get_vocab_size()
    if cfg.model_name == 'hc_ctc':
        inter_tokenizers = []
        cfg.model.inter_vocab_size = []
        for p in cfg.paths.inter_tokenizer_paths:
            tok = Tokenizer.from_file(p)
            inter_tokenizers.append(tok)
            cfg.model.inter_vocab_size.append(tok.get_vocab_size())

    # Load checkpoint 
    checkpoint = None
    if os.path.isfile(cfg.paths.ckpt_path):
        print(f'Loading checkpoint')
        checkpoint = torch.load(cfg.paths.ckpt_path, map_location=f'cuda:{gpu}')
        cfg.trainer.epochs_done = checkpoint['epochs_done']
        cfg.trainer.iterations_done = checkpoint['iterations_done']

    # Loading models
    print(f'Loading model.')
    if cfg.model_name == 'ctc':
        model = baseCTC.from_pretrained(cfg, cfg.paths.ckpt_path, f'cuda:{gpu}') if checkpoint else baseCTC(cfg).to(device)
    elif cfg.model_name == 'sc_ctc':
        model = baseSCCTC.from_pretrained(cfg, cfg.paths.ckpt_path, f'cuda:{gpu}') if checkpoint else baseSCCTC(cfg).to(device)
    elif cfg.model_name == 'hc_ctc':
        model = baseHCCTC.from_pretrained(cfg, cfg.paths.ckpt_path, f'cuda:{gpu}') if checkpoint else baseHCCTC(cfg).to(device)
    else:
        raise ValueError(f"Model '{cfg.model_name}' is invalid. Current implementation includes: 'ctc', 'sc_ctc' and 'hc_ctc'")
    print(f'# model parameters = {count_parameters(model)/1e6}M')

    # Load optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg.trainer.lr, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
    if cfg.trainer.load_opt:
        if not checkpoint:
            raise ValueError(f"'load_opt' was true but checkpoint was not provided")
        print(f'Loading optimizer from checkpoint')
        optimizer.load_state_dict(checkpoint['optimizer'])

    sampler = None
    # DDP wrapping 
    if cfg.distributed.ddp:
        model = nn.parallel.DistributedDataParallel(model)
        sampler = torch.utils.data.distributed.DistributedSampler(data, num_replicas=cfg.distributed.world_size, rank=rank)

    # Trainer init
    writer = None
    if rank == 0 or not cfg.distributed.ddp:
        writer = SummaryWriter(log_dir=cfg.paths.summary_path)
    if cfg.model_name=='ctc':
        trainer = Trainer(cfg, tokenizer, model, data, device, optimizer, sampler=sampler, rank=rank, checkpoint=checkpoint, data_val=data_val, logger=logger, writer=writer)
    elif cfg.model_name=='sc_ctc':
        trainer = TrainerSC(cfg, tokenizer, model, data, device, optimizer, sampler=sampler, rank=rank, checkpoint=checkpoint, data_val=data_val, logger=logger, writer=writer)
    elif cfg.model_name=='hc_ctc':
        trainer = TrainerHC(cfg, tokenizer, inter_tokenizers, model, data, device, optimizer, sampler=sampler, rank=rank, checkpoint=checkpoint, data_val=data_val, logger=logger, writer=writer)

    # Training starts here
    print(f'Starting training ...')
    trainer.train()
    print(f'Done')

def main(cfg):
    cfg.distributed.world_size = cfg.distributed.gpus * cfg.distributed.nnodes
    cfg.trainer.lr = cfg.trainer.base_lr * ((cfg.trainer.batch_size / cfg.trainer.base_batch_size)**0.5)
    if cfg.distributed.world_size > 1:
        cfg.distributed.ddp = True
    else:
        cfg.distributed.ddp = False

    if cfg.distributed.ddp:
        local_rank = int(os.environ['LOCAL_RANK'])
        worker(local_rank, cfg)
    else:
        worker(cfg.distributed.gpu_num, cfg)

if __name__ == "__main__":
    # Parse config_path and config_name from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--config-path", type=str, default=".", help="Path to the config directory")
    parser.add_argument("--config-name", type=str, default="config", help="Name of the config file (without .yaml extension)")
    args = parser.parse_args()

    # Use the provided config_path and config_name
    with initialize(config_path=args.config_path):
        cfg = compose(config_name=args.config_name)
        cfg.distributed.node_rank = args.node_rank
        main(cfg)

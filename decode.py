from util import *
from models import *
from train import *
from data import *
from logging.handlers import RotatingFileHandler
from tokenizers import Tokenizer
from speechbrain.processing.features import InputNormalization
from hydra import initialize, compose
from omegaconf import OmegaConf
import hydra
import torch.nn as nn
import torch
import pdb
import logging
import copy
import argparse
import time
import random
import sys
import resource
import torch.distributed as dist
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.df = pd.read_csv(cfg.paths.test_path)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        if self.cfg.corpus == 'librispeech':
            path = row.audio_file.replace('/data/corpora2/librispeech/LibriSpeech/',
                                          '/research/nfs_fosler_1/vishal/audio/libri/')
        else:
            path = row.audio_file
        ground_truth = clean4asr(row.utterance)
        return path, ground_truth

class Collator:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, lst):
        path_, ground_truth_ = zip(*lst)
        return list(path_), list(ground_truth_)

def main(cfg):
    rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=cfg.distributed.world_size, 
                            rank=rank)
    device = torch.device("cpu")

    # Data init
    data = TestDataset(cfg)
    sampler = torch.utils.data.distributed.DistributedSampler(data, 
                                                            num_replicas=cfg.distributed.world_size, 
                                                            rank=rank)
    collator = Collator(cfg)
    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=0, collate_fn=collator, sampler=sampler)

    # Load model
    print(f'Loading model.')
    if cfg.model_name == 'ctc':
        model = baseCTC.from_pretrained(cfg, cfg.paths.ckpt_path)
    elif cfg.model_name == 'sc_ctc':
        model = baseSCCTC.from_pretrained(cfg, cfg.paths.ckpt_path)
    elif cfg.model_name == 'hc_ctc':
        model = baseHCCTC.from_pretrained(cfg, cfg.paths.ckpt_path)
    else:
        raise ValueError(f"Model '{cfg.model_name}' is invalid. Current implementation includes: 'ctc', 'sc_ctc' and 'hc_ctc'")

    # Decode
    print(f'Decoding.')
    if rank==0 and not os.path.exists(cfg.paths.decode_path):
        os.makedirs(cfg.paths.decode_path)
    if rank != 0:
        while not os.path.exists(cfg.paths.decode_path):
            continue
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    rfh = RotatingFileHandler(os.path.join(cfg.paths.decode_path, f'{rank}.log'), maxBytes=1000000, backupCount=10, encoding="UTF-8")
    logger.addHandler(rfh)
    print(f'Starting transcription.')
    with open(os.path.join(cfg.paths.decode_path, f'{rank}.txt'), 'w') as dP:
        for audio_path, text in tqdm(loader, disable=(rank!=0)):
            hyp = model.transcribe(audio_path[0]).replace(" ' ", "'").replace("-", "")
            gt = text[0].replace("-", "")
            logger.info(f'{audio_path[0]} ----> {gt} ----> {hyp}')
            dP.write(f'{audio_path[0]} ----> {gt} ----> {hyp}\n')

if __name__ == "__main__":
    # Parse config_path and config_name from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--config-path", type=str, default=".", help="Path to the config directory")
    parser.add_argument("--config-name", type=str, default="config", help="Name of the config file (without .yaml extension)")

    args, unknown = parser.parse_known_args()

    # Use the provided config_path and config_name
    with initialize(config_path=args.config_path):
        cfg = compose(config_name=args.config_name, overrides=unknown)
        main(cfg)

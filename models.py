import pdb
import time
import numpy as np
import random
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
import torchaudio
import torchaudio.transforms as AT
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from omegaconf import DictConfig
from tqdm import tqdm
from util import *
from encoders import *
from tokenizers import decoders

def logsumexp(a, b):
    return np.log(np.exp(a) + np.exp(b))

class baseCTC(nn.Module):
    def __init__(self, cfg):
        super(baseCTC, self).__init__()
        self.cfg = cfg
        self.downsampler = DownSampler(cfg)
        self.encoder = ConformerEncoder(cfg)
        
    def forward(self, batch):
        x = self.downsampler(batch.speechB)
        y, _, _, y_inter = self.encoder(x)
        return y, y_inter

    @classmethod
    def from_pretrained(cls, cfg, checkpoint_path=None, device='cpu'):
        if not cfg.model.vocab_size:
            cls.tokenizer = Tokenizer.from_file(cfg.paths.tokenizer_path)
            cls.tokenizer.decoder = decoders.WordPiece()
            cfg.model.vocab_size = cls.tokenizer.get_vocab_size()
            if cfg.model_name == 'hc_ctc':
                cls.inter_tokenizers = []
                cfg.model.inter_vocab_size = []
                for p in cfg.paths.inter_tokenizer_paths:
                    tok = Tokenizer.from_file(p)
                    tok.decoder = decoders.WordPiece()
                    cls.inter_tokenizers.append(tok)
                    cfg.model.inter_vocab_size.append(tok.get_vocab_size())

        device_obj = torch.device(device)
        model = cls(cfg).to(device_obj)
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            load_dict(model, checkpoint['state_dict'])
        return model

    @torch.no_grad()
    def transcribe(self, path, layer_num=None):
        self.eval()
        if layer_num is not None and self.cfg.model_name == 'ctc':
            raise ValueError('cannot infer inter layer in ctc')
        wav, sr = torchaudio.load(path)
        if sr != self.cfg.features.sample_rate:
            wav = AT.Resample(sr, self.cfg.features.sample_rate)(wav)
        features = mel_spectogram(audio=wav, sample_rate=self.cfg.features.sample_rate,
                                    hop_length=160, win_length=400, n_mels=self.cfg.features.n_mels,
                                    power=1, normalized=False, min_max_energy_norm=True, norm="slaney",
                                    mel_scale="slaney", compression=True).permute(0,2,1)
        features, _ = normalize_spec(features)
        x = self.downsampler(features)
        y, _, _, y_inter = self.encoder(x) # (T, 1, C)
        if layer_num is None:
            y_i = y
        else:
            y_i = y_inter[layer_num]
        y_i = y_i.squeeze(1) # (T, C)
        indices = torch.argmax(y_i, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1).tolist()
        indices = [i for i in indices if i != 0]
        return self.tokenizer.decode(indices)

    @torch.no_grad()
    def transcribe_batch(batch):
        self.eval()
        y, _ = self.forward(batch) # (T, 16, C)
        indices = torch.argmax(y, dim=-1).T # (16, T)
        outs = []
        B = indices.size(0)
        for i in range(B):
            indices_i = indices[i][:batch.logitLens[i]]
            indices_i = torch.unique_consecutive(indices_i, dim=-1).tolist()
            indices_i = [i for i in indices_i if i != 0]
            outs.append(self.tokenizer.decode(indices_i))
        return outs

class baseSCCTC(baseCTC):
    def __init__(self, cfg):
        super(baseSCCTC, self).__init__(cfg)
        self.encoder = ConformerEncoderSCCTC(cfg)

class baseHCCTC(baseCTC):
    def __init__(self, cfg):
        super(baseHCCTC, self).__init__(cfg)
        self.encoder = ConformerEncoderHCCTC(cfg)


import torch
import math
import random
import json
import ast
import re
import os
import time
import pandas as pd
import pdb
import string
import numpy as np
import torch.distributed as dist
from torchaudio import transforms
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

def mel_spectogram(sample_rate, hop_length, win_length, n_mels, power, normalized, min_max_energy_norm, norm, mel_scale, compression, audio):
    audio_to_mel = transforms.Spectrogram(
        hop_length=hop_length,
        win_length=win_length,
        n_fft=win_length,
        power=power,
        normalized=normalized,
    ).to(audio.device)

    mel_scale = transforms.MelScale(
        sample_rate=sample_rate,
        n_stft=win_length // 2 + 1,
        n_mels=n_mels,
        f_min=0.,
        f_max=sample_rate//2,
        norm=norm,
        mel_scale=mel_scale,
    ).to(audio.device)
    spec = audio_to_mel(audio)
    mel = mel_scale(spec)

    if compression:
        mel = dynamic_range_compression(mel)

    return mel

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """Dynamic range compression for audio signals
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)

class statRecorder(object):
    def __init__(self, *args, **kwargs):
        self.losses = []
        self.loss_temp = {}
        self.loss_perm = {}
        for loss in args:
            self.loss_temp[loss] = 0.
            self.loss_perm[loss] = []
            self.losses.append(loss)
        self.ddp = kwargs['ddp']

    def add_loss(self, name):
        self.loss_temp[name] = 0.
        self.loss_perm[name] = []
        self.losses.append(name)

    def backward(self, loss_name, loss_val):
        self.loss_temp[loss_name] += loss_val

    def accumulate(self):
        WS = dist.get_world_size() if self.ddp else 1
        for loss_name, loss_val in self.loss_temp.items():
            if self.ddp:
                dist.all_reduce(self.loss_temp[loss_name])
            self.loss_perm[loss_name].append(self.loss_temp[loss_name].item() / WS)

    def reset(self):
        for k in self.loss_temp:
            self.loss_temp[k] = 0.

    def empty(self):
        for k in self.loss_perm:
            self.loss_perm[k] = []

    def display(self):
        out = []
        for k in self.loss_perm:
            out.append(f' {k} = {np.mean(self.loss_perm[k])} ')
        return '|'.join(out)

    def get(self):
        outDct = {}
        for k in self.loss_perm:
            outDct[k] = np.mean(self.loss_perm[k])
        return outDct

def list_batch(X, lens):
    sbatch_ = []
    for i, l in enumerate(lens):
        sbatch_.append(X[i,:l,:])
    return sbatch_

def add_delta2(features):
    buf = features.numpy()
    frameN = buf.shape[0]
    logmel = np.pad(buf, ((2,2),(0,0)), 'edge')
    
    delta = logmel[2:,:] - logmel[:-2,:]
    ddelta = delta[2:,:] - delta[:-2,:]

    ldd = np.concatenate((logmel[2:-2,:], delta[1:-1,:], ddelta), axis=1)
    ldd_shift = np.roll(ldd, -1, axis=0)
    out = np.concatenate((ldd, ldd_shift), axis=1)

    even = range(random.randint(0,1), frameN, 2)
    fea = out[even]

    return torch.from_numpy(fea)

def normalize_spec(x): # x -> (T, F)
    mean = x.mean()
    std = x.std() + 1e-5
    x_ = (x - mean) / std
    return x_, mean

def specaug2(X, M, fmask_prob=0.9, fmask_m=2, fmask_F=27, tmask_prob=0.9, tmask_m=10, tmask_m_relative_max=0.02, tmask_T=100):
    input = X.numpy()
    resetVal = M.item() #0.#input.mean()
    t_size, f_size = input.shape
    tmask_T = max(int(t_size * 0.05), 2)

    if np.random.uniform() < fmask_prob:
        num = random.randint(1, fmask_m)
        for c in range(num):
            f = random.randint(1, fmask_F)
            f0 = random.randint(0, f_size - f)
            input[:, f0:f0+f] = resetVal

    if np.random.uniform() < tmask_prob:
        num = random.randint(1, max(1, min(tmask_m, int(float(t_size)*tmask_m_relative_max))))
        for c in range(num):
            try:
                t = random.randint(1, tmask_T)
            except:
                print(f't_size = {t_size} ; tmask_T = {tmask_T}')
                raise ValueError('problem!')
            t0 = random.randint(0, t_size - t)
            input[t0:t0+t, :] = resetVal

    return torch.from_numpy(input)

def clean4asr(text):
    tokens = text.strip().split()
    out = []
    for tok in tokens:
        if tok[0] == '[' and tok[-1] == ']':
            continue
        out.append(re.sub('[^A-Za-z0-9\s\-\']+','',tok).lower())
    return ' '.join(out).strip()

def my_shuffle(x):
    if len(x) == 1:
        raise Exception
    for i in reversed(range(1, len(x))):
        # pick an element in x[:i] with which to exchange x[i]
        j = int(random.random() * i)
        x[i], x[j] = x[j], x[i]

def inject_seqn(X):
    if X.size(0) == 1:
        return X
    mask_sn = torch.ones(X.shape[0],1,1)
    zr = random.sample(list(range(X.shape[0])), int(0.2*X.shape[0]))
    mask_sn[zr] = 0.
    Z = list(range(X.shape[0]))
    my_shuffle(Z)
    X = torch.log(torch.exp(X) + 0.4 * torch.exp(X[Z]) * mask_sn)
    return X

def load_dict(model, ptdict, ddp=False):
    pretrained_dict = ptdict
    model_dict = model.state_dict()
    new_pt_dict = {}
    for k, v in pretrained_dict.items():
        k_new = k
        if not ddp and k[:7] == 'module.':
            k_new = k[7:]
        elif ddp and k[:7] != 'module.':
            k_new = f'module.{k}'
        new_pt_dict[k_new] = v
    pretrained_dict = {k: v for k, v in new_pt_dict.items() if k in model_dict}
    #for k, v in new_pt_dict.items():
    #    if k not in model_dict:
    #        print(f'NOT PRESENT = {k}')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def save_pick(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pick(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save(model, path):
    torch.save(model.state_dict(), path)

def load(model, path):
    model.load_state_dict(torch.load(path))

def save_checkpoint(state, filename):
    torch.save(state, filename)

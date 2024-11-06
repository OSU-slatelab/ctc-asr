import torch
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
import torchaudio
import torchaudio.transforms as AT
import copy
from util import *
from tqdm import tqdm
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

class batchInstance(object):
    def __init__(self, dct):
        for name in dct:
            setattr(self, name, dct[name])

    def load2gpu(self, device):
        for name, value in self.__dict__.items():
            try:
                setattr(self, name, value.to(device))
            except AttributeError:
                if isinstance(value, dict):
                    t2 = {}
                    for k, v in value.items():
                        if torch.is_tensor(v):
                            t2[k] = v.to(device)
                        else:
                            t2[k] = v
                    setattr(self, name, t2)
                elif isinstance(value, list):
                    y = []
                    for v in value:
                        if torch.is_tensor(v):
                            y.append(v.to(device))
                        else:
                            y.append(v)
                    setattr(self, name, y)

class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.df = pd.read_csv(cfg.paths.train_path)

    def get_filterbanks(self, signal):
        features = mel_spectogram(audio=signal, sample_rate=self.cfg.features.sample_rate, hop_length=160, win_length=400, n_mels=self.cfg.features.n_mels, power=1, normalized=False, min_max_energy_norm=True, norm="slaney", mel_scale="slaney", compression=True)
        return features.permute(0,2,1)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        wav, sr = torchaudio.load(row['audio_file'])
        if sr != self.cfg.features.sample_rate:
            wav = AT.Resample(sr, self.cfg.features.sample_rate)(wav)
        return self.get_filterbanks(wav)

class ASRDataset(SpeechDataset):
    def __init__(self, cfg, train=True):
        super(ASRDataset, self).__init__(cfg)
        self.train = train
        self.max_len = int(cfg.features.sample_rate*cfg.features.wav_len)

    def crop(self, signal):
        if not self.train:
            return signal
        length_adj = signal.shape[1] - self.max_len
        if length_adj > 0:
            start = random.randint(0, length_adj)
            return signal[:,start:start + self.max_len]
        return signal

    def __getitem__(self, index):
        row = self.df.iloc[index]
        if self.cfg.corpus == 'librispeech':
            audio_path = row['audio_file'].replace('/data/corpora2/librispeech/LibriSpeech/', '/research/nfs_fosler_1/vishal/audio/libri/')
            if os.path.isfile(audio_path):
                    wav, org_sr = torchaudio.load(audio_path)
            else:
                print(f'skipping {audio_path}')
                return None, -1
        elif self.cfg.corpus == 'fisher_swb' or self.cfg.corpus == 'all':
            if '.sph' in row['audio_file']:
                key = str(row['cnum'])+'_'+str(int(row['utt_id']))
                audio_path = os.path.join(f'/research/nfs_fosler_1/vishal/audio/fisher', f'{key}.npz')
                wav = torch.from_numpy(np.load(audio_path)['a'])
                org_sr = 8000
            elif 'librispeech' in row['audio_file']:
                audio_path = row['audio_file'].replace('/data/corpora2/librispeech/LibriSpeech/', '/research/nfs_fosler_1/vishal/audio/libri/')
                if os.path.isfile(audio_path):
                    wav, org_sr = torchaudio.load(audio_path)
                else:
                    print(f'skipping {audio_path}')
                    return None, -1
            else:
                audio_path = row['audio_file']
                if os.path.isfile(audio_path):
                    wav, org_sr = torchaudio.load(audio_path)
                else:
                    print(f'skipping {audio_path}')
                    return None, -1
        if org_sr != self.cfg.features.sample_rate:
            wav = AT.Resample(org_sr, self.cfg.features.sample_rate)(wav)
        wav = self.crop(wav)
        target = row['utterance']
        sign = f"{row.audio_file.split('/')[-1]}"
        return self.get_filterbanks(wav), clean4asr(target), sign

class CollatorCTC(object):
    def __init__(self, cfg, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer

    def get_speech_batch(self, speechL):
        sbatch_norm_spec = [] 
        for x in speechL:
            x_, mean = normalize_spec(x)
            sbatch_norm_spec.append(specaug2(x_, mean, fmask_F=self.cfg.features.fmask))
        pack1 = pack_sequence(sbatch_norm_spec, enforce_sorted=False)
        speechB, logitLens = pad_packed_sequence(pack1, batch_first=True)
        logitLens = torch.ceil(logitLens / 4).int()
        return speechB, logitLens

    def get_target_batch(self, textL, tokenizer):
        targetCtc = []
        targetLensCtc = []
        for x in textL:
            tgt = torch.tensor(tokenizer.encode(x).ids)
            targetCtc.append(tgt)
            targetLensCtc.append(len(tgt))
        targetCtc = torch.cat(targetCtc)
        targetLensCtc = torch.tensor(targetLensCtc)
        return targetCtc, targetLensCtc

    def __call__(self, lst):
        speechL = [x[0].squeeze(0) for x in lst if x[0].size(1) > 2 and len(x[1]) > 1]
        textL = [x[1] for x in lst if x[0].size(1) > 2 and len(x[1]) > 1]
        if len(speechL) == 0 or len(textL) == 0:
            return

        speechB, logitLens = self.get_speech_batch(speechL)
        target, targetLens = self.get_target_batch(textL, self.tokenizer)

        dct = {'speechB':speechB, 'target':target, 'targetLens':targetLens, 'logitLens':logitLens}
        batch = batchInstance(dct)

        return batch

class CollatorSCCTC(CollatorCTC):
    def __init__(self, cfg, tokenizer):
        super(CollatorSCCTC, self).__init__(cfg, tokenizer)

    def __call__(self, lst):
        speechL = [x[0].squeeze(0) for x in lst if x[0].size(1) > 2 and len(x[1]) > 1]
        textL = [x[1] for x in lst if x[0].size(1) > 2 and len(x[1]) > 1]
        if len(speechL) == 0 or len(textL) == 0:
            return

        speechB, logitLens = self.get_speech_batch(speechL)
        target, targetLens = self.get_target_batch(textL, self.tokenizer)

        targetInter = target.repeat(self.cfg.model.num_ctc-1)
        targetLensInter = targetLens.repeat(self.cfg.model.num_ctc-1)
        logitLensInter = logitLens.repeat(self.cfg.model.num_ctc-1)

        dct = {'speechB':speechB, 'target':target, 'targetLens':targetLens, 'logitLens':logitLens, 'targetInter':targetInter, 'targetLensInter':targetLensInter, 'logitLensInter':logitLensInter}
        batch = batchInstance(dct)

        return batch

class CollatorHCCTC(CollatorCTC):
    def __init__(self, cfg, tokenizer, inter_tokenizers):
        super(CollatorHCCTC, self).__init__(cfg, tokenizer)
        self.inter_tokenizers = inter_tokenizers

    def __call__(self, lst):
        speechL = [x[0].squeeze(0) for x in lst if x[0].size(1) > 2 and len(x[1]) > 1]
        textL = [x[1] for x in lst if x[0].size(1) > 2 and len(x[1]) > 1]
        if len(speechL) == 0 or len(textL) == 0:
            return

        speechB, logitLens = self.get_speech_batch(speechL)
        target, targetLens = self.get_target_batch(textL, self.inter_tokenizers[-1])

        targetInter, targetLensInter = [], []
        for tok in self.inter_tokenizers[:-1]:
            target_i, targetLens_i = self.get_target_batch(textL, tok)
            targetInter.append(target_i)
            targetLensInter.append(targetLens_i)

        dct = {'speechB':speechB, 'target':target, 'targetLens':targetLens, 'logitLens':logitLens, 'targetInter':targetInter, 'targetLensInter':targetLensInter}
        batch = batchInstance(dct)

        return batch

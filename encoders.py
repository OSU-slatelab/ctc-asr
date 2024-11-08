import pdb
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from conformer import ConformerBlock
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

def get_mask(lens, device):
    mask = torch.ones(len(lens), max(lens), device=device)
    for i, l in enumerate(lens):
        mask[i][:l] = 0.
    return mask

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class DownSampler(nn.Module):
    def __init__(self, cfg):
        super(DownSampler, self).__init__()
        if cfg.features.downsample == 4:
            self.conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
                                     nn.SiLU(),
                                     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(2,2), padding=(1,1), groups=256),
                                     nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)),
                                     nn.SiLU(),
                                     )
        elif cfg.features.downsample == 8:
            self.conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
                                 nn.SiLU(),
                                 nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(2,2), padding=(1,1), groups=256),
                                 nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)),
                                 nn.SiLU(),
                                 nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(2,2), padding=(1,1), groups=256),
                                 nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)),
                                 nn.SiLU()
                                 )
        else:
            raise ValueError(f'cfg.features.downsample should be 4 or 8')
        self.out = nn.Sequential(nn.Dropout(0.1), nn.Linear(256 * cfg.features.n_mels // cfg.features.downsample, cfg.model.hidDim))

    def forward(self, x):
        x = x.unsqueeze(1)
        out_conv = self.conv(x).permute(0, 2, 1, 3)
        B, T, C, F = out_conv.size()
        y = out_conv.reshape(B, T, C*F)
        z = self.out(y)
        return z

#class DownSampler(nn.Module):
#    def __init__(self, cfg):
#        super(DownSampler, self).__init__()
#        self.ln = nn.LayerNorm(cfg.features.n_mels)
#        self.conv = nn.Conv1d(in_channels=cfg.features.n_mels, out_channels=cfg.model.hidDim, kernel_size=cfg.features.downsample, stride=cfg.features.downsample)
#        self.lin = nn.Sequential(nn.Linear(cfg.model.hidDim, cfg.model.hidDim), nn.Dropout(cfg.model.dropout))
#
#    def forward(self, x):
#        x = self.ln(x)
#        x = self.conv(x.permute(0,2,1)).permute(0,2,1) # L -> L//4
#        x = self.lin(x)
#        return x

class ConformerLayer(nn.Module):
    def __init__(self, cfg):
        super(ConformerLayer, self).__init__()
        self.block = ConformerBlock(dim = cfg.model.hidDim, dim_head=cfg.model.headDim, heads=cfg.model.nhead, ff_mult = 4, conv_expansion_factor = 2, conv_kernel_size = cfg.model.conv_kernel_size, attn_dropout = cfg.model.dropout, ff_dropout = cfg.model.dropout, conv_dropout = cfg.model.dropout)
        
    def forward(self, x):
        return self.block(x)

class ConformerEncoder(nn.Module):
    def __init__(self, cfg):
        super(ConformerEncoder, self).__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([ConformerLayer(cfg) for _ in range(cfg.model.n_layer)])
        self.ln = nn.LayerNorm(cfg.model.hidDim)
        if isinstance(cfg.model.vocab_size, int):
            self.classifier = nn.Sequential(nn.Dropout(cfg.model.dropout), nn.Linear(cfg.model.hidDim, cfg.model.vocab_size))

    def forward(self, x):
        full = []
        for i in range(self.cfg.model.n_layer):
            x = self.layers[i](x)
            full.append(x)
        x = self.ln(x)
        out = F.log_softmax(self.classifier(x), dim=-1).permute(1,0,2)
        return out, x, full, None

class ConformerEncoderSCCTC(ConformerEncoder):
    def __init__(self, cfg):
        super(ConformerEncoderSCCTC, self).__init__(cfg)
        self.bottleneck = nn.Sequential(nn.Dropout(cfg.model.dropout), nn.Linear(cfg.model.vocab_size, cfg.model.hidDim))
        if cfg.model.n_layer % cfg.model.num_ctc:
            raise ValueError("Number of intermediate CTCs can't be evenly spread")
        self.sc_after = cfg.model.n_layer / cfg.model.num_ctc

    def forward(self, x):
        inter_out = []
        full = []
        for i in range(self.cfg.model.n_layer):
            x = self.layers[i](x)
            full.append(x)
            if not (i+1) % self.sc_after and (i+1)<self.cfg.model.n_layer:
                inter_ln = self.ln(x)
                inter_cls = self.classifier(inter_ln)
                inter_dist = torch.softmax(inter_cls, dim=-1)
                inter_out.append(torch.log(inter_dist))
                x = self.bottleneck(inter_dist) + inter_ln
        x = self.ln(x)
        out = F.log_softmax(self.classifier(x), dim=-1).permute(1,0,2)
        out_inter = torch.cat(inter_out, dim=0).permute(1,0,2)
        return out, x, full, out_inter

class ConformerEncoderHCCTC(nn.Module):
    def __init__(self, cfg):
        super(ConformerEncoderHCCTC, self).__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([ConformerLayer(cfg) for _ in range(cfg.model.n_layer)])
        if cfg.model.n_layer % cfg.model.num_ctc:
            raise ValueError("Number of intermediate CTCs can't be evenly spread")
        self.sc_after = cfg.model.n_layer / cfg.model.num_ctc
        self.ln = nn.ModuleList([nn.LayerNorm(cfg.model.hidDim) for _ in range(cfg.model.num_ctc)])
        self.classifier = nn.ModuleList([nn.Sequential(nn.Dropout(cfg.model.dropout), nn.Linear(cfg.model.hidDim, cfg.model.inter_vocab_size[i])) for i in range(cfg.model.num_ctc)])
        self.bottleneck = nn.ModuleList([nn.Sequential(nn.Dropout(cfg.model.dropout), nn.Linear(cfg.model.inter_vocab_size[i], cfg.model.hidDim)) for i in range(cfg.model.num_ctc-1)])

    def forward(self, x):
        inter_out = []
        full = []
        j = 0
        for i in range(self.cfg.model.n_layer):
            x = self.layers[i](x)
            full.append(x)
            if not (i+1) % self.sc_after and (i+1)<self.cfg.model.n_layer:
                inter_ln = self.ln[j](x)
                inter_cls = self.classifier[j](inter_ln)
                inter_dist = torch.softmax(inter_cls, dim=-1)
                inter_out.append(torch.log(inter_dist).permute(1,0,2))
                x = self.bottleneck[j](inter_dist) + inter_ln
                j += 1
        assert j+1 == self.cfg.model.num_ctc
        x = self.ln[j](x)
        out = F.log_softmax(self.classifier[j](x), dim=-1).permute(1,0,2)
        return out, x, full, inter_out

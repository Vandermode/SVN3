import sys
sys.path.append(".")
import torch
import torch.nn as nn
from typing import OrderedDict
from scipy.io import loadmat
from models.common import *


def get_elec_layer(elec_layer_type):
    if elec_layer_type == 'dws-a1':
        elec_layer = dwsconv_bn_a1
    elif elec_layer_type == 'dws-a1-infer':
        elec_layer = dwsconv_bn_a1_infer  # inference mode: merging bn into the last convolutional layer
    elif elec_layer_type == 'dws-a1-prelu':
        elec_layer = dwsconv_bn_a1_prelu
    elif elec_layer_type == 'conv':
        elec_layer = conv_bn
    else:
        raise NotImplementedError
    return elec_layer


def get_local_pool(pooling_type):
    if pooling_type == 'max':
        local_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    elif pooling_type == 'avg':
        local_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
    elif pooling_type == 'avg-s4':
        local_pool = nn.AvgPool2d(kernel_size=5, stride=4, padding=2)
    elif pooling_type == 'ps':  # pixel shuffle
        local_pool = nn.PixelUnshuffle(2)
    elif pooling_type == 'none':
        local_pool = nn.Identity()
    else:
        raise NotImplementedError
    return local_pool


def get_opto_layer(opto_layer_info, in_channels, out_channels):
    opto_layer_type = opto_layer_info.pop('type')
    if 'out_channels' in opto_layer_info:
        out_channels = opto_layer_info.pop('out_channels')    
    if opto_layer_type == 'conv':
        opto_layer = get_conv2d(in_channels=in_channels, out_channels=out_channels, **opto_layer_info)
    elif opto_layer_type == 'conv_bn':
            opto_layer = conv_bn(in_channels=in_channels, out_channels=out_channels, **opto_layer_info)
    elif opto_layer_type == 'correct': # correct layer
        opto_layer = CorrectLayer(num_channels=out_channels, **opto_layer_info)
    elif opto_layer_type == 'fconv': # factorized conv
        opto_layer = FactorizedConv2d(in_channels=in_channels, out_channels=out_channels, **opto_layer_info)
    elif opto_layer_type == 'ftconv': # factorized transposed conv
        opto_layer = FactorizedTConv2d(in_channels=in_channels, out_channels=out_channels, **opto_layer_info)
    elif opto_layer_type == 'svconv':
        opto_layer = NaiveSVConv(in_channels=in_channels, out_channels=out_channels, stride=1, **opto_layer_info)
    elif opto_layer_type == 'lrsvconv':
        opto_layer = LRSVConv(in_channels=in_channels, out_channels=out_channels, stride=1, **opto_layer_info)
    elif opto_layer_type == 'lrsvtconv':
        opto_layer = LRSVTConv(in_channels=in_channels, out_channels=out_channels, stride=1, **opto_layer_info)
    else:
        raise NotImplementedError
    return opto_layer


def get_act_func(act_type):
    if act_type == 'relu':
        act = nn.ReLU()
    elif act_type == 'gelu':
        act = nn.GELU()
    elif act_type == 'prelu':
        act = nn.PReLU()
    else:
        raise NotImplementedError
    return act


class OENetElecStage(nn.Module):
    def __init__(self, channels, num_blocks, elec_layer_type='dws-a', elec_kernel_size=3):
        super().__init__()
        blks = []
        elec_layer = get_elec_layer(elec_layer_type)
        for i in range(num_blocks):
            elec_block = elec_layer(channels, channels, elec_kernel_size, 1)
            blks.append(elec_block)
        self.blocks = nn.ModuleList(blks)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class CorrectLayer(nn.Module):
    def __init__(self, num_channels, downsample_factor=4, learn_downsample=False, single_filter=True, correct_vars_path=None) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.downsample_factor = downsample_factor
        self.learn_downsample = learn_downsample
        self.correct_vars = nn.ParameterDict({
            's1': nn.Parameter(torch.ones(1, num_channels, 1, 1)),
            's2': nn.Parameter(-torch.ones(1, num_channels, 1, 1)),
        })
        
        if downsample_factor > 1 and learn_downsample:
            if single_filter:
                self.correct_vars['df'] = nn.Parameter(torch.ones(1, 1, downsample_factor*downsample_factor))
            else:
                self.correct_vars['df1'] = nn.Parameter(torch.ones(num_channels, 1, downsample_factor*downsample_factor))
                self.correct_vars['df2'] = nn.Parameter(torch.ones(num_channels, 1, downsample_factor*downsample_factor))
                
        if correct_vars_path is not None:
            mat = loadmat(correct_vars_path)
            with torch.no_grad():
                self.correct_vars['s1'].copy_(torch.from_numpy(mat['s1']))
                self.correct_vars['s2'].copy_(torch.from_numpy(mat['s2']))

    def forward(self, paired_ftr):
        ftr_p, ftr_n = paired_ftr
        N, C, H, W = ftr_p.shape
        s1, s2 = self.correct_vars['s1'], self.correct_vars['s2']
        sf = self.downsample_factor
        if hasattr(self.correct_vars, 'df1'):
            df1 = F.softmax(self.correct_vars['df1'], dim=-1).view(self.num_channels, 1, sf, sf)
            df2 = F.softmax(self.correct_vars['df2'], dim=-1).view(self.num_channels, 1, sf, sf)
            ftr_p = F.conv2d(ftr_p, weight=df1, bias=None, stride=sf, groups=self.num_channels)
            ftr_n = F.conv2d(ftr_n, weight=df2, bias=None, stride=sf, groups=self.num_channels)
        elif hasattr(self.correct_vars, 'df'):
            df = F.softmax(self.correct_vars['df'], dim=-1).view(1, 1, sf, sf)
            ftr_p = ftr_p.view(N*C, 1, H, W)
            ftr_n = ftr_n.view(N*C, 1, H, W)
            ftr_p = F.conv2d(ftr_p, weight=df, bias=None, stride=sf).view(N, C, H//sf, W//sf)
            ftr_n = F.conv2d(ftr_n, weight=df, bias=None, stride=sf).view(N, C, H//sf, W//sf)
        elif sf > 1:
            ftr_p = F.avg_pool2d(ftr_p, kernel_size=sf, stride=sf)
            ftr_n = F.avg_pool2d(ftr_n, kernel_size=sf, stride=sf)
        ftr = s1[:, :C, ...] * ftr_p + s2[:, :C, ...] * ftr_n
        
        return ftr


class OENet(nn.Module):
    def __init__(self, num_classes=10, in_channels=1, channels=[25, 50, 100], 
                 elec_layers=[1, 1, 0], elec_kernel_sizes=[3, 3, 3], 
                 act_type='relu', pooling_type='avg', 
                 opto_layer_info={'type': 'conv', 'kernel_size': 3}, elec_layer_type='dws-a1', first_norm=False, last_norm=True,
                 out_keys=None, aux_key=None):
        super().__init__()
        self.channels = channels
        self.num_stages = len(elec_layers)
        self.opto_stem = get_opto_layer(opto_layer_info, in_channels, channels[0])
        self.first_norm = nn.BatchNorm2d(channels[0]) if first_norm else None
        self.act = get_act_func(act_type)
        self.local_pool = get_local_pool(pooling_type)
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()
        
        if isinstance(elec_layer_type, list):
            elec_layer_op = [get_elec_layer(elec_layer_type_i) for elec_layer_type_i in elec_layer_type]
        else:
            elec_layer_op = get_elec_layer(elec_layer_type)
        for stage_idx in range(self.num_stages):
            if elec_layers[stage_idx] == 0:
                layer = nn.Identity()
            else:
                elec_layer_type_i = elec_layer_type if not isinstance(elec_layer_type, list) else elec_layer_type[stage_idx]
                layer = OENetElecStage(channels[stage_idx], num_blocks=elec_layers[stage_idx], 
                                       elec_layer_type=elec_layer_type_i, elec_kernel_size=elec_kernel_sizes[stage_idx])
            self.stages.append(layer)
            if stage_idx < len(elec_layers) - 1:
                elec_layer_op_i = elec_layer_op if not isinstance(elec_layer_type, list) else elec_layer_op[stage_idx]
                transition = elec_layer_op_i(channels[stage_idx], channels[stage_idx + 1], 3, 2)
                self.transitions.append(transition)

        self.norm = nn.BatchNorm2d(channels[-1]) if last_norm else None
        self.avgpool = nn.AdaptiveAvgPool2d(1)        
        self.head = nn.Linear(channels[-1], num_classes)
        
        # segmentation related
        self.out_keys = out_keys
        self.aux_key = aux_key
        
    def remove_head(self):
        del self.head
        del self.norm
        del self.avgpool
        
    def remove_opto_stem(self):
        del self.opto_stem
        self.opto_stem = nn.Identity()
        
    def freeze_opto_stem(self):
        self.opto_stem.requires_grad_(False)
        self.opto_stem.eval()
    
    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()
    
    def get_elec_backbone_params(self):
        from itertools import chain
        return chain(self.first_norm.parameters(), self.stages.parameters(), self.transitions.parameters())
    
    def get_elec_params(self):
        from itertools import chain
        return chain(self.first_norm.parameters(), self.stages.parameters(), self.transitions.parameters(), self.norm.parameters(), self.head.parameters())

    def optical_forward(self, x, *args, **kwargs):
        x = self.opto_stem(x, *args, **kwargs)
        return x
    
    def elec_forward(self, x, extract_features=False):
        x, features = self.forward_elec_backbone(x, extract_features=extract_features)
        x = self.classify(x)
        return x, features
    
    def forward_backbone(self, x):
        x = self.optical_forward(x)
        x, _features = self.forward_elec_backbone(x, extract_features=True)        
        features = OrderedDict()
        if self.out_keys is not None:
            features['out'] = [_features[out_key] for out_key in self.out_keys]
        if self.aux_key is not None:
            features['aux'] = _features[self.aux_key]
        return features
    
    def forward_elec_backbone(self, x, extract_features=False):
        features = OrderedDict()
        if extract_features:
            features['opto_stem'] = x
        if self.first_norm:
            x = self.first_norm(x)
        x = self.act(x)
        x = self.local_pool(x)
        for stage_idx in range(self.num_stages):            
            x = self.stages[stage_idx](x)
            if extract_features:
                features[f'stage_{stage_idx}'] = x
            if stage_idx < self.num_stages - 1:
                x = self.transitions[stage_idx](x)
                if extract_features:
                    features[f'transition_{stage_idx}'] = x
        return x, features
    
    def classify(self, x):
        if self.norm:
            x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

    def forward(self, x, *args, **kwargs):
        x = self.optical_forward(x, *args, **kwargs)
        x, _ = self.elec_forward(x, extract_features=False)
        return x
    
    def feature_extract(self, x):
        x = self.optical_forward(x)
        logits, features = self.elec_forward(x, extract_features=True)
        return logits, features

    def get_bn_before_relu(self):
        def get_last_bn(module):
            last_bn = None
            for n, m in module.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    name = n
                    last_bn = m
            return last_bn
        bns = []
        bns.append(get_last_bn(self.opto_stem))
        return bns

    def get_channel_num(self):
        return self.channels


def oenet(num_classes, **params):
    return OENet(num_classes=num_classes, **params)

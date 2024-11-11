import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.signal as signal


def conv2d(x, weights):
    for weight in weights:
        x = F.conv2d(x, weight)
    return x
    
    
def deconv2d(x, weights):
    for weight in weights:
        x = F.conv_transpose2d(x, weight)
    return x


def svconv2d(x, weights):
    for weight in weights:
        N = x.shape[0]
        Cout, Cin, K, _, H, W = weight.shape
        x = F.unfold(x, K, 1, 0, 1)
        x = (weight.view(1, Cout, K*K*Cin, H*W) * x.view(N, 1, K*K*Cin, H*W)).sum(dim=2).view(N, Cout, H, W)
    return x


def svconv_transpose2d(x, weight, keep_size=True):
    N, C, H, W = x.shape
    Cin, Cout, Kh, Kw, _, _ = weight.shape    
    x = x.view(N, C, 1, 1, 1, H, W)
    y = x * weight.view(1, Cin, Cout, Kh, Kw, H, W)
    y = y.sum(dim=1)
    y = y.view(N, Cout*Kh*Kw, H*W)
    out = F.fold(y, (H + Kh - 1, W + Kw - 1), (Kh, Kw))
    if keep_size:
        pad = Kh // 2
        out = out[:, :, pad:-pad, pad:-pad]
    return out

    
def svdeconv2d(x, weights):
    for weight in weights:
        N, Cin, K1, _, H1, W1 = x.shape
        _, Cout, K2, _, H2, W2 = weight.shape
        a1 = x.view(N, Cin, 1, 1, K1*K1, H1*W1)
        a2 = F.unfold(weight.view(Cin, Cout*K2*K2, H2, W2), K1, 1, 0, 1).view(1, Cin, Cout, K2*K2, K1*K1, H1*W1)         
        a3 = (a1 * a2).sum(1)
        a3 = a3.permute(4, 0, 1, 2, 3).reshape(H1*W1*N, Cout*K2*K2, K1*K1)
        Kout = K1 + K2 - 1
        del a1
        del a2
        torch.cuda.empty_cache()
        x = F.fold(a3, output_size=(Kout, Kout), kernel_size=(K2, K2), stride=1)
        del a3
        x = x.view(H1*W1, N, Cout, Kout, Kout).permute(1, 2, 3, 4, 0).view(N, Cout, Kout, Kout, H1, W1)
    return x


def out_psf_forward(x, out_psf, flip=False):
    N = x.shape[0]
    C, D, H, W = out_psf.shape
    assert D == H * W
    out = (x.view(N, 1, -1, 1, 1) * out_psf.view(1, C, D, H, W)).sum(dim=2) # N C H W
    if flip:
        out = torch.flip(out, [-2, -1])
    return out


def fuse_bn(kernel, bias, bn, is_sv=False):
    bias = 0 if bias is None else bias
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()    
    t = (gamma / std)
    if not is_sv:
        t = t.view(-1, 1, 1, 1)        
    else:
        t = t.view(-1, 1, 1, 1, 1, 1)        
        running_mean = running_mean.view(1, -1, 1, 1) 
        std = std.view(1, -1, 1, 1) 
        gamma = gamma.view(1, -1, 1, 1) 
        beta = beta.view(1, -1, 1, 1)
    K = kernel * t
    B = (bias - running_mean) / std * gamma + beta
    return K.detach(), B.detach()


def fuse_convs(ks, bs):
    device = ks[0].get_device()
    if device < 0:
        device = None
    K = deconv2d(ks[-1], ks[-2::-1])
    B = bs[0]
    for k, b in zip(ks[1:], bs[1:]):
        B = B.view(1, -1, 1, 1) * torch.ones(1, k.shape[1], k.shape[2], k.shape[3], device=device)
        B = F.conv2d(B, k)
        B = B + b.view(1, -1, 1, 1)
    B = B.view(-1)
    return K.detach(), B.detach()


def fuse_svconv(ks, bs):
    device = ks[0].get_device()
    if device < 0:
        device = None
    B = bs[0]
    K = svdeconv2d(ks[-1], ks[-2::-1])
    for k, b in zip(ks[1:], bs[1:]):
        if B.shape[-1] == 1:
            out_size = k.shape[-1] + k.shape[-3] - 1
            B = B.expand(-1, -1, out_size, out_size)
        B = svconv2d(B, [k])
        B = B + b
    return K.detach(), B.detach()


def get_conv2d(in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1, bias=True, **_):
    if padding is None:
        padding = kernel_size // 2    
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, 
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1, **_):
    result = nn.Sequential()
    result.add_module('conv', get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding=None, dilation=1, groups=1):
    result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding, dilation=dilation, groups=groups)
    result.add_module('nonlinear', nn.ReLU())
    return result


def dwsconv_bn_a1(in_channels, out_channels, kernel_size, stride, padding=None, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()

    result.add_module('dw', nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=False))
    result.add_module('pw', nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    result.add_module('nonlinear', nn.ReLU())
    return result


def dwsconv_bn_a1_infer(in_channels, out_channels, kernel_size, stride, padding=None, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()

    result.add_module('dw', nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=False))
    result.add_module('pw', nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True))
    result.add_module('nonlinear', nn.ReLU())
    return result


def dwsconv_bn_a1_prelu(in_channels, out_channels, kernel_size, stride, padding=None, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()

    result.add_module('dw', nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=False))
    result.add_module('pw', nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    result.add_module('nonlinear', nn.PReLU(out_channels))
    return result


def cal_equivalent_kernel_size(kernel_sizes, dilations=None):
    ks = np.array(kernel_sizes)
    if dilations is None: dilations = [1] * len(kernel_sizes)
    ds = np.array(dilations)
    
    eq_kernel_size = ((ks - 1) * (ds - 1) + ks - 1).sum() + 1
    return eq_kernel_size


class SVConv2d(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if bias is not None else None
        pad = weight.shape[2] // 2
        self.padding = (pad, pad, pad, pad)
    
    def forward(self, x):
        out = svconv2d(F.pad(x, self.padding), [self.weight])
        if self.bias is not None:
            out = out + self.bias
        return out


class SVConvTranspose2d(nn.Module):
    def __init__(self, weight, bias=None, keep_size=True):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if bias is not None else None
        self.keep_size = keep_size
    
    def forward(self, x):
        out = svconv_transpose2d(x, self.weight, self.keep_size)
        if self.bias is not None:
            out = out + self.bias
        return out


class Conv2d(nn.Module):
    def __init__(self, weight, bias=None) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if bias is not None else None
        pad = weight.shape[2] // 2
        self.padding = pad

    def forward(self, x):
        out = F.conv2d(x, weight=self.weight, bias=self.bias, padding=self.padding)
        return out


class OutPSF(nn.Module):
    def __init__(self, out_psfs, bias=None):
        super().__init__()
        self.register_buffer('out_psfs', out_psfs, persistent=False)
        self.bias = nn.Parameter(bias) if bias is not None else None
    
    def forward(self, x):
        out = out_psf_forward(x, self.out_psfs)
        out = torch.flip(out, [-1, -2])
        if self.bias is not None:
            out = out + self.bias
        return out


def generate_window_function(window_type, kernel_size):
    w = signal.get_window(window_type, kernel_size, fftbins=False)
    w_2d = np.outer(w, w)
    # plt.imshow(w_2d)
    # plt.show()    
    window_weight = torch.tensor(w_2d, dtype=torch.float32)
    return window_weight


class FactorizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, small_kernel=False, aux_kernel_sizes=None, window_type=None, normalize_weight=False, **_) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        # self.stride = stride
        self.num_kernels = (kernel_size - 1) // 2
        self.weight_factors = nn.ParameterList()
        for _ in range(self.num_kernels - 1):
            self.weight_factors.append(nn.Parameter(torch.empty((out_channels, out_channels, 3, 3))))
        self.weight_factors.append(nn.Parameter(torch.empty((out_channels, in_channels, 3, 3))))        
        self.small_kernel = nn.Parameter(torch.empty((out_channels, in_channels, 3, 3))) if small_kernel else None        
        self.aux_kernels = nn.ParameterList() if aux_kernel_sizes is not None else None
        if aux_kernel_sizes:
            for ks in aux_kernel_sizes:
                self.aux_kernels.append(nn.Parameter(torch.empty((out_channels, in_channels, ks, ks))))     
        self.aux_kernel_sizes = aux_kernel_sizes
        self.window_type = window_type
        if window_type:
            window_weight = generate_window_function(window_type, kernel_size)
            self.register_buffer('window_weight', window_weight, persistent=False)
        self.normalize_weight = normalize_weight
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weight_factors:
            nn.init.kaiming_uniform_(weight, a=5**0.5)
        if self.small_kernel is not None: nn.init.kaiming_uniform_(self.small_kernel, a=5**0.5)
        if self.aux_kernels is not None: 
            for aux_kernel in self.aux_kernels:
                nn.init.kaiming_uniform_(aux_kernel, a=5**0.5)

    @property
    def weight(self):
        _weight = self.weight_factors[0]
        for factor in self.weight_factors[1:]:
            _weight = F.conv_transpose2d(_weight, factor)
        if self.small_kernel is not None:
            pad = (self.kernel_size - 3) // 2
            _weight = _weight + F.pad(self.small_kernel, pad=[pad, pad, pad, pad])
        if self.aux_kernels is not None:
            for ks, aux_kernel in zip(self.aux_kernel_sizes, self.aux_kernels):
                pad = (self.kernel_size - ks) // 2
                assert pad >= 0
                _weight = _weight + F.pad(aux_kernel, pad=[pad, pad, pad, pad])
        if self.window_type:
            _weight = self.window_weight * _weight # broadcasting

        # normalize weight such that positive and negative parts of the weights sum to 1 and -1
        if self.normalize_weight:
            _weight_p = torch.clamp(_weight, min=0)
            _weight_n = torch.clamp(_weight, max=0)
            _weight_p = _weight_p / _weight_p.sum(dim=(-1,-2), keepdim=True)
            _weight_n = _weight_n / (- _weight_n.sum(dim=(-1,-2), keepdim=True))
            _weight = _weight_p + _weight_n
            
        return _weight
        
    def forward(self, x) -> torch.Tensor:
        weight = self.weight
        x = F.conv2d(x, weight, bias=None, stride=1, padding=self.padding)
        return x
    
    def extra_repr(self) -> str:
        info = f'window_type: {self.window_type}; normalize_weight: {self.normalize_weight}\n'
        return info + super().extra_repr()


class FactorizedTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 mid_channels=None, small_kernel=False, aux_kernel_sizes=None, 
                 keep_size=False, groups=1, normalize_weight=False, **_) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.keep_size = keep_size
        self.groups = groups
        # self.stride = stride
        self.num_kernels = (kernel_size - 1) // 2
        self.weight_factors = nn.ParameterList()
        
        if mid_channels is None:
            mid_channels = [out_channels] * (self.num_kernels - 1)
        elif np.isscalar(mid_channels):
            mid_channels = [mid_channels] * (self.num_kernels - 1)
        assert len(mid_channels) == (self.num_kernels - 1)
        
        if self.num_kernels > 1:
            self.weight_factors.append(nn.Parameter(torch.empty((in_channels, mid_channels[0], 3, 3))))
            for i in range(self.num_kernels - 2):
                self.weight_factors.append(nn.Parameter(torch.empty((mid_channels[i], mid_channels[i+1], 3, 3))))
            self.weight_factors.append(nn.Parameter(torch.empty((mid_channels[-1], out_channels // groups, 3, 3))))
        else:
            self.weight_factors.append(nn.Parameter(torch.empty((in_channels, out_channels // groups, 3, 3))))
        
        self.small_kernel = nn.Parameter(torch.empty((in_channels, out_channels // groups, 3, 3))) if small_kernel else None
        self.aux_kernels = nn.ParameterList() if aux_kernel_sizes is not None else None
        if aux_kernel_sizes:
            for ks in aux_kernel_sizes:
                self.aux_kernels.append(nn.Parameter(torch.empty((in_channels, out_channels // groups, ks, ks))))
        self.aux_kernel_sizes = aux_kernel_sizes
        self.normalize_weight = normalize_weight
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weight_factors:
            nn.init.kaiming_uniform_(weight, a=5**0.5)
        if self.small_kernel is not None: nn.init.kaiming_uniform_(self.small_kernel, a=5**0.5)
        if self.aux_kernels is not None: 
            for aux_kernel in self.aux_kernels:
                nn.init.kaiming_uniform_(aux_kernel, a=5**0.5)        

    @property
    def weight(self):
        _weight = self.weight_factors[0]
        for factor in self.weight_factors[1:]:
            _weight = F.conv_transpose2d(_weight, factor)
        if self.small_kernel is not None:
            pad = (self.kernel_size - 3) // 2
            _weight = _weight + F.pad(self.small_kernel, pad=[pad, pad, pad, pad])
        if self.aux_kernels is not None:
            for ks, aux_kernel in zip(self.aux_kernel_sizes, self.aux_kernels):
                pad = (self.kernel_size - ks) // 2
                assert pad >= 0
                _weight = _weight + F.pad(aux_kernel, pad=[pad, pad, pad, pad])
            
        # normalize weight such that positive and negative parts of the weights sum to 1 and -1
        if self.normalize_weight:
            _weight_p = torch.clamp(_weight, min=0)
            _weight_n = torch.clamp(_weight, max=0)
            _weight_p = _weight_p / _weight_p.sum(dim=(-1,-2), keepdim=True)
            _weight_n = _weight_n / (- _weight_n.sum(dim=(-1,-2), keepdim=True))
            _weight = _weight_p + _weight_n

        return _weight
        
    def forward(self, x) -> torch.Tensor:
        weight = self.weight
        x = F.conv_transpose2d(x, weight, bias=None, stride=1, padding=0, groups=self.groups)
        if self.keep_size:
            pad = self.kernel_size // 2
            x = x[..., pad:-pad, pad:-pad]
        return x
    
    def extra_repr(self) -> str:
        info = f'keep_size: {self.keep_size:}; normalize_weight: {self.normalize_weight}\n'
        return info+super().extra_repr()


# Low Rank Spatially Varying Convolution
class LRSVConv(nn.Module):
    def __init__(self, img_size=32, in_channels=1, out_channels=64, kernel_size=3, kernel_sizes=None, stride=1, padding=None, 
                 bn=True, act_type='none', basis_conv_type='conv', 
                 kernel_rank=3, kernel_weight_type='full', svbias_type='none', **kwargs):
        super().__init__()
        padding = padding if padding is not None else kernel_size // 2
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.input_size = img_size
        self.kernel_rank = kernel_rank        
        if kernel_sizes is not None:
            kernel_size = cal_equivalent_kernel_size(kernel_sizes)
        self.feature_size = (img_size - kernel_size + 2 * padding) // stride + 1
        self.kernel_weight_type = kernel_weight_type
        self.svbias_type = svbias_type
        if basis_conv_type == 'conv':
            self.basis_conv = nn.Conv2d(in_channels, out_channels*kernel_rank, kernel_size, stride, padding=padding, bias=False)
        elif basis_conv_type == 'fconv':
            self.basis_conv = FactorizedConv2d(in_channels, out_channels*kernel_rank, kernel_size, small_kernel=True)

        if kernel_weight_type == 'full':
            self.kernel_weight = nn.Parameter(torch.rand(kernel_rank-1, self.feature_size, self.feature_size))
        elif kernel_weight_type == 'lr':
            self.kernel_weight = nn.ParameterDict({
                'H': nn.Parameter(torch.rand(kernel_rank-1, 1, self.feature_size)),
                'W': nn.Parameter(torch.rand(kernel_rank-1, self.feature_size, 1))
            })
        elif kernel_weight_type.startswith('x'):
            sf = int(kernel_weight_type[1:])
            self.kernel_weight = nn.Parameter(torch.rand(1, kernel_rank-1, self.feature_size // sf, self.feature_size // sf))
        elif kernel_weight_type.startswith('lrx'):
            sf = int(kernel_weight_type[3:])
            self.kernel_weight = nn.ParameterDict({
                'H': nn.Parameter(torch.rand(1, kernel_rank-1, 1, self.feature_size // sf)),
                'W': nn.Parameter(torch.rand(1, kernel_rank-1, self.feature_size // sf, 1))
            })
        elif kernel_weight_type.startswith('p'): # circular symmetric polynomial fit
            degree = kwargs['degree'] if 'degree' in kwargs else int(kernel_weight_type[1:])
            self.kernel_weight = nn.Parameter(torch.rand(kernel_rank-1, degree+1, 1, 1))
            AFOV = kwargs['AFOV'] if 'AFOV' in kwargs else None
            cord, power_series = generate_power_series(self.feature_size, degree, AFOV=AFOV)
            self.register_buffer('cord', cord)
            self.register_buffer('power_series', power_series)
        else:
            raise NotImplementedError

        if svbias_type == 'full':
            self.basis_bias = nn.Parameter(torch.zeros(1, self.out_channels, self.feature_size, self.feature_size))
        elif svbias_type == 'lr':
            self.basis_bias = nn.ParameterDict({
                'H': nn.Parameter(torch.zeros(1, self.feature_size, 1)),
                'W': nn.Parameter(torch.zeros(1, 1, self.feature_size)),
                'C': nn.Parameter(torch.zeros(1, self.out_channels, 1))
            })
        elif svbias_type.startswith('x'):
            sf = int(svbias_type[1:])
            self.basis_bias = nn.Parameter(torch.rand(1, self.out_channels, self.feature_size // sf, self.feature_size // sf))
        elif svbias_type.startswith('lrx'):
            sf = int(svbias_type[3:])
            self.basis_bias = nn.ParameterDict({
                'H': nn.Parameter(torch.zeros(1, 1, self.feature_size // sf, 1)),
                'W': nn.Parameter(torch.zeros(1, 1, 1, self.feature_size // sf)),
                'C': nn.Parameter(torch.zeros(1, self.out_channels, 1, 1))
            })
        elif svbias_type == 'uniform':
            self.basis_bias = nn.Parameter(torch.rand(1, self.out_channels, 1, 1))
        elif svbias_type == 'none':
            self.basis_bias = None
        else:
            raise NotImplementedError
        
        self.norm = nn.BatchNorm2d(self.out_channels) if bn else None
        if act_type == 'none':
            self.act = None
        elif act_type == 'relu':
            self.act = nn.ReLU()
        else:
            raise NotImplementedError

    def forward(self, x):
        B = x.size()[0]
        x = self.basis_conv(x)
        
        if self.kernel_weight_type.startswith(('x', 'lrx')):
            _, _, H, W = x.shape
            sv_weight, bias = self.cal_weight_bias((H, W))
            x = x.view(B, self.kernel_rank, self.out_channels, H, W)
        else:
            sv_weight, bias = self.cal_weight_bias()
            x = x.view(B, self.kernel_rank, self.out_channels, self.feature_size, self.feature_size)

        x = (sv_weight * x).sum(1)
        self._sv_weight = sv_weight
        if bias is not None:
            x = x + bias
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x

    def cal_weight_bias(self, feature_size=None):
        H, W = feature_size if feature_size is not None else (self.feature_size, self.feature_size)
        kernel_weight_type = self.kernel_weight_type
        svbias_type = self.svbias_type
        
        if kernel_weight_type == 'full':
            sv_weight = self.kernel_weight
        elif kernel_weight_type == 'lr':
            sv_weight = (self.kernel_weight['H'] + self.kernel_weight['W'])
        elif kernel_weight_type.startswith('x'):
            sv_weight = F.interpolate(self.kernel_weight, size=(H, W), mode='bicubic', align_corners=True)
        elif kernel_weight_type.startswith('lrx'):
            sv_weight = (self.kernel_weight['H'] + self.kernel_weight['W'])
            sv_weight = F.interpolate(sv_weight, size=(H, W), mode='bicubic', align_corners=True)
        elif kernel_weight_type.startswith('p'):
            sv_weight = (self.power_series * self.kernel_weight).sum(dim=1)[None]

        sv_weight = sv_weight.view(1, self.kernel_rank-1, 1, H, W)
        sv_weight = torch.cat([sv_weight, 1 - sv_weight.sum(dim=1, keepdim=True)], dim=1)  # Linear sum to 1
        
        # sv_weight = sv_weight.view(1, self.kernel_rank, 1, H, W)
        # sv_weight = F.softmax(sv_weight, dim=1)

        if svbias_type == 'full':
            bias = self.basis_bias
        elif svbias_type == 'lr':
            bias = (self.basis_bias['H'] + self.basis_bias['W']).view(1, 1, self.feature_size*self.feature_size)
            bias = bias + self.basis_bias['C']
            bias = bias.view(1, -1, self.feature_size, self.feature_size)
        elif svbias_type.startswith('x'):
            bias = F.interpolate(self.basis_bias, size=(H, W), mode='bilinear', align_corners=True)
        elif svbias_type.startswith('lrx'):
            bias = (self.basis_bias['H'] + self.basis_bias['W'] + self.basis_bias['C'])
            bias = F.interpolate(bias, size=(H, W), mode='bilinear', align_corners=True)
        elif svbias_type == 'uniform':
            bias = self.basis_bias
        elif svbias_type == 'none':
            bias = None
        
        return sv_weight, bias

    def generate_svkernels(self):
        kernels = self.basis_conv.weight
        sv_weight, svbias = self.cal_weight_bias()
        kernels_ = kernels.view(self.kernel_rank, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, 1, 1)
        sv_weight_ = sv_weight.view(self.kernel_rank, 1, 1, 1, 1, self.feature_size, self.feature_size)
        svkernels = (kernels_ * sv_weight_).sum(dim=0)
        if self.norm is not None:
            svkernels, svbias = fuse_bn(svkernels, svbias, self.norm, is_sv=True)        
        return svkernels, svbias

    def extra_repr(self):
        return f"input_size={self.input_size}, feature_size={self.feature_size}, in_ch={self.in_channels}, out_ch={self.out_channels}, rank={self.kernel_rank}, kernel_weight_type={self.kernel_weight_type}, svbias_type={self.svbias_type}"

    def no_weight_decay(self):
        return ('kernel_weight', 'basis_bias')


def generate_power_series(img_size, degree, AFOV=None):
    cord_x = torch.arange(img_size) - (img_size - 1) / 2.
    xx, yy = torch.meshgrid([cord_x, cord_x], indexing='ij')
    rr = (xx ** 2 + yy ** 2) ** 0.5
    R = rr.max()
    cord_r = rr / R
    if AFOV is None:
        power_series = torch.pow(cord_r[None], torch.arange(degree+1).view(degree+1, 1, 1))[None]  # 1 x degree+1 x H x W
        cord = cord_r
    else:
        theta = AFOV / 2 / 180 * np.pi
        z = 1 / np.tan(theta)
        thetas = np.arctan(cord_r / z)
        power_series = torch.pow(thetas[None], torch.arange(degree+1).view(degree+1, 1, 1))[None]  # 1 x degree+1 x H x W
        cord = thetas
    
    return cord, power_series


class LRSVTConv(nn.Module):
    def __init__(self, img_size=32, in_channels=1, out_channels=64, kernel_size=3, mid_channels=None,
                 kernel_rank=3, kernel_weight_type='full', svbias_type='none', sv_softmax=False, sv_share=True, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.input_size = img_size
        self.kernel_rank = kernel_rank
        self.kernel_weight_type = kernel_weight_type
        self.svbias_type = svbias_type
        self.Csv = Csv = 1 if sv_share else out_channels
        self.sv_share = sv_share
        self.sv_softmax = sv_softmax  # use softmax to compute sv_weight
        self.basis_conv = FactorizedTConv2d(
            in_channels*kernel_rank*Csv, out_channels, kernel_size, 
            mid_channels=mid_channels, keep_size=True, groups=Csv, **kwargs)
        self._init_kernel_svweight(kernel_weight_type=kernel_weight_type, **kwargs)
        self._init_svbias(svbias_type)
        
    def _init_kernel_svweight(self, kernel_weight_type, **kwargs):
        R = self.kernel_rank if self.sv_softmax else self.kernel_rank - 1
        C = self.Csv
        if kernel_weight_type == 'full':
            self.kernel_weight = nn.Parameter(torch.rand(C, R, self.input_size, self.input_size))
        elif kernel_weight_type.startswith('x'):
            sf = int(kernel_weight_type[1:])
            self.kernel_weight = nn.Parameter(torch.rand(C, R, self.input_size // sf, self.input_size // sf))
        elif kernel_weight_type.startswith('p'): # circular symmetric polynomial fit
            degree = kwargs['degree'] if 'degree' in kwargs else int(kernel_weight_type[1:])
            self.kernel_weight = nn.Parameter(torch.rand(C, R, degree+1, 1, 1))
            AFOV = kwargs['AFOV'] if 'AFOV' in kwargs else None
            cord, power_series = generate_power_series(self.input_size, degree, AFOV=AFOV)
            self.register_buffer('cord', cord)
            self.register_buffer('power_series', power_series)
        elif kernel_weight_type.startswith('+p'): # circular symmetric polynomial fit + x4
            sf = 4
            degree = int(kernel_weight_type[2:])
            self.kernel_weight = nn.ParameterDict({
                'P': nn.Parameter(torch.rand(C, R, degree+1, 1, 1)),
                'X': nn.Parameter(torch.rand(C, R, self.input_size // sf, self.input_size // sf))
            })
            cord_r, power_series = generate_power_series(self.input_size, degree)
            self.register_buffer('cord_r', cord_r)
            self.register_buffer('power_series', power_series)
        else:
            raise NotImplementedError

    def _init_svbias(self, svbias_type):
        if svbias_type == 'full':
            self.basis_bias = nn.Parameter(torch.zeros(1, self.out_channels, self.input_size, self.input_size))
        elif svbias_type.startswith('x'):
            sf = int(svbias_type[1:])
            self.basis_bias = nn.Parameter(torch.zeros(1, self.out_channels, self.input_size // sf, self.input_size // sf))
        elif svbias_type == 'uniform':
            self.basis_bias = nn.Parameter(torch.zeros(1, self.out_channels, 1, 1))
        elif svbias_type == 'none':
            self.basis_bias = None
        else:
            raise NotImplementedError

    def forward(self, x):
        B, C, H, W = x.shape
        Csv = self.Csv
        sv_weight, bias = self.cal_weight_bias((H, W))
        self._sv_weight = sv_weight[:, :, 0, ...]           # Csv x R x H x W     
        inputs = (x.view(B, 1, 1, C, H, W) * sv_weight)        # B 1 1 C H W  x  Csv R 1 H W = B Csv R C H W
        inputs = inputs.view(B, Csv*self.kernel_rank*C, H, W)     # B Csv*R*C H W
        outputs = self.basis_conv(inputs)
        if bias is not None:
            outputs = outputs + bias
        return outputs

    def cal_weight_bias(self, input_size=None):
        R = self.kernel_rank if self.sv_softmax else self.kernel_rank - 1
        C = self.Csv
        H, W = input_size if input_size is not None else (self.input_size, self.input_size)
        kernel_weight_type = self.kernel_weight_type
        svbias_type = self.svbias_type
        
        if kernel_weight_type == 'full':
            sv_weight = self.kernel_weight
        elif kernel_weight_type.startswith('x'):
            sv_weight = F.interpolate(self.kernel_weight, size=(H, W), mode='bicubic', align_corners=True)
        elif kernel_weight_type.startswith('p'):
            sv_weight = (self.power_series * self.kernel_weight)
            sv_weight = sv_weight.sum(dim=-3)
        elif kernel_weight_type.startswith('+p'):
            sv_weight_circular = (self.power_series * self.kernel_weight['P']).sum(dim=-3)
            sv_weight_free = F.interpolate(self.kernel_weight['X'], size=(H, W), mode='bicubic', align_corners=True)
            sv_weight = sv_weight_circular + sv_weight_free

        sv_weight = sv_weight.view(C, R, 1, H, W)
        if self.sv_softmax:
            sv_weight = F.softmax(sv_weight, dim=1)
        else:
            sv_weight = torch.cat([sv_weight, 1 - sv_weight.sum(dim=1, keepdim=True)], dim=1)  # Linear sum to 1
        
        # _, axes = plt.subplots(1, self.power_series.shape[1], figsize=(10, 5))
        # for i in range(self.power_series.shape[1]):
        #     axes[i].matshow(self.power_series[0, i, ...].detach().cpu())
        # plt.show()
        
        # _, axes = plt.subplots(1, self.kernel_rank, figsize=(10, 5))
        # for i in range(self.kernel_rank):
        #     sv_weight_ = sv_weight.detach().cpu()
        #     axes[i].matshow(sv_weight_[0, i, 0, ...])
        # plt.show()

        if svbias_type == 'full':
            bias = self.basis_bias
        elif svbias_type.startswith('x'):
            bias = F.interpolate(self.basis_bias, size=(H, W), mode='bilinear', align_corners=True)
        elif svbias_type == 'uniform':
            bias = self.basis_bias
        elif svbias_type == 'none':
            bias = None
        
        return sv_weight, bias

    def extra_repr(self):
        info = f"input_size={self.input_size}, in_ch={self.in_channels}, out_ch={self.out_channels}, rank={self.kernel_rank}\n" + \
            f"kernel_weight_type={self.kernel_weight_type}, svbias_type={self.svbias_type}\n" + \
            f"sv_softmax={self.sv_softmax}; sv_share={self.sv_share}\n"
            
        return info+super().extra_repr()

    def no_weight_decay(self):
        return ('kernel_weight', 'basis_bias')


class NaiveSVConv(nn.Module):
    def __init__(self, img_size=32, in_channels=1, out_channels=25, kernel_size=3, stride=1, padding=None,
                 bn=True, bias=True, act=None, **_):
        super().__init__()
        padding = padding if padding is not None else kernel_size // 2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.input_size = img_size
        self.feature_size = (img_size - kernel_size + 2 * padding) // stride + 1

        self.unfold = nn.Unfold(kernel_size=[kernel_size, kernel_size], padding=padding, stride=stride)

        weight = torch.empty(1, out_channels, in_channels*kernel_size*kernel_size, self.feature_size**2)
        self.weight = nn.Parameter(weight, requires_grad=True)
        self.register_parameter('weight', self.weight)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        
        if bias:
            bias = torch.zeros(1, out_channels, self.feature_size**2)
            self.bias = nn.Parameter(bias, requires_grad=True)
            self.register_parameter('bias', self.bias)
        else:
            self.bias = None
 
        self.norm = nn.BatchNorm2d(self.out_channels) if bn else None
        self.act = act() if act else None
    
    def forward(self, x):
        B = x.size()[0]
        x = self.unfold(x).view(B, 1, -1, self.feature_size**2)
        x = (self.weight * x).sum(2)
        if self.bias is not None:
            x = x + self.bias
        x = x.view(B, -1, self.feature_size, self.feature_size)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x

    def extra_repr(self):
        return f"input_size={self.input_size}, in_ch={self.in_channels}, out_ch={self.out_channels}"

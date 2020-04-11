
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import random
import functools
import operator

import torch
from torch import nn
# import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Function

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d


import torch

class MyDataParallel(torch.nn.DataParallel):
    def __init__(self, model, device_ids):
        super(MyDataParallel, self).__init__(model, device_ids)

    def __getattr__(self, name):
        try:
            return super(MyDataParallel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class UpBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, 1, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_c)
        self.lrelu = nn.LeakyReLU(0.2)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):

        out = self.upsample(x)
        out = self.conv(out)
        out = self.norm(out)
        return self.lrelu(out)

class ResidualUpBlock(nn.Module):
    def __init__(self, in_c, out_c, hw=None):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
        )

        self.convs = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
        )

        self.act = nn.Sequential(
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )



    def forward(self, x):
        out = self.convs(x)
        shortcut = self.shortcut(x)
        out = self.act(out + shortcut)

        return out


class DownBlock(nn.Module):
    def __init__(self, in_c, out_c, hw=None):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, 2, 1, bias=False)
        # self.norm = nn.InstanceNorm2d(out_c, affine=True)
        if hw:
            self.norm = nn.LayerNorm([in_c, hw, hw])
            self.lrelu = nn.LeakyReLU(0.2)

        else:
            self.norm = nn.BatchNorm2d(in_c)

            self.lrelu = nn.ReLU()


    def forward(self, x):
        out = self.norm(x)
        out = self.conv(out)
        return self.lrelu(out)

class ResidualDownBlock(nn.Module):
    def __init__(self, in_c, out_c, hw=None, act=True):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
        )

        self.convs = nn.Sequential(

            nn.Conv2d(in_c, in_c, 3, 1, 1, bias=False),
            nn.LayerNorm([in_c, hw, hw]),
            nn.ReLU(),
            nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
            nn.AvgPool2d(2),
        )

        
        if act:
            self.act = nn.Sequential(
                nn.LayerNorm([out_c, hw//2, hw//2]),
                nn.ReLU()
            )
        else:
            self.act = nn.LayerNorm([out_c, hw//2, hw//2])
        # self.norm = nn.BatchNorm2d(in_c)


    def forward(self, x):
        out = self.convs(x)
        shortcut = self.shortcut(x)

        out = self.act(out + shortcut)

        return out



# Adapted from https://github.com/LynnHo/WGAN-GP-DRAGAN-Celeba-Pytorch
class ResidualGenerator(nn.Module):

    def __init__(self, in_dim, dim=128):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU()
        )

        self.convts = nn.Sequential(
            ResidualUpBlock(dim * 8, dim * 4),
            ResidualUpBlock(dim * 4, dim * 2),
            ResidualUpBlock(dim * 2, dim),
            ResidualUpBlock(dim, dim // 2),
            ResidualUpBlock(dim // 2, dim // 4),
            ResidualUpBlock(dim // 4, dim // 8),
            ResidualUpBlock(dim // 8, dim // 16),
            
            nn.Conv2d(dim // 16, 1, 3, 1, padding=1),
            )

        self.init_weight()

    def forward(self, x):
        b = x.size(0)
        out = self.linear(x)
        out = out.view(b, -1, 4, 4)
        out = self.convts(out)
        return out

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                #print('here')
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

class ResidualDiscriminator(nn.Module):
    def __init__(self, in_dim, dim=128):
        super().__init__()

        self.convs = nn.Sequential( 
            nn.Conv2d(in_dim, dim // 16, 3, 1, 1),
            nn.ReLU(),
            ResidualDownBlock(dim//16, dim//8, 512),
            ResidualDownBlock(dim//8, dim//4, 256),
            ResidualDownBlock(dim//4, dim//2, 128),
            ResidualDownBlock(dim//2, dim, 64),
            ResidualDownBlock(dim, dim * 2, 32),
            ResidualDownBlock(dim * 2, dim * 4, 16),
            ResidualDownBlock(dim * 4, dim * 8, 8, act=False)
        )
            
        self.final = nn.Conv2d(dim * 8, 1, 4)

        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x, return_features=False):
        if return_features:
            y = self.convs(x)
            return y    

        y = self.convs(x)
        y = self.relu(y)
        y = self.final(y)
        y = y.view(-1, 1)
        return y

    

        
    #     self.convs = nn.Sequential(
    #         nn.Conv2d(in_dim, dim // 8, 5, 2, 2),
    #         nn.LeakyReLU(0.2),
    #         DownBlock(dim//8, dim//4, 256),
    #         DownBlock(dim//4, dim//2, 128),
    #         DownBlock(dim//2, dim, 64),
    #         DownBlock(dim, dim * 2, 32),
    #         DownBlock(dim * 2, dim * 4, 16),
    #         DownBlock(dim * 4, dim * 8, 8))

    #     self.output = nn.Conv2d(dim * 8, 1, 4)
            
    #     self.init_weight()

    # def forward(self, x):
    #     y = self.convs(x)
    #     y = self.output(y)
    #     y = y.view(-1)

    #     return y

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
    def features(self, x, output=False):
        y = self.convs(x)
        # if output:
        #     y = self.outpu(y)
        y = y.view(-1)

        return y


class Encoder(nn.Module):

    def __init__(self, in_dim, dim=128):
        super().__init__()


        self.convs = nn.Sequential( 
            nn.Conv2d(in_dim, dim // 16, 3, 1, 1),
            nn.ReLU(),
            ResidualDownBlock(dim//16, dim//8, 512),
            ResidualDownBlock(dim//8, dim//4, 256),
            ResidualDownBlock(dim//4, dim//2, 128),
            ResidualDownBlock(dim//2, dim, 64),
            ResidualDownBlock(dim, dim * 2, 32),
            ResidualDownBlock(dim * 2, dim * 4, 16),
            ResidualDownBlock(dim * 4, dim * 4, 8, act=False)
        )
            
        self.final = nn.Conv2d(dim * 4, 512, 4)

        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        if return_features:
            y = self.convs(x)
            return y    

        y = self.convs(x)
        y = self.relu(y)
        y = self.final(y)
        y = y.view(-1, 512)
        return y        # self.convs = nn.Sequential(
        #     DownBlock(1, dim//8),
        #     DownBlock(dim//8, dim//4),
        #     DownBlock(dim//4, dim//2),
        #     DownBlock(dim//2, dim),
        #     DownBlock(dim, dim * 2),
        #     DownBlock(dim * 2, dim * 4),
        #     DownBlock(dim * 4, dim * 8),

        #     nn.Flatten(),

        #     nn.Linear(dim * 8 * 4 * 4, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
            
        #     nn.Linear(512, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
            
        #     nn.Linear(512, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
            
        #     nn.Linear(512, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
            
        #     nn.Linear(512, 512))


        # self.init_weight()

    def forward(self, x):
        y = self.convs(x)
        y = y.view(-1, 512)
        return y

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)



class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 1, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 1, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 256,
            64: 128 * channel_multiplier,
            128: 64 * channel_multiplier,
            256: 32 * channel_multiplier,
            512: 16 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self, bs):
        device = self.input.input.device

        noises = [torch.randn(bs, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(bs, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
        multiple_latents=False
    ):
        if not input_is_latent:
            if multiple_latents:
                og_shape = styles.shape
                styles = self.style(styles.reshape(-1, 512)).reshape(og_shape)
            else:
                styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if multiple_latents:
            latent = styles
        elif len(styles) < 2:
            inject_index = self.n_latent
            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        # elif input_is_latent:
        #     latent = styles

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)


        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])

        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = skip

        if return_latents:
            return image, latent

        else:
            return image, None


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        # channels = {
        #     4: 512,
        #     8: 512,
        #     16: 512,
        #     32: 512,
        #     64: 256 * channel_multiplier,
        #     128: 128 * channel_multiplier,
        #     256: 64 * channel_multiplier,
        #     512: 32 * channel_multiplier,
        #     1024: 16 * channel_multiplier,
        # }
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 256,
            64: 128 * channel_multiplier,
            128: 64 * channel_multiplier,
            256: 32 * channel_multiplier,
            512: 16 * channel_multiplier,
            # 1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(1, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input, return_features=False):
        out = self.convs(input)
        
        if return_features:
            return out
        
        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out
    
        
# import math
# import random
# import functools
# import operator

# import torch
# from torch import nn
# from torch.nn import functional as F
# from torch.autograd import Function

# from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d


# class PixelNorm(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, input):
#         return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


# def make_kernel(k):
#     k = torch.tensor(k, dtype=torch.float32)

#     if k.ndim == 1:
#         k = k[None, :] * k[:, None]

#     k /= k.sum()

#     return k


# class Upsample(nn.Module):
#     def __init__(self, kernel, factor=2):
#         super().__init__()

#         self.factor = factor
#         kernel = make_kernel(kernel) * (factor ** 2)
#         self.register_buffer('kernel', kernel)

#         p = kernel.shape[0] - factor

#         pad0 = (p + 1) // 2 + factor - 1
#         pad1 = p // 2

#         self.pad = (pad0, pad1)

#     def forward(self, input):
#         out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

#         return out


# class Downsample(nn.Module):
#     def __init__(self, kernel, factor=2):
#         super().__init__()

#         self.factor = factor
#         kernel = make_kernel(kernel)
#         self.register_buffer('kernel', kernel)

#         p = kernel.shape[0] - factor

#         pad0 = (p + 1) // 2
#         pad1 = p // 2

#         self.pad = (pad0, pad1)

#     def forward(self, input):
#         out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

#         return out


# class Blur(nn.Module):
#     def __init__(self, kernel, pad, upsample_factor=1):
#         super().__init__()

#         kernel = make_kernel(kernel)

#         if upsample_factor > 1:
#             kernel = kernel * (upsample_factor ** 2)

#         self.register_buffer('kernel', kernel)

#         self.pad = pad

#     def forward(self, input):
#         out = upfirdn2d(input, self.kernel, pad=self.pad)

#         return out


# class EqualConv2d(nn.Module):
#     def __init__(
#         self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
#     ):
#         super().__init__()

#         self.weight = nn.Parameter(
#             torch.randn(out_channel, in_channel, kernel_size, kernel_size)
#         )
#         self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

#         self.stride = stride
#         self.padding = padding

#         if bias:
#             self.bias = nn.Parameter(torch.zeros(out_channel))

#         else:
#             self.bias = None

#     def forward(self, input):
#         out = F.conv2d(
#             input,
#             self.weight * self.scale,
#             bias=self.bias,
#             stride=self.stride,
#             padding=self.padding,
#         )

#         return out

#     def __repr__(self):
#         return (
#             f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
#             f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
#         )


# class EqualLinear(nn.Module):
#     def __init__(
#         self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
#     ):
#         super().__init__()

#         self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

#         if bias:
#             self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

#         else:
#             self.bias = None

#         self.activation = activation

#         self.scale = (1 / math.sqrt(in_dim)) * lr_mul
#         self.lr_mul = lr_mul

#     def forward(self, input):
#         if self.activation:
#             out = F.linear(input, self.weight * self.scale)
#             out = fused_leaky_relu(out, self.bias * self.lr_mul)

#         else:
#             out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

#         return out

#     def __repr__(self):
#         return (
#             f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
#         )


# class ScaledLeakyReLU(nn.Module):
#     def __init__(self, negative_slope=0.2):
#         super().__init__()

#         self.negative_slope = negative_slope

#     def forward(self, input):
#         out = F.leaky_relu(input, negative_slope=self.negative_slope)

#         return out * math.sqrt(2)


# class ModulatedConv2d(nn.Module):
#     def __init__(
#         self,
#         in_channel,
#         out_channel,
#         kernel_size,
#         style_dim,
#         demodulate=True,
#         upsample=False,
#         downsample=False,
#         blur_kernel=[1, 3, 3, 1],
#     ):
#         super().__init__()

#         self.eps = 1e-8
#         self.kernel_size = kernel_size
#         self.in_channel = in_channel
#         self.out_channel = out_channel
#         self.upsample = upsample
#         self.downsample = downsample

#         if upsample:
#             factor = 2
#             p = (len(blur_kernel) - factor) - (kernel_size - 1)
#             pad0 = (p + 1) // 2 + factor - 1
#             pad1 = p // 2 + 1

#             self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

#         if downsample:
#             factor = 2
#             p = (len(blur_kernel) - factor) + (kernel_size - 1)
#             pad0 = (p + 1) // 2
#             pad1 = p // 2

#             self.blur = Blur(blur_kernel, pad=(pad0, pad1))

#         fan_in = in_channel * kernel_size ** 2
#         self.scale = 1 / math.sqrt(fan_in)
#         self.padding = kernel_size // 2

#         self.weight = nn.Parameter(
#             torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
#         )

#         self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

#         self.demodulate = demodulate

#     def __repr__(self):
#         return (
#             f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
#             f'upsample={self.upsample}, downsample={self.downsample})'
#         )

#     def forward(self, input, style):
#         batch, in_channel, height, width = input.shape

#         style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
#         weight = self.scale * self.weight * style

#         if self.demodulate:
#             demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
#             weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

#         weight = weight.view(
#             batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
#         )

#         if self.upsample:
#             input = input.view(1, batch * in_channel, height, width)
#             weight = weight.view(
#                 batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
#             )
#             weight = weight.transpose(1, 2).reshape(
#                 batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
#             )
#             out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
#             _, _, height, width = out.shape
#             out = out.view(batch, self.out_channel, height, width)
#             out = self.blur(out)

#         elif self.downsample:
#             input = self.blur(input)
#             _, _, height, width = input.shape
#             input = input.view(1, batch * in_channel, height, width)
#             out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
#             _, _, height, width = out.shape
#             out = out.view(batch, self.out_channel, height, width)

#         else:
#             input = input.view(1, batch * in_channel, height, width)
#             out = F.conv2d(input, weight, padding=self.padding, groups=batch)
#             _, _, height, width = out.shape
#             out = out.view(batch, self.out_channel, height, width)

#         return out


# class NoiseInjection(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.weight = nn.Parameter(torch.zeros(1))

#     def forward(self, image, noise=None):
#         if noise is None:
#             batch, _, height, width = image.shape
#             noise = image.new_empty(batch, 1, height, width).normal_()

#         return image + self.weight * noise


# class ConstantInput(nn.Module):
#     def __init__(self, channel, size=4):
#         super().__init__()

#         self.input = nn.Parameter(torch.randn(1, channel, 25, 8))

#     def forward(self, input):
#         batch = input.shape[0]
#         out = self.input.repeat(batch, 1, 1, 1)

#         return out


# class StyledConv(nn.Module):
#     def __init__(
#         self,
#         in_channel,
#         out_channel,
#         kernel_size,
#         style_dim,
#         upsample=False,
#         blur_kernel=[1, 3, 3, 1],
#         demodulate=True,
#     ):
#         super().__init__()

#         self.conv = ModulatedConv2d(
#             in_channel,
#             out_channel,
#             kernel_size,
#             style_dim,
#             upsample=upsample,
#             blur_kernel=blur_kernel,
#             demodulate=demodulate,
#         )

#         self.noise = NoiseInjection()
#         # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
#         # self.activate = ScaledLeakyReLU(0.2)
#         self.activate = FusedLeakyReLU(out_channel)

#     def forward(self, input, style, noise=None):
#         out = self.conv(input, style)
#         out = self.noise(out, noise=noise)
#         # out = out + self.bias
#         out = self.activate(out)

#         return out


# class ToRGB(nn.Module):
#     def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
#         super().__init__()

#         if upsample:
#             self.upsample = Upsample(blur_kernel)

#         self.conv = ModulatedConv2d(in_channel, 1, 1, style_dim, demodulate=False)
#         self.bias = nn.Parameter(torch.zeros(1, 1, 1, 1))

#     def forward(self, input, style, skip=None):
#         out = self.conv(input, style)
#         out = out + self.bias

#         if skip is not None:
#             skip = self.upsample(skip)

#             out = out + skip

#         return out


# class Generator(nn.Module):
#     def __init__(
#         self,
#         size,
#         style_dim,
#         n_mlp,
#         channel_multiplier=2,
#         blur_kernel=[1, 3, 3, 1],
#         lr_mlp=0.01,
#     ):
#         super().__init__()

#         self.size = size

#         self.style_dim = style_dim

#         layers = [PixelNorm()]

#         for i in range(n_mlp):
#             layers.append(
#                 EqualLinear(
#                     style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
#                 )
#             )

#         self.style = nn.Sequential(*layers)
#         # self.channels = {
#         #     4: 512,
#         #     8: 512,
#         #     16: 512,
#         #     32: 512,
#         #     64: 256 * channel_multiplier,
#         #     128: 128 * channel_multiplier,
#         #     256: 64 * channel_multiplier,
#         #     512: 32 * channel_multiplier,
#         #     1024: 16 * channel_multiplier,
#         # }

#         self.channels = {
#             8: 512,
#             16: 512,
#             32: 256,
#             64: 128 * channel_multiplier,
#             128: 64 * channel_multiplier,
#             256: 16 * channel_multiplier,
#         }

#         self.input = ConstantInput(self.channels[8])
#         self.conv1 = StyledConv(
#             self.channels[8], self.channels[8], 3, style_dim, blur_kernel=blur_kernel
#         )
#         self.to_rgb1 = ToRGB(self.channels[8], style_dim, upsample=False)

#         self.log_size = int(math.log(size, 2))
#         self.num_layers = (self.log_size - 3) * 2 + 1

#         self.convs = nn.ModuleList()
#         self.upsamples = nn.ModuleList()
#         self.to_rgbs = nn.ModuleList()
#         self.noises = nn.Module()

#         in_channel = self.channels[8]

#         for layer_idx in range(self.num_layers):
#             res = (layer_idx + 1) // 2
#             shape = [1, 1, 25 * (2 ** res), 8 * (2 ** res)]
#             self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

#         for i in range(self.log_size - 2):
#             out_channel = self.channels[2 ** (i + 3) ]

#             self.convs.append(
#                 StyledConv(
#                     in_channel,
#                     out_channel,
#                     3,
#                     style_dim,
#                     upsample=True,
#                     blur_kernel=blur_kernel,
#                 )
#             )

#             self.convs.append(
#                 StyledConv(
#                     out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
#                 )
#             )

#             self.to_rgbs.append(ToRGB(out_channel, style_dim))

#             in_channel = out_channel

#         self.n_latent = self.log_size * 2 - 2


#     def make_noise(self):
#         device = self.input.input.device

#         noises = [torch.randn(1, 1, 25, 8, device=device)]

#         for i in range(self.log_size - 3):
#             for _ in range(2):
#                 noises.append(torch.randn(1, 1, 50 * (2 ** i), 16 * (2 ** i), device=device))

#         # print(len(noises))
#         return noises

#     def mean_latent(self, n_latent):
#         latent_in = torch.randn(
#             n_latent, self.style_dim, device=self.input.input.device
#         )
#         latent = self.style(latent_in).mean(0, keepdim=True)

#         return latent

#     def get_latent(self, input):
#         return self.style(input)

#     def forward(
#         self,
#         styles,
#         return_latents=False,
#         inject_index=None,
#         truncation=1,
#         truncation_latent=None,
#         input_is_latent=False,
#         noise=None,
#         randomize_noise=False
#     ):
#         if not input_is_latent:
#             styles = [self.style(s) for s in styles]




#         if noise is None:
#             if randomize_noise:
#                 noise = [None] * self.num_layers
#             else:
#                 noise = [getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)]


#         # print(len(noise), "zzzzz")
#         if truncation < 1:
#             style_t = []

#             for style in styles:
#                 style_t.append(
#                     truncation_latent + truncation * (style - truncation_latent)
#                 )

#             styles = style_t

#         if len(styles) < 2:
#             inject_index = self.n_latent

#             if styles[0].ndim < 3:
#                 latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
                
#             else:
#                 latent = styles[0]

#         elif len(styles) == 2:
#             if inject_index is None:
#                 inject_index = random.randint(1, self.n_latent - 1)

#             latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
#             latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

#             latent = torch.cat([latent, latent2], 1)
#         else:
#             latent = torch.cat(styles, 0).unsqueeze(0)

#             # print(latent.shape)

#         out = self.input(latent)
#         out = self.conv1(out, latent[:, 0], noise=noise[0])

#         skip = self.to_rgb1(out, latent[:, 1])

#         i = 1
#         for conv1, conv2, noise1, noise2, to_rgb in zip(
#             self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
#         ):
#             # print(out.shape, noise1.shape)

#             out = conv1(out, latent[:, i], noise=noise1)
#             out = conv2(out, latent[:, i + 1], noise=noise2)
#             skip = to_rgb(out, latent[:, i + 2], skip)

#             i += 2

#         image = skip

#         if return_latents:
#             return image, latent

#         else:
#             return image, None


# class ConvLayer(nn.Sequential):
#     def __init__(
#         self,
#         in_channel,
#         out_channel,
#         kernel_size,
#         downsample=False,
#         blur_kernel=[1, 3, 3, 1],
#         bias=True,
#         activate=True,
#     ):
#         layers = []

#         if downsample:
#             factor = 2
#             p = (len(blur_kernel) - factor) + (kernel_size - 1)
#             pad0 = (p + 1) // 2
#             pad1 = p // 2

#             layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

#             stride = 2
#             self.padding = 0

#         else:
#             stride = 1
#             self.padding = kernel_size // 2

#         layers.append(
#             EqualConv2d(
#                 in_channel,
#                 out_channel,
#                 kernel_size,
#                 padding=self.padding,
#                 stride=stride,
#                 bias=bias and not activate,
#             )
#         )

#         if activate:
#             if bias:
#                 layers.append(FusedLeakyReLU(out_channel))

#             else:
#                 layers.append(ScaledLeakyReLU(0.2))

#         super().__init__(*layers)


# class ResBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
#         super().__init__()

#         self.conv1 = ConvLayer(in_channel, in_channel, 3)
#         self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

#         self.skip = ConvLayer(
#             in_channel, out_channel, 1, downsample=True, activate=False, bias=False
#         )

#     def forward(self, input):
#         out = self.conv1(input)
#         out = self.conv2(out)

#         skip = self.skip(input)
#         out = (out + skip) / math.sqrt(2)

#         return out


# class Discriminator(nn.Module):
#     def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
#         super().__init__()

#         channels = {
#             4: 512,
#             8: 512,
#             16: 512,
#             32: 256,
#             64: 128 * channel_multiplier,
#             128: 64 * channel_multiplier,
#             256: 16 * channel_multiplier,
#         }


#         convs = [ConvLayer(1, channels[size], 1)]

#         log_size = int(math.log(size, 2))

#         in_channel = channels[size]

#         for i in range(log_size, 2, -1):
#             out_channel = channels[2 ** (i - 1)]

#             convs.append(ResBlock(in_channel, out_channel, blur_kernel))

#             in_channel = out_channel

#         self.convs = nn.Sequential(*convs)

#         self.stddev_group = 4
#         self.stddev_feat = 1

#         # self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)

#         self.final_conv = ConvLayer(channels[4] + 1, channels[4], 3)

#         self.final_linear = nn.Sequential(
#             EqualLinear(channels[4] * 12 * 4, channels[4], activation='fused_lrelu'),
#             EqualLinear(channels[4], 1),
#         )

#     def forward(self, input):
#         out = self.convs(input)

#         batch, channel, height, width = out.shape
#         group = min(batch, self.stddev_group)
#         stddev = out.view(
#             group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
#         )
#         stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
#         stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
#         stddev = stddev.repeat(group, 1, height, width)
#         out = torch.cat([out, stddev], 1)

#         out = self.final_conv(out)

#         out = out.view(batch, -1)
#         out = self.final_linear(out)

#         return out

#     def features(self, input):
#         out = self.convs(input)

#         batch, channel, height, width = out.shape
#         group = min(batch, self.stddev_group)
#         stddev = out.view(
#             group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
#         )
#         stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
#         stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
#         stddev = stddev.repeat(group, 1, height, width)
#         out = torch.cat([out, stddev], 1)

#         out = self.final_conv(out)

#         return out

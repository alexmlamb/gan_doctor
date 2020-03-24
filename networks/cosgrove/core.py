"""
MIT License

Copyright (c) 2019 Christopher Beckham
Copyright (c) 2017 Christian Cosgrove

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import torch
from torch import nn
import torch.nn.functional as F
#from .shared.spectral_normalization import SpectralNorm
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm
import numpy as np

channels = 3

class CBN2d(nn.Module):
    def __init__(self, y_dim, bn_f):
        super(CBN2d, self).__init__()
        self.bn = nn.BatchNorm2d(bn_f, affine=False)
        self.scale = nn.Linear(y_dim, bn_f)
        nn.init.xavier_uniform(self.scale.weight.data, 1.0)
        self.shift = nn.Linear(y_dim, bn_f)
        nn.init.xavier_uniform(self.shift.weight.data, 0.)
        # https://github.com/pfnet-research/sngan_projection/blob/13c212a7f751c8f0cfd24bc5f35410a61ecb9a45/source/links/categorical_conditional_batch_normalization.py
        # Basically, they initialise with all ones for scale and all zeros for shift.
        # Though that is basically for a one-hot encoding, and we dont have that.
    def forward(self, x, y):
        scale = self.scale(y)
        scale = scale.view(scale.size(0), scale.size(1), 1, 1)
        shift = self.shift(y)
        shift = shift.view(shift.size(0), shift.size(1), 1, 1)
        return self.bn(x)*scale + shift

class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, spec_norm=False):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        if spec_norm:
            self.spec_norm = SpectralNorm
        else:
            self.spec_norm = lambda x: x

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(True),
                self.spec_norm(self.conv1),
                nn.ReLU(True),
                self.spec_norm(self.conv2)
            )
        else:
            self.model = nn.Sequential(
                nn.ReLU(True),
                self.spec_norm(self.conv1),
                nn.ReLU(True),
                self.spec_norm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
        self.bypass = nn.Sequential()
        if in_channels != out_channels:
            self.bypass = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform(self.bypass.weight.data, np.sqrt(2))
            self.bypass = self.spec_norm(self.bypass)
        if stride != 1:
            self.bypass = nn.Sequential(
                self.bypass,
                nn.AvgPool2d(2, stride=stride, padding=0)
            )


    def forward(self, x):
        return self.model(x) + self.bypass(x)

# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, spec_norm=False):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

        if spec_norm:
            self.spec_norm = SpectralNorm
        else:
            self.spec_norm = lambda x: x

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            self.spec_norm(self.conv1),
            nn.ReLU(True),
            self.spec_norm(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            self.spec_norm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

class Discriminator(nn.Module):
    def __init__(self,
                 nf,
                 input_nc=3,
                 n_out=1,
                 n_classes=0,
                 sigmoid=False,
                 spec_norm=False):
        """
        """
        super(Discriminator, self).__init__()

        if spec_norm:
            self.spec_norm = SpectralNorm
        else:
            self.spec_norm = lambda x : x

        self.model = nn.Sequential(
            FirstResBlockDiscriminator(input_nc, nf,
                                       stride=2, spec_norm=spec_norm),
            ResBlockDiscriminator(nf, nf*2,
                                  stride=2, spec_norm=spec_norm),
            ResBlockDiscriminator(nf*2, nf*4,
                                  stride=2, spec_norm=spec_norm),
            ResBlockDiscriminator(nf*4, nf*8,
                                  stride=2, spec_norm=spec_norm),
            ResBlockDiscriminator(nf*8, nf*8,
                                  spec_norm=spec_norm),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(nf*8, n_out)
        nn.init.xavier_uniform(self.fc.weight.data, 1.)
        self.fc = self.spec_norm(self.fc)

        if n_classes > 0:
            self.cls = nn.Linear(nf*8, n_classes)
            nn.init.xavier_uniform(self.cls.weight.data, 1.)
            self.cls = self.spec_norm(self.cls)
        else:
            self.cls = None

        self.sigmoid = sigmoid

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, x.size(1))
        result = self.fc(x)
        if self.sigmoid:
            result = F.sigmoid(result)
        return result

class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        self.bn = nn.BatchNorm2d(in_channels)
        
        self.relu = nn.ReLU()
        self.ups = nn.Upsample(scale_factor=stride)

        self.bn2 = nn.BatchNorm2d(out_channels)
        
        bypass = []
        if stride != 1:
            bypass.append(nn.Upsample(scale_factor=stride))
            if in_channels != out_channels:
                bypass.append(nn.Conv2d(in_channels, out_channels, 1, 1))
        self.bypass = nn.Sequential(*bypass)

    def forward(self, inp):
        x = self.bn(inp)
        x = self.relu(x)
        x = self.ups(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + self.bypass(inp)

class Generator(nn.Module):
    def __init__(self, nf, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.nf = nf

        self.dense = nn.Linear(self.z_dim, 4 * 4 * nf)
        self.final = nn.Conv2d(nf, channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform(self.dense.weight.data, 1.)
        nn.init.xavier_uniform(self.final.weight.data, 1.)

        self.rbn1 = ResBlockGenerator(nf, nf,
                                      stride=2)
        self.rbn2 = ResBlockGenerator(nf, nf,
                                      stride=2)
        self.rbn3 = ResBlockGenerator(nf, nf,
                                      stride=2)
        self.bn = nn.BatchNorm2d(nf)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()


    def forward(self, z):
        #return self.model(self.dense(z).view(-1, GEN_SIZE, 4, 4))
        x = self.dense(z).view(-1, self.nf, 4, 4)
        x = self.rbn1(x)
        x = self.rbn2(x)
        x = self.rbn3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.final(x)
        x = self.tanh(x)
        return x

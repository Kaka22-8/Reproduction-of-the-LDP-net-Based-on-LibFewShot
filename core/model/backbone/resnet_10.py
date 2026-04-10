# -*- coding: utf-8 -*-
import torch
from torch import nn
import math

def init_layer(layer):
    if isinstance(layer, nn.Conv2d):
        n = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
        layer.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.fill_(1)
        layer.bias.data.fill_(0)

class SimpleBlock(nn.Module):
    def __init__(self, indim, outdim, stride, dilated_rate):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.stride = stride
        self.dilated_rate = dilated_rate

        self.C1 = nn.Conv2d(
            indim,
            outdim,
            kernel_size=3,
            stride=self.stride,
            dilation=self.dilated_rate,
            padding=self.dilated_rate,
            bias=False,
        )
        self.BN1=nn.BatchNorm2d(outdim)
        self.C2 = nn.Conv2d(
            outdim,
            outdim,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.BN2=nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.BN1, self.C2, self.BN2]

        if indim != outdim:
            self.shortcut = nn.Conv2d(
                indim,
                outdim,
                kernel_size=1,
                stride=self.stride,
                bias=False,
            )
            self.BNshortcut = nn.BatchNorm2d(outdim)
            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = "1x1"
        else:
            self.shortcut_type = "identity"

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)

        out = self.C2(out)
        out = self.BN2(out)

        short_out = (
            x if self.shortcut_type == "identity"
            else self.BNshortcut(self.shortcut(x))
        )

        out = out + short_out
        out = self.relu2(out)
        return out

class ResNet(nn.Module):
    def __init__(
            self,
            list_of_out_dims=None,
            list_of_stride=None,
            list_of_dilated_rate=None,
    ):
        super(ResNet, self).__init__()

        if list_of_out_dims is None:
            list_of_out_dims = [64, 128, 256, 512]
        if list_of_stride is None:
            list_of_stride = [1, 2, 2, 2]
        if list_of_dilated_rate is None:
            list_of_dilated_rate = [1, 1, 1, 1]

        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        bn1 = nn.BatchNorm2d(64)
        relu1 = nn.ReLU(inplace=True)
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)

        trunk = [conv1, bn1, relu1, pool1]

        indim = 64
        for i in range(4):
            block = SimpleBlock(
                indim,
                list_of_out_dims[i],
                list_of_stride[i],
                list_of_dilated_rate[i],
            )
            trunk.append(block)
            indim = list_of_out_dims[i]

        self.trunk = nn.Sequential(*trunk)

    def forward(self, x):
        out = self.trunk(x)
        out = torch.mean(out, dim=(2,3))
        return out

def resnet10(**kwargs):
    return ResNet(
        list_of_out_dims=kwargs.get('list_of_out_dims', [64, 128, 256, 512]),
        list_of_stride=kwargs.get('list_of_stride', [1, 2, 2, 2]),
        list_of_dilated_rate=kwargs.get('list_of_dilated_rate', [1, 1, 1, 1]),
    )
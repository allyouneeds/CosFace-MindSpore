# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""model architecture of CosFace"""
import math
import mindspore.numpy as np
import mindspore.nn as nn

from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer, Normal


def conv(inp, oup, k, s, p, bias=False):
    if isinstance(k, int):
        weight_shape = (oup, inp, k, k)
        num_n = k * k
    else:
        weight_shape = (oup, inp, k[0], k[1])
        num_n = k[0] * k[1]
    num_n = num_n * oup
    weight = initializer(Normal(mean=0, sigma=math.sqrt(2. / num_n)), shape=weight_shape, dtype=mstype.float32)
    return nn.Conv2d(inp, oup, kernel_size=k,
                     stride=s, padding=p, weight_init=weight,
                     has_bias=bias, pad_mode="pad")


def dw_conv(inp, oup, k, s, p, bias=False):
    if isinstance(k, int):
        weight_shape = (inp, 1, k, k)
        num_n = k * k
    else:
        weight_shape = (inp, 1, k[0], k[1])
        num_n = k[0] * k[1]
    num_n = num_n * oup
    weight = initializer(Normal(mean=0, sigma=math.sqrt(2. / num_n)), shape=weight_shape, dtype=mstype.float32)
    return nn.Conv2d(inp, oup, kernel_size=k,
                     stride=s, padding=p, weight_init=weight,
                     has_bias=bias, pad_mode="pad", group=inp)


def bn(oup):
    return nn.BatchNorm2d(oup)


class Bottleneck(nn.Cell):

    def __init__(self, inp, oup, stride, expansion):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup

        self.conv1 = conv(inp, inp * expansion, 1, 1, 0)
        self.bn1 = bn(inp * expansion)
        self.prelu1 = nn.PReLU(inp * expansion)
        self.conv2 = dw_conv(inp * expansion, inp * expansion, 3, stride, 1)
        self.bn2 = bn(inp * expansion)
        self.prelu2 = nn.PReLU(inp * expansion)
        self.conv3 = conv(inp * expansion, oup, 1, 1, 0)
        self.bn3 = bn(oup)

        self.conv = nn.SequentialCell(
            self.conv1,
            self.bn1,
            self.prelu1,
            self.conv2,
            self.bn2,
            self.prelu2,
            self.conv3,
            self.bn3,
        )

    def construct(self, x):

        if self.connect:
            output = x + self.conv(x)
        else:
            output = self.conv(x)
        return output


class ConvBlock(nn.Cell):

    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = dw_conv(inp, oup, k, s, p)
        else:
            self.conv = conv(inp, oup, k, s, p)
        self.bn = bn(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            output = x
        else:
            output = self.prelu(x)
        return output


Mobilefacenet_bottleneck_setting = [
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1]
]


class MY_MobileFaceNet(nn.Cell):

    def __init__(self, bottleneck_setting):
        super(MY_MobileFaceNet, self).__init__()
        self.conv1 = ConvBlock(3, 64, 3, 2, 1)
        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)
        self.inplanes = 64
        block = Bottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)
        self.conv2 = ConvBlock(128, 512, 1, 1, 0)
        self.linear7 = ConvBlock(512, 512, 7, 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(512, 128, 1, 1, 0, linear=True)
        self.squeeze = P.Squeeze(axis=(2, 3))

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c
        return nn.SequentialCell(*layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = self.squeeze(x)
        return x


def MobileFaceNet():
    return MY_MobileFaceNet(Mobilefacenet_bottleneck_setting)


class CosMarginProduct(nn.Cell):

    def __init__(self, in_features=128, out_features=200, s=32.0, m=0.50, easy_margin=False):
        super(CosMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        shape = (out_features, in_features)
        self.weight = Parameter(initializer('XavierUniform', shape=shape, dtype=mstype.float32), name='weight')
        self.matmul = P.MatMul(transpose_b=True)
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.l2_norm = P.L2Normalize(axis=1)

    def construct(self, x, label):
        cosine = self.matmul(self.l2_norm(x).astype(np.float16), self.l2_norm(self.weight).astype(np.float16))
        phi = cosine - self.m
        one_hot = self.one_hot(label, phi.shape[1], self.on_value, self.off_value)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class WholeNet(nn.Cell):

    def __init__(self, train_phase=True, num_class=10571, num_s=32.0, num_m=0.40):
        super(WholeNet, self).__init__()
        self.train_phase = train_phase
        self.backbone = MobileFaceNet()
        self.product = CosMarginProduct(out_features=num_class, s=num_s, m=num_m)

    def construct(self, x, y):
        x = self.backbone(x)
        if not self.train_phase:
            output = x
        else:
            output = self.product(x, y)
        return output

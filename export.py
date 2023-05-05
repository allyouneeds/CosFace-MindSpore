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
"""
##############export checkpoint file into mindir models#################
python export.py
"""
import argparse
from src.CosFace import MobileFaceNet
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor, load_checkpoint, load_param_into_net, export

parser = argparse.ArgumentParser(description='reidentification')
parser.add_argument('--ckpt_file', type=str, default='./checkpoint/CosFace_Epoch_98_batsh_128_ckpt_0__32s__040m.ckpt')
parser.add_argument('--file_name', type=str, default='Cosface_mindir')
parser.add_argument('--file_format', type=str, default='MINDIR')

args = parser.parse_args()


class Network(nn.Cell):
    def __init__(self, network):
        super(Network, self).__init__()
        self.network = network

    def construct(self, x):
        output = self.network(x)
        return output


class WholeNet(nn.Cell):
    def __init__(self):
        super(WholeNet, self).__init__()
        self.backbone = MobileFaceNet()

    def construct(self, x):
        x = self.backbone(x)
        return x


def export_cosface():
    """ export cosface """
    model = WholeNet()
    net = Network(model)
    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(net, param_dict)
    my_input = np.random.uniform(0.0, 1.0, size=[1, 3, 112, 112]).astype(np.float32)
    export(net, Tensor(my_input), file_name=args.file_name, file_format=args.file_format)
    print("Model exported successfully, format: ", args.file_format)


if __name__ == '__main__':
    export_cosface()

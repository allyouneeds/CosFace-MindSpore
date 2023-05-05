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
"310 inferential data preprocessing script"

import os
import argparse

from shutil import copyfile
from PIL import Image


parser = argparse.ArgumentParser(description='Face validation')
parser.add_argument('--data_dir', type=str, default='./lfw_align_112', help='images directory path')
parser.add_argument('--result_path', type=str, default='./result_images', help='images save path')
args = parser.parse_args()


class LFW():
    def __init__(self):
        self.flip = True

    def process(self, path):
        imgL = Image.open(path).convert('RGB')
        flip_imgL = imgL.transpose(Image.FLIP_LEFT_RIGHT)

        return flip_imgL

    def __len__(self):
        return len(self.nameLs)


if __name__ == '__main__':
    save_path = os.path.join(args.result_path, "LFW_112")
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    root = os.path.join(args.data_dir, "lfw_align_112")
    paths = os.listdir(root)
    image_list = []
    dataset = LFW()

    for p1 in paths:
        pathDir = os.path.join(root, p1)
        sub_paths = os.listdir(pathDir)
        for p2 in sub_paths:
            pathFile = os.path.join(pathDir, p2)

            target = os.path.join(save_path, p2)
            copyfile(pathFile, target)

            flip = dataset.process(pathFile)
            flip.save(os.path.join(save_path, p2[:-4] + "_flip.jpg"))

            image_list.append(pathFile)

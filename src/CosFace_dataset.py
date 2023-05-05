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
"""the dataset of CosFace"""

import os
import mindspore.dataset as ds
import mindspore.common.dtype as mstype
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2

from PIL import Image


def create_dataset_webface(cfg, rank_size=None, rank_id=None):
    mean = [128.0, 128.0, 128.0]
    std = [128.0, 128.0, 128.0]
    random_horizontal = C.RandomHorizontalFlip()
    normalize_op = C.Normalize(mean=mean, std=std)
    resize_op = C.Resize(cfg.image_resize)
    changeswap_op = C.HWC2CHW()

    transform_img = [resize_op, random_horizontal, normalize_op, changeswap_op]
    type_cast_op = C2.TypeCast(mstype.int32)
    transform_label = [type_cast_op]

    casia_ds = ds.ImageFolderDataset(cfg.webface_dir, decode=True, num_shards=rank_size,
                                     shard_id=rank_id, shuffle=True)
    casia_ds = casia_ds.map(input_columns='image', operations=transform_img, num_parallel_workers=cfg.num_work)
    casia_ds = casia_ds.map(input_columns='label', operations=transform_label, num_parallel_workers=cfg.num_work)
    casia_ds = casia_ds.project(columns=["image", "label"])

    casia_ds = casia_ds.shuffle(buffer_size=10000)
    casia_ds = casia_ds.batch(batch_size=cfg.batch_size, drop_remainder=True)
    casia_ds = casia_ds.repeat(1)

    return casia_ds


class LFW():

    def __init__(self, nameLs, nameRs, flags):
        self.nameLs = nameLs
        self.nameRs = nameRs
        self.flags = flags

    def __getitem__(self, index):
        imgL = Image.open(self.nameLs[index]).convert('RGB')
        flip_imgL = imgL.transpose(Image.FLIP_LEFT_RIGHT)
        imgR = Image.open(self.nameRs[index]).convert('RGB')
        flip_imgR = imgR.transpose(Image.FLIP_LEFT_RIGHT)
        return imgL, flip_imgL, imgR, flip_imgR, self.flags[index]

    def __len__(self):
        return len(self.nameLs)


def create_dataset_lfw(cfg, nameLs, nameRs, flags):
    dataset = LFW(nameLs, nameRs, flags)
    lfw_ds = ds.GeneratorDataset(dataset, ["imageL", "flip_imageL", "imageR", "flip_imageR", "flag"], shuffle=False)
    mean = [128.0, 128.0, 128.0]
    std = [128.0, 128.0, 128.0]
    normalize_op = C.Normalize(mean=mean, std=std)
    changeswap_op = C.HWC2CHW()
    resize_op = C.Resize(cfg.image_resize)
    transform_img = [resize_op, normalize_op, changeswap_op]

    type_cast_op = C2.TypeCast(mstype.int32)
    transform_label = [type_cast_op]

    lfw_ds = lfw_ds.map(input_columns='imageL', operations=transform_img)
    lfw_ds = lfw_ds.map(input_columns='imageR', operations=transform_img)
    lfw_ds = lfw_ds.map(input_columns='flip_imageL', operations=transform_img)
    lfw_ds = lfw_ds.map(input_columns='flip_imageR', operations=transform_img)
    lfw_ds = lfw_ds.map(input_columns='flag', operations=transform_label)

    lfw_ds = lfw_ds.project(columns=["imageL", "flip_imageL", "imageR", "flip_imageR", "flag"])
    lfw_ds = lfw_ds.batch(batch_size=cfg.batch_size, drop_remainder=False)
    lfw_ds = lfw_ds.repeat(1)

    return lfw_ds


def parseList(root):
    with open(os.path.join(root, 'pairs.txt')) as f:
        pairs = f.read().splitlines()[1:]
    folder_name = 'lfw_align_112'
    nameLs = []
    nameRs = []
    flags = []
    folds = []
    for i, p in enumerate(pairs):
        p = p.split('\t')
        if len(p) == 3:
            nameL = os.path.join(root, folder_name, p[0], p[0] + '_' + '{:04}.jpg'.format(int(p[1])))
            nameR = os.path.join(root, folder_name, p[0], p[0] + '_' + '{:04}.jpg'.format(int(p[2])))
            flag = 1
            fold = i // 600
        elif len(p) == 4:
            nameL = os.path.join(root, folder_name, p[0], p[0] + '_' + '{:04}.jpg'.format(int(p[1])))
            nameR = os.path.join(root, folder_name, p[2], p[2] + '_' + '{:04}.jpg'.format(int(p[3])))
            flag = -1
            fold = i // 600
        nameLs.append(nameL)
        nameRs.append(nameR)
        flags.append(flag)
        folds.append(fold)
    return [nameLs, nameRs, folds, flags]

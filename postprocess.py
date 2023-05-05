# Copyright 2022 Huawei Technologies Co., Ltd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
"310 script for reasoning accuracy calculation"

import os
import argparse
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.common.dtype as mstype
import numpy as np
import scipy.io

parser = argparse.ArgumentParser(description='Face validation')
parser.add_argument('--label_dir', type=str)
parser.add_argument('--result_dir', type=str)
args_opt = parser.parse_args()


class LFW():
    def __init__(self, nameLs, nameRs, flags):
        self.nameLs = nameLs
        self.nameRs = nameRs
        self.flags = flags

    def __getitem__(self, index):
        imgL = np.fromfile(self.nameLs[index], np.float32)
        flip_imgL = np.fromfile(self.nameLs[index][:-6] + '_flip' + self.nameLs[index][-6:], np.float32)
        imgR = np.fromfile(self.nameRs[index], np.float32)
        flip_imgR = np.fromfile(self.nameRs[index][:-6] + '_flip' + self.nameRs[index][-6:], np.float32)
        return imgL, flip_imgL, imgR, flip_imgR, self.flags[index]

    def __len__(self):
        return len(self.nameLs)


def create_dataset_lfw(nameLs, nameRs, flags):
    dataset = LFW(nameLs, nameRs, flags)
    lfw_ds = ds.GeneratorDataset(dataset, ["imageL", "flip_imageL", "imageR", "flip_imageR", "flag"], shuffle=False)

    type_cast_op = C2.TypeCast(mstype.int32)
    transform_label = [type_cast_op]

    lfw_ds = lfw_ds.map(input_columns='flag', operations=transform_label)
    lfw_ds = lfw_ds.project(columns=["imageL", "flip_imageL", "imageR", "flip_imageR", "flag"])
    lfw_ds = lfw_ds.batch(batch_size=256, drop_remainder=False)
    lfw_ds = lfw_ds.repeat(1)

    return lfw_ds


def parseList(label_dir, result_dir):
    with open(os.path.join(label_dir, 'pairs.txt')) as f:
        pairs = f.read().splitlines()[1:]
    nameLs = []
    nameRs = []
    flags = []
    folds = []
    for i, p in enumerate(pairs):
        p = p.split('\t')
        if len(p) == 3:
            nameL = os.path.join(result_dir, p[0] + '_' + '{:04}.bin'.format(int(p[1])))
            nameL = nameL[:-4] + '_0' + nameL[-4:]
            nameR = os.path.join(result_dir, p[0] + '_' + '{:04}.bin'.format(int(p[2])))
            nameR = nameR[:-4] + '_0' + nameR[-4:]
            flag = 1
            fold = i // 600
        elif len(p) == 4:
            nameL = os.path.join(result_dir, p[0] + '_' + '{:04}.bin'.format(int(p[1])))
            nameL = nameL[:-4] + '_0' + nameL[-4:]
            nameR = os.path.join(result_dir, p[2] + '_' + '{:04}.bin'.format(int(p[3])))
            nameR = nameR[:-4] + '_0' + nameR[-4:]

            flag = -1
            fold = i // 600
        nameLs.append(nameL)
        nameRs.append(nameR)
        flags.append(flag)
        folds.append(fold)
    return [nameLs, nameRs, folds, flags]


def getAccuracy(scores, flags, threshold):
    p = np.sum(scores[flags == 1] > threshold)
    n = np.sum(scores[flags == -1] < threshold)
    return 1.0 * (p + n) / len(scores)


def getThreshold(scores, flags, thrNum):
    accuracys = np.zeros((2 * thrNum + 1, 1))
    thresholds = np.arange(-thrNum, thrNum + 1) * 1.0 / thrNum
    for i in range(2 * thrNum + 1):
        accuracys[i] = getAccuracy(scores, flags, thresholds[i])

    max_index = np.squeeze(accuracys == np.max(accuracys))
    bestThreshold = np.mean(thresholds[max_index])
    return bestThreshold


def evaluation_10_fold(feature_save_dir):
    ACCs = np.zeros(10)
    result = scipy.io.loadmat(feature_save_dir)
    for i in range(10):
        fold = result['fold']
        flags = result['flag']
        featureLs = result['fl']
        featureRs = result['fr']
        valFold = fold != i
        testFold = fold == i
        flags = np.squeeze(flags)

        mu = np.mean(np.concatenate((featureLs[valFold[0], :], featureRs[valFold[0], :]), 0), 0)
        mu = np.expand_dims(mu, 0)

        featureLs = featureLs - mu
        featureRs = featureRs - mu
        featureLs = featureLs / np.expand_dims(np.sqrt(np.sum(np.power(featureLs, 2), 1)), 1)
        featureRs = featureRs / np.expand_dims(np.sqrt(np.sum(np.power(featureRs, 2), 1)), 1)

        scores = np.sum(np.multiply(featureLs, featureRs), 1)
        threshold = getThreshold(scores[valFold[0]], flags[valFold[0]], 10000)
        ACCs[i] = getAccuracy(scores[testFold[0]], flags[testFold[0]], threshold)

    return ACCs


def getFeatureFromMindspore(label_dir, result_dir, feature_save_dir):
    nameLs, nameRs, folds, flags = parseList(label_dir, result_dir)
    lfw_dataset = create_dataset_lfw(nameLs, nameRs, flags)
    eval_dataset = lfw_dataset.create_tuple_iterator()
    featureLs = None
    featureRs = None

    for IL, FIL, IR, FIR, _ in eval_dataset:
        featureL = IL.asnumpy()
        featureR = IR.asnumpy()
        featureFL = FIL.asnumpy()
        featureFR = FIR.asnumpy()
        featureL = np.concatenate((featureL, featureFL), 1)
        featureR = np.concatenate((featureR, featureFR), 1)
        if featureLs is None:
            featureLs = featureL
        else:
            featureLs = np.concatenate((featureLs, featureL), 0)
        if featureRs is None:
            featureRs = featureR
        else:
            featureRs = np.concatenate((featureRs, featureR), 0)

    result = {'fl': featureLs, 'fr': featureRs, 'fold': folds, 'flag': flags}
    scipy.io.savemat(feature_save_dir, result)


if __name__ == '__main__':
    my_feature_save_dir = os.path.join(args_opt.label_dir, 'rusult.mat')

    getFeatureFromMindspore(args_opt.label_dir, args_opt.result_dir, my_feature_save_dir)
    get_ACCs = evaluation_10_fold(feature_save_dir=my_feature_save_dir)

    print('AVE    {:.2f}'.format(np.mean(get_ACCs) * 100))

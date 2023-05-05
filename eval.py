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
"""eval launch."""

import os
import sys
import argparse
import scipy.io
import numpy as np
import mindspore.common.dtype as mstype

from mindspore import Tensor
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.CosFace import WholeNet
from src.CosFace_dataset import parseList, create_dataset_lfw


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


def getFeatureFromMindspore(input_args, weight_file_dir):
    net = WholeNet(train_phase=False, num_class=input_args.num_class)
    param_dict = load_checkpoint(weight_file_dir)
    load_param_into_net(net, param_dict)

    nameLs, nameRs, folds, flags = parseList(input_args.lfw_path)
    lfw_dataset = create_dataset_lfw(input_args, nameLs, nameRs, flags)
    eval_dataset = lfw_dataset.create_tuple_iterator()
    featureLs = None
    featureRs = None
    net.set_train(False)
    for IL, FIL, IR, FIR, _ in eval_dataset:
        featureL = net(IL, Tensor(1, mstype.int32)).asnumpy()
        featureR = net(IR, Tensor(1, mstype.int32)).asnumpy()
        featureFL = net(FIL, Tensor(1, mstype.int32)).asnumpy()
        featureFR = net(FIR, Tensor(1, mstype.int32)).asnumpy()
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
    scipy.io.savemat(input_args.feature_save_dir, result)
    print("Save the {0:} successfully !".format(input_args.feature_save_dir))


def main(input_args):
    if input_args.enable_pengcheng_cloud:
        import moxing as mox
        data_dir = input_args.workroot + '/data'
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        obs_data_url = input_args.data_url
        mox.file.copy_parallel(obs_data_url, data_dir)
        print("Successfully Download {} to {}".format(obs_data_url, data_dir))

        # This is the path of the default dataset. Please modify it if necessary
        input_args.lfw_path = os.path.join(data_dir, 'LFW_112')
        if input_args.ckpt_path[0] == "." and input_args.ckpt_path[1] == "/":
            input_args.ckpt_path = input_args.ckpt_path[2:]
        elif input_args.ckpt_path[0] == "/":
            input_args.ckpt_path = input_args.ckpt_path[1:]
        else:
            pass
        input_args.ckpt_path = os.path.join(sys.path[0], input_args.ckpt_path)

    context.set_context(mode=context.GRAPH_MODE, device_target=input_args.device_target)
    getFeatureFromMindspore(input_args, input_args.ckpt_path)
    ACCs = evaluation_10_fold(input_args.feature_save_dir)

    print("-" * 40)
    for i in range(len(ACCs)):
        txt_str = ' --{0:}--  {1:.2f}'.format(i + 1, ACCs[i] * 100)
        print(txt_str)
    print("-" * 40)
    print('Test Result,  AVE:  {:.2f}'.format(np.mean(ACCs) * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='reidentification')

    # eval option
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_work', type=int, default=2)
    parser.add_argument('--device_target', type=str, default="Ascend")
    parser.add_argument('--num_class', type=float, default=10572,
                        help="Number of categories for the training dataset")
    parser.add_argument('--image_resize', type=tuple, default=(112, 112),
                        help="Data set picture specified size")

    # url option
    parser.add_argument('--feature_save_dir', type=str, default='./lfw_rusult.mat',
                        help="The address of lfw's feature")
    parser.add_argument('--lfw_path', type=str, default='./data/LFW_112',
                        help="The address of lfw dataset")
    parser.add_argument('--ckpt_path', type=str,
                        default='./checkpoint/CosFace_Epoch_98_batsh_128_ckpt_0__32s__040m.ckpt',
                        help="The absolute address of ckpt")

    # PengCheng cloud brain option
    parser.add_argument('--enable_pengcheng_cloud', type=int, default=0,
                        help="Whether it runs on Pengcheng cloud brain")
    parser.add_argument('--workroot', type=str, default='/home/work/user-job-dir',
                        help="Cloud brain working environment for training tasks")
    parser.add_argument('--train_url', type=str, default=' ',
                        help="Training task result saving address")
    parser.add_argument('--data_url', type=str, default=' ',
                        help="Dataset address of training task")

    my_args = parser.parse_args()
    main(my_args)

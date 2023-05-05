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
"""train launch."""

import os

import time
import argparse
import numpy as np

import mindspore.nn as nn
from mindspore import context
from mindspore import Tensor

from mindspore.train.model import Model

from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.communication.management import init, get_rank, get_group_size

from src.CosFace import WholeNet
from src.create_callback import CosFace_Callback
from src.CosFace_dataset import create_dataset_webface
from src.lr_schedule import get_multi_step_lr, warmup_cosine_annealing_lr

from eval import evaluation_10_fold, getFeatureFromMindspore


class Loss_Network(nn.Cell):

    def __init__(self, network, criterion):
        super(Loss_Network, self).__init__()
        self.network = network
        self.criterion = criterion

    def construct(self, input_data, label):
        output = self.network(input_data, label)
        loss = self.criterion(output, label)
        return loss


def get_best_model(input_args):
    ckpt_old_list = os.listdir(input_args.save_ckpt_dir)
    ckpt_new_list = []
    for ckpt_name in ckpt_old_list:
        type_name = ckpt_name.split(".")[-1]
        if type_name == "ckpt":
            ckpt_new_list.append(ckpt_name)

    best_acc = 0
    best_ckpt_name = " "
    for ckpt_name in ckpt_new_list:
        weight_file = os.path.join(input_args.save_ckpt_dir, ckpt_name)
        getFeatureFromMindspore(input_args, weight_file)
        ACCs = evaluation_10_fold(input_args.feature_save_dir)
        ACCs = np.mean(ACCs)
        if ACCs > best_acc:
            best_acc = ACCs
            best_ckpt_name = ckpt_name
    print("-" * 40)
    print("The best model is {0:}".format(best_ckpt_name))
    print("-" * 40)
    print("The best accuracy is {0:.2f}".format(best_acc * 100))


def get_dynamic_lr(args, train_dataset):
    lr = args.init_lr
    if args.lr_strategy == 'Default':
        lr = args.init_lr
    elif args.lr_strategy == 'Multistep':
        if args.is_distributed:
            lr = get_multi_step_lr(train_dataset.get_dataset_size() * 4, init_lr=args.init_lr, epoch=args.epoch_size)
            lr = Tensor(lr[::4])
        else:
            lr = get_multi_step_lr(train_dataset.get_dataset_size(), init_lr=args.init_lr, epoch=args.epoch_size)
            lr = Tensor(lr)
    elif args.lr_strategy == 'Cosine':
        if args.is_distributed:
            lr = warmup_cosine_annealing_lr(train_dataset.get_dataset_size() * 4, init_lr=args.init_lr,
                                            max_epoch=args.epoch_size)
            lr = Tensor(lr[::4])
        else:
            lr = warmup_cosine_annealing_lr(train_dataset.get_dataset_size(), init_lr=args.init_lr,
                                            max_epoch=args.epoch_size)
            lr = Tensor(lr)
    else:
        raise Exception("Please enter the correct learning rate policy keyword!")

    return lr


def main(args):
    if args.enable_pengcheng_cloud:
        import moxing as mox
        data_dir = args.workroot + '/data'
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        if args.data_is_soft_ink:
            if args.is_distributed:
                pass
            else:
                cmd_download = "wget -O {0:}/train.zip '{1:}'".format(data_dir, args.data_soft_link)
                cmd_unzip = "unzip {0:}/train.zip -d {0:}".format(data_dir)

                print(" Start downloading the training set ! ")
                command__download = os.popen(cmd_download).read()
                for _ in command__download:
                    pass
                print(" Finish downloading the training set ! ")

                print(" Start unzip the training set ! ")
                command__unzip = os.popen(cmd_unzip)
                for _ in command__unzip:
                    pass
                print(" Finish unzip the training set ! ")
        else:
            obs_data_url = args.data_url
            mox.file.copy_parallel(obs_data_url, data_dir)
            print("Successfully Download {} to {}".format(obs_data_url, data_dir))

        # This is the path of the default dataset. Please modify it if necessary
        args.webface_dir = os.path.join(data_dir, "WebFace_122/images")
        args.save_ckpt_dir = args.workroot + '/model'

    if not os.path.exists(args.save_ckpt_dir):
        os.mkdir(args.save_ckpt_dir)

    if args.is_distributed:
        device_id = int(os.getenv('DEVICE_ID'))
        context.set_context(device_id=device_id)
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
        init()
        rank_id = get_rank()
        rank_size = get_group_size()

        if args.data_is_soft_ink and args.enable_pengcheng_cloud:
            txt_flag = os.path.join(args.workroot, "txt_flag.txt")

            # Default: Downloading data and saving model are the same rank_id.
            # If any change is required, please modify the next line
            if rank_id == args.save_ckpt_device:
                cmd_download = "wget -O {0:}/train.zip '{1:}'".format(data_dir, args.data_soft_link)
                cmd_unzip = "unzip {0:}/train.zip -d {0:}".format(data_dir)

                print(" Start downloading the training set ! ")
                command__download = os.popen(cmd_download).read()
                for _ in command__download:
                    pass
                print(" Finish downloading the training set ! ")

                print(" Start unzip the training set ! ")
                command__unzip = os.popen(cmd_unzip)
                for _ in command__unzip:
                    pass
                print(" Finish unzip the training set ! ")
                txt_file = open(txt_flag, 'w')
                txt_file.write("Finish!")
                txt_file.close()
            else:
                while not os.path.exists(txt_flag):
                    time.sleep(2)

        context.set_auto_parallel_context(parallel_mode=context.ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True, parameter_broadcast=True)
        train_dataset = create_dataset_webface(args, rank_size, rank_id)
        my_call = CosFace_Callback(args, train_dataset.get_dataset_size(), rank_id)

    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
        rank_id = args.save_ckpt_device
        train_dataset = create_dataset_webface(args, rank_size=None, rank_id=None)
        my_call = CosFace_Callback(args, train_dataset.get_dataset_size())

    net = WholeNet(num_class=args.num_class, num_s=args.num_s, num_m=args.num_m)

    dynamic_lr = get_dynamic_lr(args, train_dataset)

    group_params = [{'params': net.trainable_params()}]
    opt = nn.SGD(group_params, learning_rate=dynamic_lr, momentum=args.momentum, weight_decay=args.weight_decay,
                 nesterov=True)

    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    train_net = Loss_Network(net, loss)
    model = Model(train_net, optimizer=opt)

    model.train(args.epoch_size, train_dataset, callbacks=[my_call], dataset_sink_mode=(args.dataset_sink_mode == 1))
    context.set_auto_parallel_context(parallel_mode=context.ParallelMode.STAND_ALONE,
                                      gradients_mean=True, parameter_broadcast=False)
    # Optimal model selection only in non cloud brain environment
    if args.select_model_flag and (not args.enable_pengcheng_cloud) and rank_id == args.save_ckpt_device:
        get_best_model(args)

    if args.enable_pengcheng_cloud:
        mox.file.copy_parallel(args.save_ckpt_dir, args.train_url)
        print("Successfully Upload {} to {}".format(args.save_ckpt_dir, args.train_url))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face Recognization')

    # model option
    parser.add_argument('--num_s', type=float, default=32.0,
                        help="the scaling parameter")
    parser.add_argument('--num_m', type=float, default=0.40,
                        help="the cosine margin")
    parser.add_argument('--image_resize', type=tuple, default=(112, 112),
                        help="Data set picture specified size")
    parser.add_argument('--num_class', type=float, default=10572,
                        help="Number of categories for the training dataset")

    # train option
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--init_lr', type=float, default=0.1)
    parser.add_argument('--epoch_size', type=int, default=100)
    parser.add_argument('--num_work', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--lr_strategy', type=str, default="Multistep",
                        choices=['Default', 'Multistep', 'Cosine'],
                        help=" The dynamic learning rate strategy ")
    parser.add_argument('--epoch_star_save', type=int, default=50,
                        help=" Start saving the initial epoch of the model ")
    parser.add_argument('--epoch_per_save', type=int, default=2,
                        help=" The epoch interval to save the model ")

    # device option
    parser.add_argument('--dataset_sink_mode', type=int, default=1)
    parser.add_argument('--device_target', type=str, default="Ascend")
    parser.add_argument('--is_distributed', type=int, default=0,
                        help=" Start distributed training ")
    parser.add_argument('--save_ckpt_device', type=int, default=0,
                        help=" In distributed mode, the id of device to save ckpt model ")

    # url option
    parser.add_argument('--save_ckpt_dir', type=str, default='./model',
                        help=" Absolute address to save the model ")
    parser.add_argument('--webface_dir', type=str, default='./data/WebFace_122/images',
                        help=" The absolute address of image folder ")

    # PengCheng cloud brain option
    parser.add_argument('--enable_pengcheng_cloud', type=int, default=0,
                        help=" Whether it runs on Pengcheng cloud brain ")
    parser.add_argument('--data_is_soft_ink', type=int, default=1,
                        help=" Whether to download the data set in the form of soft link ")
    parser.add_argument('--workroot', type=str, default='/home/work/user-job-dir',
                        help=" Cloud brain working environment for training tasks ")
    parser.add_argument('--train_url', type=str, default=' ',
                        help=" Training task result saving address ")
    parser.add_argument('--data_url', type=str, default=' ',
                        help=" Dataset address of training task ")
    parser.add_argument('--data_soft_link', type=str,
                        default=' ',
                        help=" In the soft link mode, the download address of the dataset ")

    # Get the best model after training
    parser.add_argument('--select_model_flag', type=int, default=1,
                        help="Get the best model after training")
    parser.add_argument('--feature_save_dir', type=str, default='./lfw_rusult.mat',
                        help="The address of lfw's feature")
    parser.add_argument('--lfw_path', type=str, default='./data/LFW_112',
                        help="The address of lfw dataset")

    train_args = parser.parse_args()
    main(train_args)

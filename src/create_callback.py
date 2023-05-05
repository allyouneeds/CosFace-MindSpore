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
"""callback function of CosFace"""

import time
import os
import mindspore
from mindspore.train.callback import Callback


class CosFace_Callback(Callback):

    def __init__(self, cfg, epoch_step, rank_id=0):
        super(CosFace_Callback, self).__init__()
        self.cfg = cfg
        self.rank_id = rank_id
        self.epoch_step = epoch_step
        self.epoch_time_begin = 0
        self.epoch_time_end = 0

    def step_begin(self, run_context):
        if self.cfg.dataset_sink_mode:
            pass
        else:
            self.step_time_begin = time.time()

    def step_end(self, run_context):
        if self.cfg.dataset_sink_mode:
            pass
        else:
            self.step_time_end = time.time()
            step_time = self.step_time_end - self.step_time_begin
            cb_params = run_context.original_args()
            cur_step = cb_params.cur_step_num
            num_step = cb_params.batch_num

            epoch_num = cb_params.cur_epoch_num
            print_step = cur_step % num_step
            if print_step == 0:
                print_step = num_step

            loss = cb_params.net_outputs
            print("epoch time: {0:.3f} h, per step time: {1:.3f} s ".format(step_time * num_step / 3600,
                                                                            step_time))
            print("epoch: {0:} step: {1:}, loss is {2:}".format(epoch_num, cur_step, loss))

    def epoch_begin(self, run_context):
        self.epoch_time_begin = time.time()

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        loss = cb_params.net_outputs
        self.epoch_time_end = time.time()
        epoch_time = self.epoch_time_end - self.epoch_time_begin

        print("epoch time: {0:.3f} min, per step time: {1:.3f} ms ".format(epoch_time / 60,
                                                                           1000 * epoch_time / self.epoch_step))
        print("epoch: {0:} step: {1:}, loss is {2:}".format(epoch_num, self.epoch_step, loss))

        save_ckpt_flag = 0
        epoch_true_flag = epoch_num >= self.cfg.epoch_star_save and epoch_num % self.cfg.epoch_per_save == 0
        if self.cfg.is_distributed:
            if epoch_true_flag and self.rank_id == self.cfg.save_ckpt_device:
                save_ckpt_flag = 1
                ckpt_file_name = "CosFace_Distributed_Epoch_{0:}_.ckpt".format(epoch_num)
            else:
                pass
        else:
            if epoch_true_flag:
                ckpt_file_name = "CosFace_Single_Epoch_{0:}_.ckpt".format(epoch_num)
                save_ckpt_flag = 1
            else:
                pass

        if save_ckpt_flag:
            save_file_path = os.path.join(self.cfg.save_ckpt_dir, ckpt_file_name)
            feature_net = cb_params.train_network.network
            mindspore.save_checkpoint(feature_net, save_file_path)
            print(' Save the {0:} to {1:}'.format(ckpt_file_name, self.cfg.save_ckpt_dir))
        else:
            pass

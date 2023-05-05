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
"""generate dynamic learning rate of CosFace"""

import math
from collections import Counter
import numpy as np


def linear_warmup_lr(current_step, warmup_steps, base_lr, init_lr):
    """Linear learning rate"""
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    lr = float(init_lr) + lr_inc * current_step
    return lr


def warmup_step_lr(steps_per_epoch, lr, lr_epochs, warmup_epochs, max_epoch, gamma=0.1):
    """Linear warm up learning rate"""
    base_lr = lr
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    milestones = lr_epochs
    milestones_steps = []
    for milestone in milestones:
        milestones_step = milestone * steps_per_epoch
        milestones_steps.append(milestones_step)

    lr_each_step = []
    lr = base_lr
    milestones_steps_counter = Counter(milestones_steps)
    for i in range(total_steps):
        if i < warmup_steps:
            lr = linear_warmup_lr(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            lr = lr * gamma ** milestones_steps_counter[i]
        lr_each_step.append(lr)

    return np.array(lr_each_step).astype(np.float32)


def get_multi_step_lr(steps_per_epoch, init_lr=0.1, epoch=100):
    lr = warmup_step_lr(steps_per_epoch,
                        init_lr,
                        [36, 48, 60],
                        0,
                        epoch,
                        0.1,
                        )
    return lr


def warmup_cosine_annealing_lr(steps_per_epoch, init_lr=0.1, warmup_epochs=3, max_epoch=100, global_step=0):
    """
    generate learning rate array with cosine

    Args:
       lr(float): base learning rate
       steps_per_epoch(int): steps size of one epoch
       warmup_epochs(int): number of warmup epochs
       max_epoch(int): total epochs of training
       global_step(int): the current start index of lr array
    Returns:
       np.array, learning rate array
    """
    base_lr = init_lr
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    decay_steps = total_steps - warmup_steps

    lr_each_step = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr = linear_warmup_lr(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            linear_decay = (total_steps - i) / decay_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * 2 * 0.47 * i / decay_steps))
            decayed = linear_decay * cosine_decay + 0.00001
            lr = base_lr * decayed
        lr_each_step.append(lr)

    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[global_step:]
    return learning_rate

#!/bin/bash
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

# bash run_standalone_train_ascend.sh
if [ $# -ne  4 ]; then
     echo "Usage: bash run_stanalone_train_ascend.sh [DEVICE_ID] [image_path] [test_image_path] [save_ckpt_dir]"
exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo $1
    else
        echo "$(realpath -m ${PWD}/$1)"
    fi
}
export DEVICE_ID=$1
PATH1=$(get_real_path $2)
PATH2=$(get_real_path $3)
PATH3=$(get_real_path $4)

cd ..
python train.py --webface_dir $PATH1 --lfw_path $PATH2 --save_ckpt_dir $PATH3 --enable_pengcheng_cloud 0 --is_distributed 0 > train.log 2>&1 &

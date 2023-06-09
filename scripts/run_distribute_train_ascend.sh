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

# bash run_distribute_train_ascend.sh [RANK_TABLE_FILE]

if [ $# -ne  5 ]; then
    echo "Usage: bash run_distribute_train_ascend.sh [DEVICE_NUM] [RANK_TABLE_FILE] [image_path] [test_image_path] [save_ckpt_dir]"
exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo $1
    else
        echo "$(realpath -m ${PWD}/$1)"
    fi
}

PATH2=$(get_real_path $2)
PATH3=$(get_real_path $3)
PATH4=$(get_real_path $4)
PATH5=$(get_real_path $5)

if [ ! -f ${PATH2} ]; then
    echo "error: RANK_TABLE_FILE=$PATH2 is not a file."
exit 1
fi

DEVICE_NUM=$1
if [ $DEVICE_NUM -ne 2 ] && [ $DEVICE_NUM -ne 4 ] && [ $DEVICE_NUM -ne 8 ]; then
  echo "error: DEVICE_NUM=$DEVICE_NUM must be 2/4/8"
  exit 1
fi


cd ..
export RANK_TABLE_FILE=$PATH2

echo 'start training'
for((i=0;i<${DEVICE_NUM};i++));do
    export DEVICE_ID=$i
    export RANK_ID=$i
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp ./*.py ./train_parallel$i
    cp -r ./src ./train_parallel$i
    cd ./train_parallel$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log

    python train.py --webface_dir $PATH3 --lfw_path $PATH4 --save_ckpt_dir $PATH5 --enable_pengcheng_cloud 0 --is_distributed 1 > ./train$i.log  2>&1 &
    cd ../
done


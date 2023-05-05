# 目录

<!-- TOC -->

- [目录](#目录)
- [CosFace描述](#CosFace描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [导出过程](#导出过程)
        - [导出](#导出)
    - [推理过程](#推理过程)
        - [推理](#推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
            - [CASIA-WebFace数据集上的CosFace](#CASIA-WebFace数据集上的CosFace)
        - [评估性能](#评估性能)
            - [LFW数据集上的CosFace](#LFW数据集上的CosFace)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# CosFace描述

该文提出了一个增强CNN特征辨识力度的损失函数：large margin cosine loss (LMCL)。该损失函数能够增加不同类别的图像特征在特征空间的距离，从而提升CNN的分类能力。  
[论文](<https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_CosFace_Large_Margin_CVPR_2018_paper.html>)：Wang, H., Wang, Y., Zhou, Z., Ji, X., Gong, D., Zhou, J., ... & Liu, W. (2018). Cosface: Large margin cosine loss for deep face recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 5265-5274).

# 模型架构

为了进一步增加不同类别间的特征在特征空间上的距离，CosFace在原始softmax基础上引入了一个Margin来约束类间特征，致使类间特征在特征空间中的距离更大，从而增强重识别精度。

# 数据集

使用的数据集：[CASIA-WebFace](<https://github.com/XiaohangZhan/face_recognition_framework/tree/fd5971f1a3a1d82fb34843762321764341fcd939>)

- 数据集大小：2.5G，共10572个类、490623张112x112彩色或黑白的人脸对齐图像
- 数据格式：RGB
    - 注：数据将在src/CosFace_dataset.py中处理。
- 数据集的目录结构:

```markdown
  ./data/WebFace_122/images
  ├── 0
  │   ├──0_1.jpg
  │   ├──0_2.jpg
  │   ├──0_3.jpg
  │   ├──0_4.jpg
  │   ├──......
  ├── 1
  ├── 2
  ├── 3
  ├──......
```

使用的数据集: [LFW](<https://github.com/wujiyang/Face_Pytorch>)

- 数据集大小：68M，共5749个类、13233张112x112彩色的人脸对齐图像
- 数据格式：RGB
    - 注：数据将在src/CosFace_dataset.py和eval.py中处理。
- 数据集的目录结构:

```markdown
  ./data/LFW_112
  ├── lfw_align_112
  │   ├── Aaron_Peirsol
  │       ├──Aaron_Peirsol_0001.jpg
  │       ├──Aaron_Peirsol_0002.jpg
  │       ├──Aaron_Peirsol_0003.jpg
  │       ├──......
  │   ├── Aaron_Guiel
  │   ├── Aaron_Eckhart
  │   ├── Aaron_Patterson
  │   ├──......
  ├── pairs.txt
```

# 环境要求

- 硬件（Ascend）
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```train.py
  # 训练时，需要修改以下参数:  
  webface_dir: "./data/WebFace_122/images"       # CASIA-WebFace数据集图像文件夹的路径
  save_ckpt_dir: "./model"                       # ckpt模型保存目录
  lfw_path: "./data/LFW_112"                     # LFW数据集, 包含图像文件夹和标签文件的上一级目录

  ```

  ```eval.py
  # 评估时，需要修改以下参数:  
  feature_save_dir: "./lfw_rusult.mat"        # LFW数据集特征提取后的存放路径
  lfw_path: "./data/LFW_112"                  # LFW数据集, 包含图像文件夹和标签文件的上一级目录
  ckpt_path: "./checkpoint/CosFace_Epoch_98_batsh_128_ckpt_0__32s__040m.ckpt"    # ckpt模型的路径

  ```

  ```python
  # 运行训练示例
  cd scripts
  bash run_standalone_train_ascend.sh [DEVICE_ID] [IMAGE_PATH] [TEST_IMAGE_PATH] [SAVE_CKPT_DIR]
  # example: bash run_standalone_train_ascend.sh 0 ~/faces_webface_112x112/images ~/LFW_112 ./ckpt_standalone

  # 运行分布式训练示例
  cd scripts
  bash run_distribute_train_ascend.sh [DEVICE_NUM] [RANK_TABLE_FILE] [IMAGE_PATH] [TEST_IMAGE_PATH] [SAVE_CKPT_DIR]
  # example: bash run_distribute_train_ascend.sh 8 ~/hccl_8p.json ~/faces_webface_112x112/images ~/LFW_112 ./ckpt_distribute

  # 运行评估示例
  cd scripts
  bash run_eval_ascend.sh [DEVICE_ID] [TEST_IMAGE_PATH] [CKPT_FILES]
  # example: bash run_eval_ascend.sh 0 ~/LFW_112 ./ckpt_distribute/CosFace_Distributed_Epoch_100_.ckpt
  ```

  对于分布式, 训练需要提前创建JSON格式的hccl配置文件. 请遵循以下链接中的说明:

  <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools>

- 在 ModelArts（鹏城云脑） 进行训练 (如果你想在modelarts（鹏城云脑) 上运行，可以参考以下文档 [modelarts](https://git.openi.org.cn/zeizei/OpenI_Learning))

    - 在 ModelArts 上训练 CASIA-WebFace 数据集

      ```python

      # (1) 创建个人项目，上传相关代码文件，将CASIA-WebFace数据集上传至云脑，
      #     或直接使用云脑上的数据集[WebFace_122.zip](<https://git.openi.org.cn/wangzq/data_collection/datasets>)
      # (2) 创建调试任务，选择载入CASIA-WebFace数据集，获得CASIA-WebFace数据集的软链接
      # (3) 在train.py文件中，将获得的软链接写入参数data_soft_link，通过软链接方式载入数据集。（因数据集大，若以镜像obs方式载入，速度极慢且容易失败）
      # (4) 执行a或者b
      #       a. 在 train.py 文件中设置 "enable_pengcheng_cloud = 1"
      #          在 train.py 文件中设置 "workroot = '/home/work/user-job-dir'"
      #          在 train.py 文件中设置 "data_is_soft_ink" = 1， 启用软链接方式载入数据集
      #          在 train.py 文件中设置 "is_distributed = 1" 或 "is_distributed = 0"，决定是否开启分布式训练
      #          在 train.py 文件中设置 其他参数
      #          随后新建训练任务，选择npu类型
      #
      #       b. 首先新建训练任务，选择npu类型
      #          在网页上设置 "enable_pengcheng_cloud = 1"
      #          在网页上设置 "workroot = '/home/work/user-job-dir'"
      #          在 train.py 文件中设置 "data_is_soft_ink" = 1， 启用软链接方式载入数据集
      #          在网页上设置 "is_distributed = 1" 或 "is_distributed = 0"，决定是否开启分布式训练
      #          在网页上设置 其他参数
      # (5) 在网页上设置启动文件为 "train.py"
      # (6) 在网页上设置任意"数据集"（软链接载入数据集方式下）、并设置相应的"设备规格"、"mindspore版本"等
      # (7) 启动训练任务
      # (8) 在"日志"中查看训练信息，训练结束后在"结果下载"处下载模型

      ```

    - 在 ModelArts 上评估 LFW 数据集

      ```python

      # (1) 创建个人项目，上传相关代码文件和训练好的权重文件，将LFW数据集上传至云脑
      # (2) 执行a或者b
      #       a. 在 eval.py 文件中设置 "enable_pengcheng_cloud = 1"
      #          在 eval.py 文件中设置 "workroot = '/home/work/user-job-dir'"
      #          在 eval.py 文件中设置 代码仓中权重的存放路径"，如"ckpt_path = './checkpoint/CosFace_Epoch_98_batsh_128_ckpt_0__32s__040m.ckpt'"
      #          在 eval.py 文件中设置 其他参数
      #          随后新建训练任务，选择npu类型
      #
      #       b. 首先新建训练任务，选择npu类型
      #          在网页上设置 "enable_pengcheng_cloud = 1"
      #          在网页上设置 "workroot = '/home/work/user-job-dir'"
      #          在网页上设置 代码仓中权重的存放路径"，如"ckpt_path = './checkpoint/CosFace_Epoch_98_batsh_128_ckpt_0__32s__040m.ckpt'"
      #          在网页上设置 其他参数
      # (4) 在网页上设置启动文件为 "eval.py"
      # (6) 在网页上设置"数据集"为所上传的LFW数据集的压缩文件、并设置相应的"设备规格"、"mindspore版本"等
      # (7) 启动训练任务
      # (8) 在"日志"中查看评估精度信息

      ```

# 脚本说明

## 脚本及样例代码

```bash

├── model_zoo
    ├── README.md                           // 所有模型相关说明
    ├── CosFace
        ├── README.md                     // Cosface相关说明
        ├── ascend310_infer              // 实现310推理源代码
        ├── scripts
        │   ├──run_infer_310.sh                   // 执行310推理的shell脚本
        │   ├──run_distribute_train.sh            // 分布式到Ascend训练的shell脚本
        │   ├──run_eval_standalone_Ascend.sh      // Ascend评估的shell脚本
        │   ├──run_standalone_train_ascend.sh     // 单卡Ascend训练的shell脚本
        ├── src
        │   ├──create_callback.py                 // 回调函数
        │   ├──CosFace.py                         // CosFace网络结构
        │   ├──CosFace_dataset.py                 // 数据处理
        │   ├──lr_schedule.py                     // 动态学习率
        ├── requirements.txt
        ├── checkpoint               // 评估脚本的ckpt文件存放目录
        ├── postprocess.py          // 310推理精度计算脚本
        ├── preprocess.py           // 310推理数据预处理脚本
        ├── train.py               // 训练脚本
        ├── eval.py               // 评估脚本
        ├── export.py            // 将checkpoint文件导出成mindir

```

## 脚本参数

在train.py中配置训练参数，在eval.py中配置评估参数

- 配置训练参数。

  ```python

    num_s: 32.0                     # 缩放因子
    num_m: 0.40                     # 余弦裕度
    image_resize: (112,112)         # 输入网络的图像尺寸
    num_class: 10572                # 训练集的类别数
    batch_size: 128                 # 训练集的batch大小
    momentum: 0.9                   # momentum因子
    init_lr: 0.1                    # 初始学习率
    epoch_size: 100                 # 训练周期
    num_work: 2                     # 载入数据集的线程数
    epoch_star_save: 50             # 开始保存ckpt模型的epoch数
    epoch_per_save: 2               # 保存ckpt模型的epoch间隔数  
    weight_decay: 5e-4              # 网络的权重衰减系数
    lr_strategy: "Multistep"        # 学习率模式
    device_target: "Ascend"         # 设备类型
    dataset_sink_mode: 1            # 是否数据下沉
    is_distributed: 1               # 是否分布式
    save_ckpt_device: 0             # 保存ckpt的设备序号
    save_ckpt_dir: './model'        # 保存ckpt模型的目录
    webface_dir: './data/WebFace_122/images'                   # CASIA-WebFace数据集的图像文件夹images的路径
    enable_pengcheng_cloud: 1                                  # 运行环境是否为鹏城云脑的训练任务
    data_is_soft_ink: 1                                        # 是否通过软链接方式载入训练集
    workroot: '/home/work/user-job-dir'                        # 鹏城云脑的训练任务的运行环境
    train_url: " "                  # 鹏城云脑训练任务下的运行结果的镜像地址，启动训练任务时自动载入，无需设置
    data_url: " "                   # 鹏城云脑训练任务下的数据集obs地址，启动训练任务时自动载入，无需设置
    data_soft_link: " "             # 训练集的软链接，通过调试任务获取

  ```

- 配置评估参数。

 ```python

    batch_size: 256                 # 测试集batch大小
    num_work: 2                     # 载入测试集的线程数
    device_target: "Ascend"         # 设备类型
    num_class: 10572                # 训练时训练集的类别数
    image_resize: (112,112)         # 输入网络的图像尺寸
    feature_save_dir: './lfw_rusult.mat'                                              # LFW数据集特征提取后的存放路径
    lfw_path: './data/LFW_112'                                                        # LFW数据集, 包含图像文件夹和标签文件的上一级目录
    ckpt_path: './checkpoint/CosFace_Epoch_98_batsh_128_ckpt_0__32s__040m.ckpt'       # 所要测试的ckpt模型的路径
    enable_pengcheng_cloud: 1                                                         # 运行环境是否为鹏城云脑的训练任务
    workroot: '/home/work/user-job-dir'                                               # 鹏城云脑的训练任务的运行环境
    train_url: " "                  # 鹏城云脑训练任务下的运行结果的镜像地址，启动训练任务时自动载入，无需设置
    data_url: " "                   # 鹏城云脑训练任务下的数据集obs地址，启动训练任务时自动载入，无需设置

  ```

更多配置细节请参考脚本`train.py和eval.py`。

## 训练过程

### 训练

- Ascend处理器环境运行

  修改路径和相关参数后，运行训练示例

  ```bash
  cd scripts
  bash run_standalone_train_ascend.sh [DEVICE_ID] [IMAGE_PATH] [TEST_IMAGE_PATH] [SAVE_CKPT_DIR]
  ```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。训练结束后，您可在./model脚本文件夹下找到检查点文件。所记录的损失值形如：  
  epoch time：0.308 min,per step time：35.435 ms  
  epoch：1 step：479, loss is 21.010275  
  ...

### 分布式训练

- Ascend处理器环境运行

  对于分布式, 训练需要提前创建JSON格式的hccl配置文件. 请遵循以下链接中的说明:

  <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools>

  运行分布式训练示例

  ```bash
  cd scripts
  bash run_distribute_train_ascend.sh [DEVICE_NUM] [RANK_TABLE_FILE] [IMAGE_PATH] [TEST_IMAGE_PATH] [SAVE_CKPT_DIR]
  ```

  上述shell脚本将在后台运行分布训练。您可以通过/train_parallelX/trainX.log文件查看结果。所记录的损失值形如：  
  epoch time：0.308 min,per step time：38.598 ms  
  epoch：4 step：479, loss is 10.349318
  ...

## 评估过程

### 评估

- Ascend环境运行

  在运行以下命令之前将 ckpt_path 设置为所要测试的ckpt模型的路径

  ```bash
  cd scripts
  bash run_eval_ascend.sh [DEVICE_ID] [TEST_IMAGE_PATH] [CKPT_PATH]
  ```

  测试数据集的准确性如下：  
  Test result, AVE: 99.27

## 导出过程

### 导出

```python
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

参数`ckpt_file`为必填项，`file_format` 选为 "MINDIR"

## 推理过程

### 推理

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

在执行推理前，`mindir`文件必须通过`export.py`脚本导出。目前`LFW`数据集仅支持`batch_size`为1的推理。

```shell
# 昇腾310 推理
bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [DEVICE_ID]
# bash run_infer_310.sh ../Cosface_mindir.mindir 'LFW' ../data/LFW_112/ 0
```

`MINDIR_PATH` : `mindir`文件路径

`DATASET_NAME` : 使用的推理数据集名称，默认为`LFW`

`DATASET_PATH` : 推理数据集目录路径

`DEVICE_ID` : 可选，默认值为0

  ```log
  AVE    99.28

  ```

# 模型描述

## 性能

### 训练性能

#### CASIA-WebFace数据集上的CosFace

| 参数                 | Ascend
| -------------------------- | ----|
| 模型版本              | CosFace|CosFace|
| 资源                   | Ascend 910；系统 Euler2.8
| 上传日期              | 2022-6-30
| 版本          | MindSpore 1.5.1 |
| 数据集                    | CASIA-WebFace
| 训练参数        | 单卡: epoch=100, batch_size=128, lr=0.1 <br>八卡: epoch=100, batch_size=128, lr=0.1
| 优化器                  | SGD
| 损失函数              | CosFace
| 输出                    | 概率
| 损失                       | 4.0~5.0
| 速度                      | 单卡：35毫秒/步；八卡：38毫秒/步；
| 微调检查点 | 27M (.ckpt文件)

### 评估性能

#### LFW数据集上的CosFace

| 参数          | Ascend
| ------------------- | ----------------------
| 模型版本       | CosFace
| 资源            |  Ascend 910；系统 Euler2.8
| 上传日期       | 2022-6-30
| MindSpore 版本   | 1.5.1
| 数据集             | LFW数据集, 13233张图像
| batch_size          | 256
| 输出             | 概率
| 准确性            |  8卡：99.27%

# 随机情况说明

网络的初始参数均为随即初始化。

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。



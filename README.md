# 基于LibFewShot的LDP-net算法复现

## 项目简介

本项目基于 [LibFewShot](https://github.com/RL-VIG/LibFewShot) 框架，实现了论文 **Revisiting Prototypical Network for Cross-Domain Few-Shot Learning** 中 LDP-net 方法的复现。

原文链接：[CVPR 2023 Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Zhou_Revisiting_Prototypical_Network_for_Cross_Domain_Few-Shot_Learning_CVPR_2023_paper.html)
原文代码仓库：[NWPUZhoufei/LDP-Net](https://github.com/NWPUZhoufei/LDP-Net)

本项目主要完成了以下工作：

- 在 LibFewShot 框架中加入 LDP-Net 方法实现
- 完成 miniImageNet 训练流程
- 完成 CUB、Places 等跨域 few-shot 测试流程
- 对照官方实现与论文结果进行方法细节复现和实验记录

## 1.如何开始

### 1.1 环境配置

1. 克隆项目并进入代码目录：
```bash
git clone https://github.com/Kaka22-8/Reproduction-of-the-LDP-net-Based-on-LibFewShot.git
cd Reproduction-of-the-LDP-net-Based-on-LibFewShot/LibFewShot
```

2. 创建并激活conda环境：
```bash
conda create -n libfewshot python=3.8
conda activate libfewshot
```

3. 根据本机CUDA版本安装Pytorch和torchvision，可参考[Pytorch官方安装引导](https://pytorch.org/get-started/locally/)。

4. 安装项目依赖：
```bash
pip install -r requirements.txt 
```

### 1.2 下载数据集与预训练模型

本项目使用的数据集和预训练模型均来自 LDP-Net 官方论文仓库提供的下载链接：

[LDP-Net Official Repository](https://github.com/NWPUZhoufei/LDP-Net)

本项目主要使用：

- miniImageNet：用于训练
- CUB：用于 cross-domain few-shot 测试
- Places：用于 cross-domain few-shot 测试
- StanfordCar：用于 cross-domain few-shot 测试

下载后，请按照LibFewShot的数据读取逻辑整理数据集格式，并在配置文件或测试脚本中修改对应的`data_root`

训练配置实例：

```yaml
data_root: D:/datasets/miniImageNet--ravi
```

测试配置示例：

```yaml
VAR_DICT = {
    "data_root": "D:/datasets/CUB_200_2011_FewShot",
    ...
}
```

详细信息可参考[LibFewShot数据集格式文档](https://libfewshot-en.readthedocs.io/zh-cn/latest/tutorials/t2-add_a_new_dataset.html)

本项目训练阶段使用 LDP-Net 官方提供的预训练模型。下载后，将预训练模型放到：

```text
LibFewShot/pretrain/399_state.pth
```

## 2.训练流程

训练入口为：
```bash
python run_trainer.py
```

当前`run_trainer.py`默认读取配置：
```python
Config("./config/ldp_net_miniimagenet.yaml")
```

## 3.测试流程

测试入口为：
```bash
python run_test.py
```

测试前需要修改`run_test.py`中的`PATH`，使其指向某次训练产生的实验目录，例如：

```python
PATH = "./results/LDPNet-miniImageNet--ravi-resnet10-5-5-xxxx"
```

由于原论文研究的是`cross-domain few-shot`问题，测试阶段还需要根据目标数据集修改 `VAR_DICT`，例如 CUB 1-shot 测试：
```python
VAR_DICT = {
    "data_root": "D:/datasets/CUB_200_2011_FewShot",
    ...
}
```
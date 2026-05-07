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

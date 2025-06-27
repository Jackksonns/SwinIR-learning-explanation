# SwinIR 学习与复现笔记（SwinIR-learning-explanation）

作者：Jackson KK  
仓库地址：[https://github.com/Jackksonns/SwinIR-learning-explanation](https://github.com/Jackksonns/SwinIR-learning-explanation)

---

## 1. 项目简介 | Project Overview

本仓库为我个人深度学习 SwinIR（Swin Transformer for Image Restoration）模型架构的过程式记录与实验平台。**SwinIR** 是基于 Swin Transformer 架构的高性能图像复原模型，发表于 ICCV 并被 CCF-A 收录，在超分辨率（Super-Resolution）、图像去噪（Denoising）、JPEG压缩伪影去除（JPEG Compression Artifacts Removal）等任务中表现优异。

> This repository aims for a comprehensive, hands-on understanding and re-implementation of SwinIR.  
> All codes have detailed Chinese comments and learning logs, supporting full training, inference, and evaluation workflows.

---

## 2. SwinIR 推理与参数说明 | Inference & Argument Guide

### 基础测试命令（Test Command Example）

```bash
python main_test_swinir.py \
  --task color_dn \
  --noise 15 \
  --model_path model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth \
  --folder_gt ./my_images
```

**主要参数说明：**

- `--task`  任务类型 | Task type:
    - `classical_sr`: 普通超分辨率 Classical Super-Resolution
    - `real_sr`: 真实超分辨率 Real-world Super-Resolution
    - `gray_dn`: 灰度去噪 Grayscale Denoising
    - `color_dn`: 彩色去噪 Color Denoising
    - `jpeg_car`: JPEG压缩伪影去除 JPEG Compression Artifacts Removal
- `--scale`  放大倍数（如超分任务用2、3、4等）
- `--model_path`  权重文件路径 | Model checkpoint path
- `--folder_lq`  输入图片文件夹 | Input folder for low-quality images
- `--folder_gt`  高质量图片文件夹（可选）| High-quality ground truth images (optional)

---

## 3. SwinIR 训练流程与实践 | Training Workflow

### 3.1 数据准备 | Data Preparation

- 采用高质量（GT）和低质量（LQ）图片一一对应的数据结构。
- 利用自编写的 `prepare_data.py` 工具，自动生成训练所需的 LQ 图像，支持多种图像退化模式。

#### 超分辨率任务（下采样降质）:

```bash
python prepare_data.py --task sr --gt_folder datasets/train/GT --lq_folder datasets/train/LQ --scale 2
```

#### 去噪任务（添加高斯噪声）:

```bash
python prepare_data.py --task denoise --gt_folder datasets/train/GT --lq_folder datasets/train/LQ --noise 15
```

#### JPEG伪影去除任务:

```bash
python prepare_data.py --task jpeg --gt_folder datasets/train/GT --lq_folder datasets/train/LQ --quality 40
```

> 只需将高质量图片放入 `GT` 文件夹，自动生成对应 LQ 图片，保证训练集严格一一对应。

---

### 3.2 模型训练 | Model Training

#### 经典超分辨率训练指令（Classical SR）:

```bash
python main_train_swinir.py \
  --task classical_sr \
  --scale 2 \
  --folder_gt datasets/train/GT \
  --folder_lq datasets/train/LQ \
  --batch_size 16 \
  --epochs 1000 \
  --save_dir experiments/training
```

#### 轻量级模型训练（Lightweight Version，主力实验方向）:

```bash
python main_train_swinir.py \
  --task lightweight_sr \
  --scale 2 \
  --training_patch_size 64 \
  --batch_size 32 \
  --epochs 1000 \
  --lr 2e-4 \
  --folder_gt datasets/train/GT \
  --folder_lq datasets/train/LQ \
  --save_dir experiments/lightweight_sr
```

- `--task lightweight_sr` 表示采用轻量级超分模型（如 embed_dim=60, upsampler='pixelshuffledirect'），适合硬件资源有限场景。
- 数据集制作时严格去除异常格式（如 gif），确保 GT 与 LQ 完全一一对应。

---

### 3.3 训练中常见问题与解决方案 | Issues & Solutions

- **图片数量不匹配：**  
  由于部分格式（如 gif）无法处理，初次生成 LQ 时与 GT 数量不符。需剔除异常文件后重新生成，保证一致。
- **损失函数（loss）波动：**  
  训练过程中 loss 并非严格单调下降，整体下降趋势正常即可。更应关注验证集上的 PSNR/SSIM 等指标。
- **代码适配与 bug 修复：**  
  原版 SwinIR 代码未涵盖所有任务分支（如 lightweight_sr），需对 `dataset.py`、`main_train_swinir.py` 等文件进行适配和中文详细注释。

---

## 4. 模型评估与指标分析 | Evaluation & Metrics

本仓库实现了完整的模型评估流程，支持 PSNR（峰值信噪比）、SSIM（结构相似性）等经典指标。  
使用 `evaluate_model.py` 评估脚本：

```bash
python evaluate_model.py \
    --task lightweight_sr \
    --scale 2 \
    --model_path experiments/lightweight_sr/models/model_epoch_10.pth \
    --test_lr_folder testsets/Set5/LR_bicubic/X2 \
    --test_hr_folder testsets/Set5/HR \
    --save_folder evaluation_results
```

**评估输出说明：**

- `evaluation_results.json`：详细分项指标
- `evaluation_report.txt`：可读性报告
- `metrics_distribution.png`：可视化分布图
- `images/`：所有预测结果图片

> 小轮次训练（如10 epoch）指标较低属于正常。后续可通过增加 epoch、调整学习率、批量大小等方式优化模型表现，并尝试 Early Stopping 等训练策略。

---

## 5. 版本管理与同步 | Version Control & Sync

- 所有代码、日志、数据在本仓库统一管理，便于溯源与复现。
- 推送大文件需增大 Git 缓冲区：`git config --global http.postBuffer 524288000`。

---

## 6. 后续研究方向 | Future Plans

- 深入注释与完善各训练/测试/评估脚本，补充数据集来源说明。
- 理解 SwinIR 各分支网络结构，探索更多模型变体及参数调优（如 Early Stopping, lr schedule）。
- 以本仓库为 baseline，推进个人图像复原方向的创新性实验。
- 探索模型可扩展性，如更高分辨率、多任务联合训练等。

---

## 7. 致谢 | Acknowledgments

感谢 SwinIR 官方开源项目、B 站优质课程资源，以及谢学长的耐心指导。

---

## 8. English Summary

This repository documents a professional, hands-on journey into SwinIR, including:

- Data preparation tools for super-resolution, denoising, and artifact removal.
- Training, inference, and evaluation scripts (with detailed Chinese comments).
- Full pipeline experience, issues encountered, and practical solutions.
- All experiments, models, and results are reproducible.

Pull requests, discussions, and collaborations are welcome!

---

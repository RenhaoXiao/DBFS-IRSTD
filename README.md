# LQM-DBFS-IRSTD

A lightweight infrared small target detection framework built upon DEIMv2, featuring a dual-branch feature stream Transformer decoder with Layer-wise Query Memory (LQM) and Minimal BBox Enhancer (MBE).

> 面向红外小目标检测（IRSTD）的轻量化 Transformer 检测框架。  
> 本项目基于 **DEIMv2** 改进，核心创新实现在 `engine/deim/deim_decoder.py`，提出了：
> - **Dual-Branch Feature Stream Decoder**
> - **Layer-wise Query Memory (LQM)**
> - **Minimal BBox Enhancer (MBE)**

---

## 📌 Highlights

- 面向 **Infrared Small Target Detection (IRSTD)** 的轻量化检测框架
- 基于 **DEIMv2** 改进，兼顾检测精度与实时性
- 在解码阶段引入 **双分支特征流解码器**
- 通过 **LQM** 缓解深层解码中的目标语义稀释问题
- 通过 **MBE** 提升红外微小目标的定位稳定性
- 支持 **IRSTD-1k** / **NUDT-SIRST** 等红外小目标数据集
- 保留 DEIMv2 工程化训练、推理、导出和部署能力

---

## 🧠 Method Overview

Infrared small targets usually occupy only a few pixels and are easily overwhelmed by cluttered backgrounds. Existing DETR-style decoders often suffer from:

1. **Semantic dilution during iterative decoding**
2. **Unstable localization for tiny targets**

To address these issues, this project reconstructs the DEIMv2 decoder into a **Dual-Branch Feature Stream Decoder**, and introduces two lightweight modules:

- **LQM (Layer-wise Query Memory)**  
  Dynamically selects reliable queries and aggregates their semantics across decoder layers to stabilize target-related information.

- **MBE (Minimal BBox Enhancer)**  
  Applies lightweight residual refinement to predicted bounding boxes for better localization stability on tiny infrared targets.

The core implementation is located in:

```bash
engine/deim/deim_decoder.py
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your_name/LQM-DBFS-IRSTD.git
cd LQM-DBFS-IRSTD
```

### 2. Create environment

建议使用 Python 3.10+ 和 CUDA 对应版本的 PyTorch。

```bash
conda create -n lqm-dbfs-irstd python=3.10 -y
conda activate lqm-dbfs-irstd
```

### 3. Install dependencies

先安装与你 CUDA 版本匹配的 PyTorch，然后安装其余依赖：

```bash
pip install -r requirements.txt
```

如果你还没有安装 PyTorch，可以参考官网安装，例如：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> 请根据你的 CUDA 版本自行调整。

---

## 📊 Supported Datasets

This project is mainly designed for infrared small target detection on:

- **IRSTD-1k**
- **NUDT-SIRST**

请将数据集按你配置文件中的路径组织，并确保对应的 YAML 配置文件可正确读取数据。

相关配置位于：

```bash
configs/dataset/
configs/deimv2_IRSTD/
```

你当前训练配置示例：

```bash
configs/deimv2_IRSTD/MBE+LQM_IRSTD-1k.yml
```

---

## 🚀 Training

### Single-GPU training

你当前可直接使用的训练命令为：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py -c configs/deimv2_IRSTD/MBE+LQM_IRSTD-1k.yml --use-amp
```

### Windows 写法说明

如果你在 Windows 环境中使用原始路径，也可以写成：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py -c configs\deimv2_IRSTD\MBE+LQM_IRSTD-1k.yml --use-amp
```

> 推荐在 README 中统一使用 `/`，这样在 Linux/macOS/WSL 下更通用。

### 参数说明

- `OMP_NUM_THREADS=1`：限制 OpenMP 线程数，减少额外 CPU 占用
- `CUDA_VISIBLE_DEVICES=0`：指定使用第 0 块 GPU
- `torchrun`：分布式/标准训练启动方式
- `--master_port=7777`：指定通信端口
- `--nproc_per_node=1`：单卡训练
- `-c ...yml`：指定训练配置文件
- `--use-amp`：启用混合精度训练，加快训练并减少显存占用

---

## 🧪 Evaluation / Testing

如果只进行测试或验证，可以使用：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py -c configs/deimv2_IRSTD/MBE+LQM_IRSTD-1k.yml --test-only
```

如果需要从某个权重恢复测试，可加上 `-r`：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py -c configs/deimv2_IRSTD/MBE+LQM_IRSTD-1k.yml -r path/to/checkpoint.pth --test-only
```

---

## 🔁 Resume Training

从中断的 checkpoint 恢复训练：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py -c configs/deimv2_IRSTD/MBE+LQM_IRSTD-1k.yml -r path/to/checkpoint.pth --use-amp
```

---

## 🎯 Finetuning

如果你希望从已有权重进行微调，可以使用：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py -c configs/deimv2_IRSTD/MBE+LQM_IRSTD-1k.yml -t path/to/pretrained.pth --use-amp
```

---

## 📁 Core Contribution

本项目核心创新代码主要位于：

```bash
engine/deim/deim_decoder.py
```

主要改进包括：

- **Dual-Branch Feature Stream Decoder**
- **Layer-wise Query Memory (LQM)**
- **Minimal BBox Enhancer (MBE)**

如果你阅读源码，推荐从以下文件开始：

```bash
engine/deim/deim_decoder.py
engine/deim/deim.py
configs/deimv2_IRSTD/MBE+LQM_IRSTD-1k.yml
train.py
```

---

## 📈 Results

| Model | Dataset | Params (M) | GFLOPs | AP50:95 | AR50:95 |
|------|---------|------------|--------|---------|---------|
| DEIMv2_atto | IRSTD-1k | 0.48 | 1.80 | 0.316 | 0.546 |
| LQM-DBFS-IRSTD | IRSTD-1k | 0.52 | 1.88 | 0.433 | 0.557 |

---

## 🖼️ Visualization and Inference

项目中提供了推理与部署相关脚本：

```bash
tools/inference/torch_inf.py
tools/inference/torch_inf_vis.py
tools/inference/onnx_inf.py
tools/deployment/export_onnx.py
```

你可以根据自己的模型权重和配置文件进行推理、可视化和导出。

---

## 🙏 Acknowledgements

This repository is built upon the following excellent works:

- [DEIMv2: Real-Time Object Detection Meets DINOv3](#)
- [DEIM: DETR with Improved Matching for Fast Convergence](#)
- [RT-DETR](https://github.com/lyuwenyu/RT-DETR)

Thanks to the authors for their great open-source contributions.

---

## 📬 Contact

如有问题，欢迎提 issue 或联系作者。

- GitHub Issues: `https://github.com/your_name/LQM-DBFS-IRSTD/issues`

---

## ⭐ Star

如果这个项目对你有帮助，欢迎点个 **Star** 支持一下！
```

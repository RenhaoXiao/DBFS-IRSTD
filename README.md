# LQM-DBFS-IRSTD

A lightweight infrared small target detection framework built upon DEIMv2, featuring a dual-branch feature stream Transformer decoder with Layer-wise Query Memory (LQM) and Minimal BBox Enhancer (MBE).

>A lightweight Transformer detection framework for Infrared Small Target Detection (IRSTD).
>This project is an improvement on **DEIMv2**, with core innovations implemented in `engine/deim/deim_decoder.py`, proposing:
> - **Dual-Branch Feature Stream Decoder**
> - **Layer-wise Query Memory (LQM)**
> - **Minimal BBox Enhancer (MBE)**

---

## 📌 Highlights

- A lightweight detection framework for **Infrared Small Target Detection (IRSTD)**
- Improved based on **DEIMv2**, balancing detection accuracy and real-time performance
- Introduces a **dual-branch feature stream decoder** in the decoding stage
- Alleviates the target semantic dilution problem in deep decoding through **LQM**
- Improves the localization stability of infrared small targets through **MBE**
- Supports infrared small target datasets such as **IRSTD-1k** / **NUDT-SIRST**
- Retains the engineering training, inference, export, and deployment capabilities of DEIMv2

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

It is recommended to use Python 3.10+ and the corresponding version of PyTorch for CUDA.

```bash
conda create -n lqm-dbfs-irstd python=3.10 -y
conda activate lqm-dbfs-irstd
```

### 3. Install dependencies

First, install PyTorch that matches your CUDA version, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Training

### Single-GPU training

The training commands you can currently use directly are:

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py -c configs/deimv2_IRSTD/MBE+LQM_IRSTD-1k.yml --use-amp
```

### Parameter Description

- `OMP_NUM_THREADS=1`: Limits the number of OpenMP threads, reducing additional CPU usage.
- `CUDA_VISIBLE_DEVICES=0`: Specifies the use of GPU 0.
- `torchrun`: Distributed/standard training startup mode.
- `--master_port=7777`: Specifies the communication port.
- `--nproc_per_node=1`: Single-GPU training.
- `-c ...yml`: Specifies the training configuration file.
- `--use-amp`: Enables mixed-precision training, speeding up training and reducing GPU memory usage.

---

## 🧪 Evaluation / Testing

If you only need to perform testing or verification, you can use:

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py -c configs/deimv2_IRSTD/MBE+LQM_IRSTD-1k.yml --test-only
```

If you need to resume the test from a specific weight, add `-r`:

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py -c configs/deimv2_IRSTD/MBE+LQM_IRSTD-1k.yml -r path/to/checkpoint.pth --test-only
```

---

## 🎯 Finetuning

If you wish to fine-tune the existing weights, you can use:

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py -c configs/deimv2_IRSTD/MBE+LQM_IRSTD-1k.yml -t path/to/pretrained.pth --use-amp
```

---

## 📈 Results

| Model | Dataset | Params (M) | GFLOPs | AP50:95 | AR50:95 |
|------|---------|------------|--------|---------|---------|
| DEIMv2_atto | IRSTD-1k | 0.48 | 1.80 | 0.316 | 0.546 |
| LQM-DBFS-IRSTD | IRSTD-1k | 0.52 | 1.88 | 0.433 | 0.557 |

---

## 🖼️ Visualization and Inference

The project provides inference and deployment related scripts:

```bash
tools/inference/torch_inf.py
tools/inference/torch_inf_vis.py
tools/inference/onnx_inf.py
tools/deployment/export_onnx.py
```

You can perform inference, visualization, and export based on your own model weights and configuration files.

---

## 🙏 Acknowledgements

This repository is built upon the following excellent works:

- [DEIMv2: Real-Time Object Detection Meets DINOv3](#)
- [DEIM: DETR with Improved Matching for Fast Convergence](#)
- [RT-DETR](https://github.com/lyuwenyu/RT-DETR)

Thanks to the authors for their great open-source contributions.

---

## 📬 Contact

If you have any questions, please feel free to submit an issue or contact the author.

- GitHub Issues: `https://github.com/your_name/LQM-DBFS-IRSTD/issues`

---

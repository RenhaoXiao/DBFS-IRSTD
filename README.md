# DBFS-IRSTD

A lightweight infrared small target detection framework built upon DEIMv2, featuring a dual-branch feature stream Transformer decoder with Layer-wise Query Memory (LQM) and Minimal BBox Enhancer (MBE).

>A lightweight Transformer detection framework for Infrared Small Target Detection (IRSTD).
>With core innovations implemented in `engine/deim/deim_decoder.py`, proposing:
> - **Dual-Branch Feature Stream Decoder**
> - **Layer-wise Query Memory (LQM)**
> - **Minimal BBox Enhancer (MBE)**

---

## 📌 Highlights

- A lightweight detection framework for **Infrared Small Target Detection (IRSTD)**
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

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/RenhaoXiao/DBFS-IRSTD.git
cd DBFS-IRSTD
```

### 2. Install dependencies

Install PyTorch that matches your CUDA version, then install the remaining dependencies:

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
- `-c`: Specifies the training configuration file.
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

### Main Results

| Model | Dataset | Params (M) | GFLOPs | AP50:95 | AR50:95 |
|------|---------|------------:|-------:|---------:|---------:|
| DEIMv2_atto (Baseline) | IRSTD-1k | 0.48 | 1.80 | 0.316 | 0.546 |
| LQM-DBFS-IRSTD (Ours) | IRSTD-1k | 0.52 | 1.88 | 0.433 | 0.557 |
| DEIMv2_atto (Baseline) | NUDT-SIRST | 0.48 | 1.80 | 0.646 | 0.711 |
| LQM-DBFS-IRSTD (Ours) | NUDT-SIRST | 0.52 | 1.88 | 0.702 | 0.766 |

---

## 🔬 Ablation Studies

### Ablation on IRSTD-1k

| LQM | MBE | AP50:95 | AR50:95 | Params (M) | GFLOPs |
|:---:|:---:|---------:|---------:|-----------:|-------:|
| ✗ | ✗ | 0.316 | 0.546 | 0.48 | 1.80 |
| ✗ | ✓ | 0.426 | 0.549 | 0.49 | 1.88 |
| ✓ | ✗ | 0.431 | 0.550 | 0.51 | 1.87 |
| ✓ | ✓ | **0.433** | **0.557** | 0.52 | 1.88 |

### Ablation on NUDT-SIRST

| LQM | MBE | AP50:95 | AR50:95 | Params (M) | GFLOPs |
|:---:|:---:|---------:|---------:|-----------:|-------:|
| ✗ | ✗ | 0.646 | 0.711 | 0.48 | 1.80 |
| ✗ | ✓ | 0.689 | 0.759 | 0.49 | 1.88 |
| ✓ | ✗ | 0.658 | 0.736 | 0.51 | 1.87 |
| ✓ | ✓ | **0.702** | **0.766** | 0.52 | 1.88 |

---

## 📊 Quantitative Comparison

### IRSTD-1k

| Model | Params (M) | GFLOPs | AP50:95 | AR50:95 | Pd@0.5 | Pd@0.75 | F1@0.5 | F1@0.75 |
|------|-----------:|-------:|---------:|---------:|-------:|--------:|-------:|--------:|
| **CNN-based Models** |||||||||
| YOLOv5n | 2.50 | 7.10 | 0.475 | 0.556 | 0.822 | 0.418 | 0.847 | 0.431 |
| YOLOv8n | 3.01 | 8.10 | **0.482** | 0.562 | 0.838 | 0.428 | 0.869 | 0.443 |
| YOLOv9t | 1.97 | 7.60 | 0.431 | 0.522 | 0.815 | 0.502 | 0.849 | 0.523 |
| YOLOv10n | 2.27 | 6.50 | 0.469 | 0.568 | 0.832 | 0.387 | 0.823 | 0.383 |
| YOLO11n | 2.58 | 6.30 | 0.444 | 0.535 | 0.852 | 0.411 | 0.859 | 0.414 |
| YOLO12n | 2.56 | 6.30 | 0.455 | 0.539 | 0.845 | 0.532 | **0.887** | **0.558** |
| YOLO26n | 2.38 | 5.20 | 0.480 | **0.581** | 0.882 | 0.471 | 0.863 | 0.461 |
| **Transformer-based Models** |||||||||
| RT-DETRv2_r18 | 19.03 | 19.60 | 0.387 | 0.542 | 0.879 | 0.495 | 0.645 | 0.363 |
| Lite-DETR | 47.19 | 43.40 | 0.427 | 0.575 | 0.875 | 0.519 | 0.799 | 0.473 |
| D-FINE_n | 3.72 | 4.80 | 0.446 | 0.546 | **0.916** | 0.475 | 0.802 | 0.416 |
| DEIMv2_n | 3.54 | 4.60 | 0.438 | 0.440 | 0.902 | 0.481 | 0.816 | 0.435 |
| DEIMv2_pico | 1.49 | 3.30 | 0.422 | 0.423 | 0.886 | 0.498 | 0.832 | 0.468 |
| DEIMv2_femto | 0.94 | 2.40 | 0.435 | 0.437 | 0.896 | 0.478 | 0.786 | 0.419 |
| **Ours (MBE+LQM)** | **0.52** | **1.88** | 0.433 | 0.557 | 0.906 | **0.542** | 0.785 | 0.470 |

### NUDT-SIRST

| Model | Params (M) | GFLOPs | AP50:95 | AR50:95 | Pd@0.5 | Pd@0.75 | F1@0.5 | F1@0.75 |
|------|-----------:|-------:|---------:|---------:|-------:|--------:|-------:|--------:|
| **CNN-based Models** |||||||||
| YOLOv5n | 2.50 | 7.10 | 0.676 | 0.732 | 0.960 | 0.810 | 0.934 | 0.788 |
| YOLOv8n | 3.01 | 8.10 | 0.664 | 0.727 | 0.968 | 0.786 | 0.938 | 0.762 |
| YOLOv9t | 1.97 | 7.60 | 0.679 | 0.739 | 0.976 | 0.818 | 0.944 | 0.792 |
| YOLOv10n | 2.27 | 6.50 | 0.650 | 0.751 | 0.955 | 0.807 | 0.860 | 0.728 |
| YOLO11n | 2.58 | 6.30 | 0.675 | 0.737 | 0.955 | 0.810 | 0.928 | 0.788 |
| YOLO12n | 2.56 | 6.30 | 0.624 | 0.681 | 0.979 | 0.770 | 0.951 | 0.748 |
| YOLO26n | 2.38 | 5.20 | **0.730** | 0.801 | 0.987 | **0.874** | 0.880 | 0.779 |
| **Transformer-based Models** |||||||||
| RT-DETRv2_r18 | 19.03 | 19.60 | 0.681 | 0.772 | 0.984 | 0.861 | 0.815 | 0.713 |
| Lite-DETR | 47.19 | 43.40 | 0.717 | **0.831** | 0.976 | **0.904** | 0.957 | **0.886** |
| D-FINE_n | 3.72 | 4.80 | 0.716 | 0.776 | 0.992 | 0.874 | 0.926 | 0.816 |
| DEIMv2_n | 3.54 | 4.60 | 0.668 | 0.722 | **0.997** | 0.807 | **0.975** | 0.790 |
| DEIMv2_pico | 1.49 | 3.30 | 0.638 | 0.706 | 0.992 | 0.786 | 0.915 | 0.725 |
| DEIMv2_femto | 0.94 | 2.40 | 0.660 | 0.725 | **0.997** | 0.874 | 0.949 | 0.832 |
| **Ours (MBE+LQM)** | **0.52** | **1.88** | 0.702 | 0.766 | 0.987 | 0.872 | 0.897 | 0.792 |


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

- [DEIMv2: Real-Time Object Detection Meets DINOv3](https://github.com/Intellindust-AI-Lab/DEIMv2)
- [DEIM: DETR with Improved Matching for Fast Convergence](https://github.com/Intellindust-AI-Lab/DEIM)
- [RT-DETR: DETRs Beat YOLOs on Real-time Object Detection](https://github.com/lyuwenyu/RT-DETR)

Thanks to the authors for their great open-source contributions.

---

## 📬 Contact

If you have any questions, please feel free to submit an issue or contact the author.

- GitHub Issues: `https://github.com/RenhaoXiao/DBFS-IRSTD/issues`

---

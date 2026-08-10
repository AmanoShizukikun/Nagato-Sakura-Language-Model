# Nagato-Sakura-Language-Model

[![GitHub Repo stars](https://img.shields.io/github/stars/AmanoShizukikun/Nagato-Sakura-Language-Model?style=social)](https://github.com/AmanoShizukikun/Nagato-Sakura-Language-Model/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/AmanoShizukikun/Nagato-Sakura-Language-Model)](https://github.com/AmanoShizukikun/Nagato-Sakura-Language-Model/commits/main)
[![GitHub release](https://img.shields.io/github/v/release/AmanoShizukikun/Nagato-Sakura-Language-Model)](https://github.com/AmanoShizukikun/Nagato-Sakura-Language-Model/releases)

\[ [中文](https://github.com/AmanoShizukikun/Nagato-Sakura-Language-Model/blob/main/README.md) | English | [日本語](https://github.com/AmanoShizukikun/Nagato-Sakura-Language-Model/blob/main/assets/docs/README_jp.md) \]

## Introduction
**Nagato Sakura LM** (**NS-LM**) is an autoregressive language model project implemented in PyTorch. Its goal is to provide a training/inference system with **transparent architecture, minimal dependencies, and end-to-end full-stack control by a single developer**.

## Announcements
Version 1.6.0 is a major update that optimizes the model architecture, significantly improves model efficiency, and substantially reduces memory consumption. It maintains backward compatibility with older models (though running older models will be slightly slower, they are still ~2x faster than before).

## Recent Changes
### 1.6.0 (August 10, 2026)
![t2i](https://raw.githubusercontent.com/AmanoShizukikun/Nagato-Sakura-Language-Model/refs/heads/main/assets/preview/1.6.0.jpg)
### Important Changes
- [Major] Adjusted tokenizer to add more fallback tokens.
- [Major] Refactored model architecture; the new model now delivers nearly doubled performance in both training and inference.
### New Features & Updates
- [New] Weight quantization feature, supporting native INT8 quantization (INT4 is currently in progress).
- [Update] Improved processing speed of `data_utils.py`, enhancing data loading efficiency.
- [Update] Improved `kv_cache.py`, significantly improving memory growth and efficiency during inference.
- [Update] Improved `train.py` efficiency, drastically reducing memory consumption during training.
- [Update] Improved WebUI layout and display effects, adding more functional options.
- [Update] Improved flexibility of `tokenizer.py`, allowing freer selection of fallback tokens and characters.
- [Fix] Fixed an issue where ByteLevel token 124 overlapped with UTF-8 continuation bytes, producing corrupted text.
### Known Issues
- N/A


[All Releases](https://github.com/AmanoShizukikun/Nagato-Sakura-Language-Model/blob/main/assets/docs/Changelog.md)


## Model Architecture
NS-LM is a standard **Pre-Norm Decoder-only Transformer**, engineered specifically for "efficient small model training + extreme memory compression":

| Component | Design | Description |
|---|---|---|
| Positional Encoding | RoPE (Rotary Position Embedding) | Supports dynamic length scaling (`_dynamic_frequency_update`), configurable `rope_theta` |
| Normalization | RMSNorm (Pre-Norm) | Dual-layer structure with `input_layernorm` / `post_attention_layernorm`, plus `final_layernorm` |
| Attention | **GQA** (Grouped-Query Attention) + Fused QKV Projection | Automatically sets `num_key_value_heads` to the largest divisor of `num_attention_heads // 4` if unspecified; prioritizes PyTorch **SDPA** (`scaled_dot_product_attention`), retaining handcrafted fallback |
| FFN | **SwiGLU** MLP | `intermediate_size` automatically defaults to `hidden_size × 2.7` |
| Word Embedding | Tied embedding (Optional) | Shares input/output projections when `tie_word_embeddings=True` |
| Loss Calculation | **Chunked Cross-Entropy** (Custom `torch.autograd.Function` backward pass) | Avoids materializing the massive `(B×S, vocab)` logits matrix at once, significantly reducing peak training VRAM |
| VRAM Optimization | Gradient Checkpointing, `torch.compile` (with automatic fallback) | Both can be independently toggled via CLI flags |

The model is defined in `NSConfig` (`nagato_sakura_model.py`), seamlessly scaling from the "pico" specification (used in the default training script demo: `hidden_size=128, num_layers=2`) to larger scales without modifying any architecture code.

## Quick Start
> [!NOTE]
> If you do not use model training features or do not have an NVIDIA GPU, installing the first three items is sufficient.
### Environment Setup
- **Python 3**
  - Download: https://www.python.org/downloads/windows/
- **PyTorch**
  - Download: https://pytorch.org/
- NVIDIA GPU Driver
  - Download: https://www.nvidia.com/en-us/geforce/drivers/
- NVIDIA CUDA Toolkit
  - Download: https://developer.nvidia.com/cuda-toolkit
- NVIDIA cuDNN
  - Download: https://developer.nvidia.com/cudnn
> [!TIP]
> Please install the CUDA version supported by your current PyTorch installation.

### Installation
> [!IMPORTANT]
> This is a required step.
```shell
git clone https://github.com/AmanoShizukikun/Nagato-Sakura-Language-Model.git
cd Nagato-Sakura-Language-Model
pip install -r requirements.txt
```

## Training Models
```shell
python train.py
```

## CLI Inference
```shell
python inference.py --mode interactive --model_path /path/to/your/model/folder
```

## Web Inference
### Method 1: Via inference.py (Recommended)
```shell
python inference.py --mode web --model_path /path/to/your/model/folder
```

### Method 2: Direct Script Launch
```shell
python tools/web_demo_flask.py --model_path /path/to/your/model/folder
```

## TODO
N/A

## Acknowledgements
Special thanks to the following projects and contributors:

### Projects

### Contributors
<a href="https://github.com/AmanoShizukikun/Nagato-Sakura-Language-Model/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=AmanoShizukikun/Nagato-Sakura-Language-Model" />
</a>
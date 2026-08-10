# Nagato-Sakura-Language-Model

[![GitHub Repo stars](https://img.shields.io/github/stars/AmanoShizukikun/Nagato-Sakura-Language-Model?style=social)](https://github.com/AmanoShizukikun/Nagato-Sakura-Language-Model/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/AmanoShizukikun/Nagato-Sakura-Language-Model)](https://github.com/AmanoShizukikun/Nagato-Sakura-Language-Model/commits/main)
[![GitHub release](https://img.shields.io/github/v/release/AmanoShizukikun/Nagato-Sakura-Language-Model)](https://github.com/AmanoShizukikun/Nagato-Sakura-Language-Model/releases)

\[ 中文 | [English](https://github.com/AmanoShizukikun/Nagato-Sakura-Language-Model/blob/main/assets/docs/README_en.md) | [日本語](https://github.com/AmanoShizukikun/Nagato-Sakura-Language-Model/blob/main/assets/docs/README_jp.md) \]

## 簡介
**Nagato Sakura LM**（長門櫻語言模型，簡稱 **NS-LM**）是一套以 PyTorch 實作的自回歸語言模型專案，目標是提供一個**架構透明、依賴極少、單人可全鏈路掌控**的訓練/推理系統。

## 公告
1.6.0 版本為大規模更新，重新優化了模型架構並大幅提升了模型效率並顯著降低記憶體的消耗，保持向下兼容舊版模型的能力但速度會稍微比較慢，但舊模型的運行速度依舊會比以前快個2倍左右。

## 近期變動
### 1.6.0 (2026 年 8 月 10 日)
![t2i](https://raw.githubusercontent.com/AmanoShizukikun/Nagato-Sakura-Language-Model/refs/heads/main/assets/preview/1.6.0.jpg)
### 重要變更
- 【重大】調整 tokenizer 新增加更多保底符號。
- 【重大】重構了模型架構現在的新模型在訓練、推理基本都有翻倍的性能提升。
### 新增功能
- 【新增】權重量化功能，支援原生INT8量化 (INT4目前尚未完成)。
- 【更新】改進 data_utils.py 的處理速度，提高資料的載入效率。
- 【更新】改進 kv_cache.py 顯著改善推理記憶體上升以及效率問題。
- 【更新】改進 train.py 的效率問題，並大幅降低訓練時的記憶體消耗。
- 【更新】改進 webui 頁面顯示效果，以及更多功能選項。
- 【更新】改進 tokenizer.py 的自由性，能更自由的選擇保底符號、字元。
- 【修復】ByteLevel token 124 與 UTF-8 後半 Continuation Byte 重疊產生亂碼的問題。
### 已知問題
- N/A


[所有發行版本](https://github.com/AmanoShizukikun/Nagato-Sakura-Language-Model/blob/main/assets/docs/Changelog.md)


## 模型架構
NS-LM 是標準的 **Pre-Norm Decoder-only Transformer**，並針對「小模型高效訓練 + 極限記憶體壓縮」做了工程強化：

| 組件 | 設計 | 說明 |
|---|---|---|
| 位置編碼 | RoPE（旋轉位置編碼） | 支援動態長度擴充（`_dynamic_frequency_update`），`rope_theta` 可調 |
| 正規化 | RMSNorm（Pre-Norm） | `input_layernorm` / `post_attention_layernorm` 雙層結構，附 `final_layernorm` |
| 注意力 | **GQA**（Grouped-Query Attention）+ 融合 QKV 投影 | `num_key_value_heads` 未指定時自動取 `num_attention_heads // 4` 且能整除的最大值；優先走 PyTorch **SDPA**（`scaled_dot_product_attention`），並保留手寫 fallback |
| FFN | **SwiGLU** MLP | `intermediate_size` 預設自動校正為 `hidden_size × 2.7` |
| 詞嵌入 | Tied embedding（可選） | `tie_word_embeddings=True` 時共享輸入/輸出投影 |
| 損失計算 | **Chunked Cross-Entropy**（`torch.autograd.Function` 自訂反傳） | 避免一次性物化 `(B×S, vocab)` 巨大 logits 矩陣，大幅降低訓練顯存峰值 |
| 顯存優化 | Gradient Checkpointing、`torch.compile`（含自動降級 fallback） | 兩者皆可透過 CLI 開關獨立控制 |

模型定義於 `NSConfig`（`nagato_sakura_model.py`），可自由從「pico」規格（預設訓練腳本示範用的 `hidden_size=128, num_layers=2`）一路擴展到更大規模，而不需更動任何架構程式碼。

## 快速開始
> [!NOTE]
> 如果沒有使用到模型訓練功能或著非 NVIDIA 顯卡用戶可只安裝前三項即可。
### 環境設置
- **Python 3**
  - 下載: https://www.python.org/downloads/windows/
- **PyTorch**
  - 下載: https://pytorch.org/
- NVIDIA GPU驅動程式
  - 下載: https://www.nvidia.com/zh-tw/geforce/drivers/
- NVIDIA CUDA Toolkit
  - 下載: https://developer.nvidia.com/cuda-toolkit
- NVIDIA cuDNN
  - 下載: https://developer.nvidia.com/cudnn
> [!TIP]
> 請按照當前 PyTorch 支援安裝對應的 CUDA 版本。

### 安裝倉庫
> [!IMPORTANT]
> 此為必要步驟。
```shell
git clone https://github.com/AmanoShizukikun/Nagato-Sakura-Language-Model.git
cd Nagato-Sakura-Language-Model
pip install -r requirements.txt
```

## 進行模型訓練
```shell
python train.py
```

## Cli 推理
```shell
python inference.py --mode interactive --model_path 你的模型資料夾路徑
```

## Web 推理
### 方式 1：透過 inference.py（建議）
```shell
python inference.py --mode web --model_path 你的模型資料夾路徑
```

### 方式 2：直接啟動腳本
```shell
python tools/web_demo_flask.py --model_path 你的模型資料夾路徑
```

## 待辦事項
N/A

## 致謝
特別感謝以下項目和貢獻者：

### 項目

### 貢獻者
<a href="https://github.com/AmanoShizukikun/Nagato-Sakura-Language-Model/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=AmanoShizukikun/Nagato-Sakura-Language-Model" />
</a>
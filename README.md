# Nagato-Sakura-Language-Model

[![GitHub Repo stars](https://img.shields.io/github/stars/AmanoShizukikun/Nagato-Sakura-Language-Model?style=social)](https://github.com/AmanoShizukikun/Nagato-Sakura-Language-Model/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/AmanoShizukikun/Nagato-Sakura-Language-Model)](https://github.com/AmanoShizukikun/Nagato-Sakura-Language-Model/commits/main)
[![GitHub release](https://img.shields.io/github/v/release/AmanoShizukikun/Nagato-Sakura-Language-Model)](https://github.com/AmanoShizukikun/Nagato-Sakura-Language-Model/releases)

\[ 中文 | [English](https://github.com/AmanoShizukikun/Nagato-Sakura-Language-Model/blob/main/assets/docs/README_en.md) | [日本語](https://github.com/AmanoShizukikun/Nagato-Sakura-Language-Model/blob/main/assets/docs/README_jp.md) \]

## 簡介
「長門櫻-語言模型」是「長門櫻計畫」的衍生，是作為「長門櫻」底層的語言模型，實現自然語言處理功能。

## 公告
由於設備性能問題，目前以小巧高性能 60M 參數模型為目標開發模型中。

## 近期變動
### 1.2.0 (2026 年 4 月 17 日)
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-Language-Model/blob/main/assets/preview/1.2.0.jpg)
### 重要變更
- 【重大】
### 新增功能
- 【新增】inference.py 新增臨時聊天模式。
- 【新增】inference.py 新增量化功能(由於參數過小不建議開啟)。
- 【修復】inference.py 額外輸入導致回復異常的錯誤。
- 【修復】train.py 輸出訊息會切斷進度條顯示的問題。
### 已知問題
- N/A


[所有發行版本](https://github.com/AmanoShizukikun/Nagato-Sakura-Language-Model/blob/main/assets/docs/Changelog.md)

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

## 待辦事項
N/A

## 致謝
特別感謝以下項目和貢獻者：

### 項目

### 貢獻者
<a href="https://github.com/AmanoShizukikun/Nagato-Sakura-Language-Model/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=AmanoShizukikun/Nagato-Sakura-Language-Model" />
</a>
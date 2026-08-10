# Nagato-Sakura-Language-Model

[![GitHub Repo stars](https://img.shields.io/github/stars/AmanoShizukikun/Nagato-Sakura-Language-Model?style=social)](https://github.com/AmanoShizukikun/Nagato-Sakura-Language-Model/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/AmanoShizukikun/Nagato-Sakura-Language-Model)](https://github.com/AmanoShizukikun/Nagato-Sakura-Language-Model/commits/main)
[![GitHub release](https://img.shields.io/github/v/release/AmanoShizukikun/Nagato-Sakura-Language-Model)](https://github.com/AmanoShizukikun/Nagato-Sakura-Language-Model/releases)

\[ [中文](https://github.com/AmanoShizukikun/Nagato-Sakura-Language-Model/blob/main/README.md) | [English](https://github.com/AmanoShizukikun/Nagato-Sakura-Language-Model/blob/main/assets/docs/README_en.md) | 日本語 \]

## 概要
**Nagato Sakura LM**（長門桜言語モデル、略称 **NS-LM**）は、PyTorchで実装された自己回帰型言語モデルプロジェクトです。**透明性の高いアーキテクチャ、最小限の依存関係、単一開発者がフルスタックで制御可能**な学習/推論システムを提供することを目的としています。

## お知らせ
バージョン 1.6.0 は大規模なアップデートであり、モデルアーキテクチャの最適化によってモデルの効率を大幅に向上させ、メモリ消費を著しく削減しました。旧バージョンモデルとの下位互換性を保持していますが（旧モデルの動作はやや低速になります）、以前と比べて約2倍の高速化が維持されています。

## 最近の変更点
### 1.6.0 (2026年8月10日)
![t2i](https://raw.githubusercontent.com/AmanoShizukikun/Nagato-Sakura-Language-Model/refs/heads/main/assets/preview/1.6.0.jpg)
### 重要な変更
- 【重要】tokenizer を調整し、フォールバック記号を追加しました。
- 【重要】モデルアーキテクチャを再構築しました。新しいモデルは学習および推論速度において約2倍の性能向上を実現しています。
### 新機能・更新
- 【新機能】重み量子化機能を追加。ネイティブ INT8 量子化に対応（INT4 は現在開発中）。
- 【更新】`data_utils.py` の処理速度を向上させ、データロード効率を高めました。
- 【更新】`kv_cache.py` を改善し、推論時のメモリ増加および効率の問題を大幅に改善しました。
- 【更新】`train.py` の効率を改善し、学習時のメモリ（VRAM）消費量を大幅に削減しました。
- 【更新】WebUI ページの表示効果を改善し、機能オプションを追加しました。
- 【更新】`tokenizer.py` の自由度を向上させ、フォールバック記号や文字をより自由に選択できるようになりました。
- 【修正】ByteLevel token 124 と UTF-8 の後半 Continuation Byte が重複して文字化けが発生する問題を修正しました。
### 既知の問題
- N/A


[すべてのリリース](https://github.com/AmanoShizukikun/Nagato-Sakura-Language-Model/blob/main/assets/docs/Changelog.md)


## モデルアーキテクチャ
NS-LM は標準的な **Pre-Norm Decoder-only Transformer** であり、「小規模モデルの高効率学習 + 極限までのメモリ圧縮」に向けてエンジニアリングの強化が行われています：

| コンポーネント | 設計 | 説明 |
|---|---|---|
| 位置符号化 | RoPE（Rotary Position Embedding） | 動的長さ拡張（`_dynamic_frequency_update`）に対応、`rope_theta` 調整可能 |
| 正規化 | RMSNorm（Pre-Norm） | `input_layernorm` / `post_attention_layernorm` の2層構造、`final_layernorm` 付き |
| アテンション | **GQA**（Grouped-Query Attention）+ 統合 QKV 射影 | `num_key_value_heads` 未指定時は `num_attention_heads // 4` の割り切れる最大値を自動取得。PyTorch **SDPA**（`scaled_dot_product_attention`）を優先し、手書き fallback も保持 |
| FFN | **SwiGLU** MLP | `intermediate_size` はデフォルトで `hidden_size × 2.7` に自動調整 |
| 単語埋め込み | Tied embedding（オプション） | `tie_word_embeddings=True` 時に入力/出力射影を共有 |
| 損失計算 | **Chunked Cross-Entropy**（カスタム `torch.autograd.Function` 逆伝播） | 一括での `(B×S, vocab)` 巨大 logits 行列の作成を回避し、学習時の VRAM ピークを大幅に削減 |
| メモリ最適化 | Gradient Checkpointing、`torch.compile`（自動フォールバック機能付き） | 両機能とも CLI フラグで個別に制御可能 |

モデルは `NSConfig`（`nagato_sakura_model.py`）で定義されており、アーキテクチャのコードを変更することなく、デフォルトの学習スクリプト用「pico」仕様（`hidden_size=128, num_layers=2`）からより大規模なモデルまで自由に拡張可能です。

## クイックスタート
> [!NOTE]
> モデルの学習機能を使用しない場合、または NVIDIA GPU 以外のユーザーは、最初の3項目のみインストールすれば問題ありません。
### 環境設定
- **Python 3**
  - ダウンロード: https://www.python.org/downloads/windows/
- **PyTorch**
  - ダウンロード: https://pytorch.org/
- NVIDIA GPU ドライバー
  - ダウンロード: https://www.nvidia.com/ja-jp/geforce/drivers/
- NVIDIA CUDA Toolkit
  - ダウンロード: https://developer.nvidia.com/cuda-toolkit
- NVIDIA cuDNN
  - ダウンロード: https://developer.nvidia.com/cudnn
> [!TIP]
> お使いの PyTorch がサポートしている CUDA バージョンに合わせてインストールしてください。

### リポジトリのインストール
> [!IMPORTANT]
> これは必須の手順です。
```shell
git clone https://github.com/AmanoShizukikun/Nagato-Sakura-Language-Model.git
cd Nagato-Sakura-Language-Model
pip install -r requirements.txt
```

## モデルの学習
```shell
python train.py
```

## CLI 推理
```shell
python inference.py --mode interactive --model_path あなたのモデルフォルダのパス
```

## Web 推理
### 方法 1：inference.py を使用（推奨）
```shell
python inference.py --mode web --model_path あなたのモデルフォルダのパス
```

### 方法 2：スクリプトを直接起動
```shell
python tools/web_demo_flask.py --model_path あなたのモデルフォルダのパス
```

## TODO
N/A

## 謝辞
以下のプロジェクトおよび貢献者に心より感謝いたします：

### プロジェクト

### 貢献者
<a href="https://github.com/AmanoShizukikun/Nagato-Sakura-Language-Model/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=AmanoShizukikun/Nagato-Sakura-Language-Model" />
</a>
### 1.2.1 (2026 年 4 月 18 日)
### 重要變更
- 【重大】分詞器升級為 UTF-8 安全路徑：Byte-level BPE 訓練流程強化 `unk_token`/`byte_fallback` 與 Unicode 健檢。
### 新增功能
- 【新增】`src/tokenizer.py`：新增中英日符號與 emoji 的 round-trip 健檢，訓練後若出現 `�` 會直接失敗。
- 【新增】`train.py`：`--force_retrain_tokenizer` 現在要求搭配 `--no_resume`，避免新舊詞表混用。
- 【新增】`src/trainer.py`：checkpoint 恢復前新增 tokenizer hash、vocab、權重 shape 相容性檢查。
- 【新增】`src/data_utils.py`：pretokenize 快取簽名納入 tokenizer 設定，避免重用錯誤快取。
- 【新增】`inference.py`：載入 tokenizer 時加入 UTF-8 健檢警示。
### 已知問題
- 舊版 checkpoint 的 legacy tokenizer 若 `byte_fallback=false`，仍可能在推理時對未見字元產生 `�`；建議用新 run 重建 tokenizer。

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
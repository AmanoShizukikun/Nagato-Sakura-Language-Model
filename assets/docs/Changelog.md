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

### 1.5.0 (2026 年 8 月 7 日)
### 重要變更
- N/A
### 新增功能
- 【修復】ByteLevel token 124 與 UTF-8 後半 Continuation Byte 重疊產生亂碼的問題。
### 已知問題
- N/A

### 1.4.0 (2026 年 5 月 15 日)
### 重要變更
- N/A
### 新增功能
- 【更新】改進 webui 頁面顯示效果，以及更多功能選項。
### 已知問題
- N/A

### 1.3.0 (2026 年 4 月 26 日)
### 重要變更
- 【重大】調整 tokenizer 新增加更多保底符號。
### 新增功能
- 【更新】改進 webui 頁面顯示效果，以及更多功能選項。
### 已知問題
- N/A

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
- 【錯誤】ByteLevel token 124 與 UTF-8 後半 Continuation Byte 重疊產生亂碼的問題。

### 1.2.0 (2026 年 4 月 17 日)
![t2i](https://raw.githubusercontent.com/AmanoShizukikun/Nagato-Sakura-Language-Model/refs/heads/main/assets/preview/1.2.0.jpg)
### 重要變更
- 【重大】首個公開版本。
### 新增功能
- 【新增】inference.py 新增臨時聊天模式。
- 【新增】inference.py 新增量化功能(由於參數過小不建議開啟)。
- 【修復】inference.py 額外輸入導致回復異常的錯誤。
- 【修復】train.py 輸出訊息會切斷進度條顯示的問題。
### 已知問題
- 【錯誤】ByteLevel token 124 與 UTF-8 後半 Continuation Byte 重疊產生亂碼的問題。
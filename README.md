# 語者分離 (Speaker-Separation)

**即時與離線多語者分離框架**

---

## 🔍 專案概覽
本專案採用 SpeechBrain 的 SepFormer 架構，實現 **三語者分離** 系統，支援 **離線批次訓練／推論** 以及 **即時串流分離**，能產出純淨的三路語者音軌，供後續語者辨識、自動語音辨識 (ASR)、情感分析等模組串流使用。

## 📂 專案結構
```
├── .vscode/                   # VSCode 工作區設定
├── 16k-models/                # 16 kHz 模型資料夾及主執行腳本
│   ├── separate_3.py          # 離線批次分離主程式
│   │ ...
├── Audios/                    # 錄音樣本（原始與處理後）
├── flowChart.drawio           # 系統架構圖
├── record.drawio              # 即時錄音與分離流程圖
├── recognition.py             # 語者辨識範例程式
├── record.py                  # 即時錄音與分離主程式
├── sep_3p.py                  # 三語者離線分離範例程式
├── test_GPU.py                # GPU 可用性測試
├── output_log.txt             # 訓練／推論日誌範例
├── requirements.txt           # 相依套件列表
└── README.md                  # 專案說明文件
```

## ⚙️ 功能特色
- **離線批次分離**：使用 `16k-models/separate_3.py`
- **即時串流分離**：透過 PyAudio + ThreadPoolExecutor 實現低延遲輸出
- **頻譜閘控降噪**：後處理一次性降噪以提升語音純淨度
- **雙模式取樣率**：16 kHz（高品質）與 8 kHz（低延遲）

## 🚀 安裝步驟
1. **複製專案**
   ```bash
   git clone https://github.com/MISCapstoneProject/Speaker-Separation.git
   cd Speaker-Separation
   ```
2. **建立虛擬環境**（建議）
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **安裝相依套件**
   ```bash
   pip install -r requirements.txt
   ```

## 🎬 使用說明

### 離線批次分離
```bash
# 使用主執行腳本 separate_3.py 對 WAV 檔案進行三語者分離
python 16k-models/separate_3.py --input path/to/mix.wav --output_dir outputs
```
執行完成後，`outputs/` 目錄下會產生 `speaker1.wav`、`speaker2.wav`、`speaker3.wav`。

- 採用 6 秒窗、50% 重疊
- 背景執行 SepFormer 分離任務
- 每段分離結果存於 `outputs/`

## 📈 效能評估
- **SI-SNR 提升**：在 Librimix 驗證集約 +12 dB
- **批次推論效能**：3 秒音訊／0.15 秒 GPU 時間
- **即時延遲**：RTX 4090 上約 0.3 秒（8 kHz）

## 🤝 貢獻指南
1. Fork 此專案
2. 建立功能分支
3. 提交改動
4. 開啟 Pull Request

請遵守專案 Python 程式碼風格，並為新增功能撰寫測試。

## 📜 授權資訊
本專案採用 **MIT License**，詳情請見 [LICENSE](LICENSE)。

---
*MISCapstoneProject © 2025*

# 語者分離 (Speaker-Separation)

**即時與離線多語者分離框架**

---

## 🔍 專案概覽
本專案使用 SpeechBrain 的 SepFormer 架構，實現 **三語者分離** 系統，支援 **離線批次訓練／推論** 以及 **即時串流分離**，可產出純淨語者音軌供後續語者辨識、自動語音辨識 (ASR)、情感分析等模組使用。

## 📂 專案結構
```
├── .vscode/                   # VSCode 工作區設定
├── 16K-model/                 # 16 kHz SepFormer 模型檢查點與設定
├── Audios/                    # 錄音樣本（原始與處理後）
├── flowChart.drawio           # 系統架構圖
├── record.drawio              # 即時錄音與分離流程圖
├── recognition.py             # 語者辨識範例程式
├── record.py                  # 即時錄音與分離主程式
├── separate.py                # 離線批次分離程式
├── sep_3p.py                  # 三語者分離範例程式
├── test_GPU.py                # GPU 可用性測試
├── output_log.txt             # 訓練／推論日誌範例
├── requirements.txt           # 相依套件列表
└── README.md                  # 專案說明文件
```

## ⚙️ 功能特色
- **離線訓練與推論**：結合動態混音 (Dynamic Mixing) 與 WHAM! 噪音增強
- **即時串流分離**：透過 PyAudio 及 ThreadPoolExecutor 實現低延遲輸出
- **頻譜閘控降噪**：後處理一次性降噪以提高純淨度
- **彈性取樣率**：16 kHz 高品質 vs. 8 kHz 低延遲
- **Permutation Invariant Training (PIT)** 搭配 SI-SNR 損失

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

### 1. 離線批次分離
```bash
# 將單一混合檔案分離為三個語者音軌
python separate.py --input path/to/mix.wav --output_dir outputs
```  
輸出：`speaker1.wav`、`speaker2.wav`、`speaker3.wav` 於 `outputs/`

### 2. 即時串流分離
```bash
# 啟動即時錄音並分離
python record.py --output_dir live_outputs
```  
- 錄製 6 秒窗，50% 重疊
- 背景執行 SepFormer 分離
- 輸出語者音軌至 `live_outputs/`

## 📈 效能評估
- **SI-SNR 提升**：在 Librimix 驗證集約 +12 dB
- **批次推論效能**：3 秒音訊 / 0.15 秒 GPU 時間
- **即時延遲**：RTX 4090 上約 0.3 秒（8 kHz）
- 詳細日誌與指標見 `output_log.txt`

## 🤝 貢獻指南
1. Fork 此專案
2. 建立功能分支
3. 提交改動
4. 開啟 Pull Request

請維持程式碼風格一致，並為新功能撰寫對應測試。

## 📜 授權資訊
本專案採用 **MIT License** 授權，詳情請見 [LICENSE](LICENSE)。

---
*MISCapstoneProject © 2025*

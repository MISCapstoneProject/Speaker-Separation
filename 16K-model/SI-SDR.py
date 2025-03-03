import torch
import torchaudio
import itertools
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio  # ✅ 正確的 import

# 🔹 SI-SDR 計算物件
si_sdr = ScaleInvariantSignalDistortionRatio()

# 🔹 讀取 3 個原始音檔（t1, t2, t3）
clean_speakers = [torchaudio.load(f"t{i+1}.wav")[0].mean(dim=0, keepdim=True) for i in range(3)]
sr_clean = torchaudio.load("t1.wav")[1]  # 取得採樣率

# 🔹 讀取 3 個分離後的音檔（s1, s2, s3）
est_speakers = [torchaudio.load(f"speaker{i+1}.wav")[0].mean(dim=0, keepdim=True) for i in range(3)]
sr_est = torchaudio.load("speaker1.wav")[1]

# 🔹 確保採樣率一致
if sr_clean != sr_est:
    resample = torchaudio.transforms.Resample(orig_freq=sr_est, new_freq=sr_clean)
    est_speakers = [resample(e) for e in est_speakers]

# 🔹 確保音訊長度一致（截斷較長的音檔）
min_len = min([c.shape[1] for c in clean_speakers] + [e.shape[1] for e in est_speakers])
clean_speakers = [c[:, :min_len] for c in clean_speakers]
est_speakers = [e[:, :min_len] for e in est_speakers]

# 🔹 嘗試所有排列組合，找出最佳 SI-SDR
best_sdr = -float("inf")
best_perm = None

for perm in itertools.permutations(range(3)):  # 共有 3! = 6 種排列組合
    total_sdr = sum(
        si_sdr(
            est_speakers[perm[i]].unsqueeze(0),
            clean_speakers[i].unsqueeze(0)
        ).item()
        for i in range(3)
    )
    if total_sdr > best_sdr:
        best_sdr = total_sdr
        best_perm = perm  # 記錄最佳匹配順序

# 🔹 輸出最佳排列組合
print(f"最佳匹配順序: {best_perm}")
print(f"最佳 SI-SDR (總和): {best_sdr:.2f} dB")

# 🔹 計算個別 SI-SDR
for i in range(3):
    si_sdr_val = si_sdr(
        est_speakers[best_perm[i]].unsqueeze(0),
        clean_speakers[i].unsqueeze(0)
    ).item()
    print(f"說話者 {i+1} 的 SI-SDR: {si_sdr_val:.2f} dB")

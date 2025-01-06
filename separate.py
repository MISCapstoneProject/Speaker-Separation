from speechbrain.inference import SepformerSeparation as seperator
import torchaudio

"""

分離模型(預訓練)
分離兩位語者的音訊
輸入：兩語者混合音訊(英文)
輸出：分離後的音訊(兩個音檔)

待研究議題：
分離後的排序問題(speaker1, speaker2), 誰是1 誰是2
有沒有辦法再錄音輸出音檔的同時進行語者分離
將分離後的音訊傳送到下一個section

對中文語者進行分離
"""

model = seperator.from_hparams(
    source='speechbrain/sepformer-wsj02mix',
    savedir='pretrained_models/sepformer-wsj02mix'
)

# 讀取檔案路徑
est_sources = model.separate_file(path='mixed_voice_noise.wav')
# print(f"分離後音頻形狀: {est_sources.shape}")

print(f"分離後音頻形狀: {est_sources.shape}")
print(f"分離後數據範圍: {est_sources.min().item()}, {est_sources.max().item()}")

# 提取語者數據並轉換為 [channels, samples] 格式
# speaker1 = est_sources[:, :, 0].detach().squeeze(0) # 去掉 batch 維度
# speaker2 = est_sources[:, :, 1].detach().squeeze(0)


# 確保數據形狀正確
# print(f"Speaker 1 形狀: {speaker1.shape}")
# print(f"Speaker 2 形狀: {speaker2.shape}")

# 分離語者
torchaudio.save("separate_voice/speaker1_noise.wav", est_sources[:, :, 0].detach().cpu(), 8000)
torchaudio.save("separate_voice/speaker2_noise.wav", est_sources[:, :, 1].detach().cpu(), 8000)

print("語者分離完成，已儲存 speaker1.wav 和 speaker2.wav")
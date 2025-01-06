import librosa
import soundfile as sf

def mix_speakers(speaker1_file, speaker2_file, output_file, ratio=0.5):
    # 讀取語者音頻
    y1, sr1 = librosa.load(speaker1_file, sr=None)
    y2, sr2 = librosa.load(speaker2_file, sr=None)

    # 確保取樣率相同
    assert sr1 == sr2, "取樣率不一致！"

    # 確保音頻長度一致
    min_len = min(len(y1), len(y2))
    y1, y2 = y1[:min_len], y2[:min_len]

    # 混合音頻
    mixed = ratio * y1 + (1 - ratio) * y2

    # 保存混合音頻
    sf.write(output_file, mixed, sr1)
    print(f"混合音頻已保存：{output_file}")

# 測試混合語者音頻
mix_speakers('speaker1.wav', 'speaker2.wav', 'mixed.wav', ratio=0.5)

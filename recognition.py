import torchaudio
import torch
from speechbrain.inference import EncoderClassifier
import numpy as np
from scipy.spatial.distance import cosine

class SpeakerSimilarity:
    def __init__(self):
        # 載入預訓練的語者辨識模型
        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )
    
    def get_embeddings(self, audio_path):
        """獲取音訊的語者嵌入向量"""
        # 讀取音訊
        signal, fs = torchaudio.load(audio_path)
        
        # 確保採樣率正確（模型期望16kHz）
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(fs, 16000)
            signal = resampler(signal)
        
        # 如果是立體聲，轉換為單聲道
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        
        # 提取嵌入向量
        embeddings = self.encoder.encode_batch(signal)
        return embeddings.squeeze()
    
    def compute_similarity(self, audio_path1, audio_path2):
        """計算兩個音訊的相似度"""
        # 獲取兩個音訊的嵌入向量
        emb1 = self.get_embeddings(audio_path1)
        emb2 = self.get_embeddings(audio_path2)
        
        # 計算餘弦相似度（1表示完全相同，0表示完全不同）
        similarity = 1 - cosine(emb1.cpu().numpy(), emb2.cpu().numpy())
        return similarity
    
    def is_same_speaker(self, similarity, threshold=0.75):
        """根據相似度判斷是否為同一個說話者"""
        return similarity > threshold


def main():
    speaker_similarity = SpeakerSimilarity()
    
    # 比較兩個音訊檔的相似度
    audio1_path = "separate_voice/speaker1-0204.wav"
    audio2_path = "Audios\output_separatedAudio\speaker1_20250204-19_26_30_2.wav"
    
    try:
        similarity = speaker_similarity.compute_similarity(audio1_path, audio2_path)
        print(f"語者相似度: {similarity:.4f}")
        
        # 判斷是否為同一位語者
        is_same = speaker_similarity.is_same_speaker(similarity)
        if is_same:
            print("判斷結果：很可能是同一位語者")
        else:
            print("判斷結果：應該是不同語者")
            
    except Exception as e:
        print(f"處理過程中發生錯誤: {str(e)}")

if __name__ == "__main__":
    main()
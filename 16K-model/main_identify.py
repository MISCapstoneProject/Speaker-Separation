"""
語音辨識 檔案介紹

    使用方式: 
        呼叫process_audio_file(audio_file)函式，audio_file須為wav檔，程式會抓前10秒聲音進行判斷

        在此檔案最下方的242行，是程式使用的範例，可以直接更改使用
    
    產生結果:
        1. 新音檔會與資料夾裡的每個已保存的嵌入向量檔，產生各自的餘弦距離
        2. 如果找出距離夠近的檔案(在設定閾值內)，就判斷為與此檔案是同一個人的聲音

        如果與現有的向量都距離太遠，就會判斷為是新的人，並儲存成新的未命名向量檔案
        (新檔案前半段的檔名可改為人名)

        
    51~56行: 設定儲存嵌入向量的資料夾位置，以及聲音距離的閾值設定

    嵌入向量檔案命名規則:
        新檔案名稱: n1xx_2.npy
            n: 開頭n的檔案為新的人，新增加的陌生聲音還未命名此人是誰，所以代表是未命名檔案
            1: 未命名檔案的序號，假設當前有n1xx, n2xx, n4xx，下一個新建的未命名檔案為n5xx
            xx: 隨機的兩位小寫字母，所以可能會是rn或gs等等  (一開始是怕未命名檔案n+序號在改名之後新檔案可能會撞到名稱，現在改善後並不會衝突，因此是可以刪掉這隨機兩位功能)
            _: 底線是做區隔，底線前的可以隨意更改改為人名，像是A、B、C等等
            2: 數字用來代表做了幾次加權移動平均，因此數字會隨更新次數而增加 (目前設計是更新嵌入向量時會+1，且對現有的嵌入向量檔案做平均)

        自由命名檔名: 小明_2.npy, David_6.npy
            可以把_底線前面的檔名做更動
            !底線與後方數字一定要有，且數字不建議自由更改!
"""

import os
import random
import re
import string
import sys
import numpy as np
import torch
import torchaudio
import soundfile # torchaudio套件需要
from numpy.linalg import norm
from scipy.spatial.distance import cosine
import warnings
import logging
import subprocess  # 引入 subprocess 模組來呼叫 ffmpeg

# 隱藏多餘的警告和日誌
warnings.filterwarnings("ignore")
logging.getLogger("speechbrain").setLevel(logging.ERROR)

# 自定義 Tee 類別
class Tee:
    def __init__(self, file_name, mode="w"):
        self.file = open(file_name, mode, encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, message):
        self.file.write(message)
        self.stdout.write(message)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

# 開始重定向輸出
sys.stdout = Tee("output_log.txt")

# 導入模型
from speechbrain.inference import SpeakerRecognition

# 嵌入向量存放的目錄
EMBEDDING_DIR = "embeddingFiles"
# 設定距離閾值
THRESHOLD_LOW = 0.2  # 過於相似的閾值，小於此值的嵌入向量不更新
THRESHOLD_UPDATE = 0.34  # 進行更新的最大距離閾值
THRESHOLD_NEW = 0.36  # 超過此距離，視為新說話者


# 嘗試載入 SpeechBrain 語音辨識模型
try:
    model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",  # SpeechBrain 預訓練模型的來源
        savedir="models/speechbrain_recognition"     # 模型儲存的路徑
    )
    print("SpeechBrain 模型加載成功！")
except ImportError:
    print("SpeechBrain 未正確安裝，請運行: pip install speechbrain")
    exit()


def resample_with_ffmpeg(input_path, output_path, target_sr):
    """
    使用 ffmpeg 重新採樣音訊
    """
    # 使用 subprocess.run 呼叫 ffmpeg 來進行音訊重新採樣
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path, "-ar", str(target_sr),"-ac", "1", output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # 靜默輸出，不顯示任何資訊

def extract_embedding(audio_path):
    """
    從音檔中提取語音嵌入向量 (embedding)。
    - 限制音檔的最大長度為 10 秒。
    - 先將音訊降頻到 8kHz，再升頻回 16kHz。

    參數:
    audio_path (str): 音檔的路徑

    回傳:
    np.ndarray: 提取的語音嵌入向量
    """
    temp_8k_path = "temp_8k.wav"  # 8kHz 音訊檔案的暫存路徑
    temp_16k_path = "temp_16k.wav"  # 16kHz 音訊檔案的暫存路徑
    
    # 降頻到 8kHz
    resample_with_ffmpeg(audio_path, temp_8k_path, target_sr=8000)
    
    # 再升頻到 16kHz
    resample_with_ffmpeg(temp_8k_path, temp_16k_path, target_sr=16000)
    
    # 讀取 16kHz 音檔
    signal, fs = torchaudio.load(temp_16k_path)
    
    # 限制音訊檔案的最大長度（10 秒）
    max_length = fs * 10  # 計算 10 秒對應的樣本數
    signal = signal[:, :max_length] if signal.shape[1] > max_length else signal  # 如果長度超過 10 秒，截取前 10 秒
    
    # print(f"Final sampling rate: {fs}, shape: {signal.shape}")  # 顯示最終的取樣率和音訊的形狀
    
    # 使用 SpeechBrain 模型提取嵌入向量
    embedding = model.encode_batch(signal).squeeze().numpy()
    
    # # 清理暫存檔案
    # os.remove(temp_8k_path)  # 刪除 8kHz 音訊檔案
    # os.remove(temp_16k_path)  # 刪除 16kHz 音訊檔案
    
    return embedding  # 回傳提取的語音嵌入向量



def compare_all_npy(new_embedding):
    """
    將新嵌入向量與目錄中的所有現有 `.npy` 檔案進行比較。
    - 計算每個檔案的餘弦距離，並返回距離結果。
    - 測試用功能：顯示每個檔案與新嵌入的距離。

    參數:
    new_embedding (np.ndarray): 新提取的語音嵌入向量

    回傳:
    str: 與新嵌入最相近的檔案名稱
    float: 最小的餘弦距離
    list: 所有檔案的距離列表
    """

    # 確保嵌入向量目錄存在
    if not os.path.exists(EMBEDDING_DIR):
        os.makedirs(EMBEDDING_DIR)
    
    # 搜索目錄中的 .npy 檔案
    npy_files = [f for f in os.listdir(EMBEDDING_DIR) if f.endswith(".npy")]
    if not npy_files:
        print("No existing .npy files found in embedding directory.")
        return None, float('inf'), []

    # 記錄所有距離的列表
    distances = []
    print("=== Comparing with existing embeddings ===")  # 測試用，顯示所有比較結果
    for npy_file in npy_files:
        file_path = os.path.join(EMBEDDING_DIR, npy_file)
        saved_embedding = np.load(file_path)  # 加載保存的嵌入向量

        # 計算新嵌入與現有嵌入的餘弦距離
        distance = cosine(new_embedding, saved_embedding)
        distances.append((npy_file, distance))
        print(f"File: {npy_file}, Distance: {distance:.4f}")  # 測試用，逐一顯示距離
    
    # 找到距離最小的檔案
    best_match = min(distances, key=lambda x: x[1])
    return best_match[0], best_match[1], distances


def update_embedding(old_embedding, new_embedding, n_samples):
    """
    使用加權移動平均更新嵌入向量。
    - 更新過程中考慮已存在樣本的數量，對新向量進行加權。

    參數:
    old_embedding (np.ndarray): 現有的嵌入向量
    new_embedding (np.ndarray): 新的嵌入向量
    n_samples (int): 已經累積的樣本數

    回傳:
    np.ndarray: 更新後的嵌入向量
    int: 更新後的樣本數量
    """
    updated_embedding = (old_embedding * n_samples + new_embedding) / (n_samples + 1)
    return updated_embedding, n_samples + 1

def generate_unique_filename(directory):
    """
    根據資料夾中的檔案，生成新檔名：
    - 開頭為 'n'，接著數字遞增。
    - 後面接兩位隨機小寫英文字母。
    """
    
    # 獲取資料夾中所有以 'n' 開頭且是 `.npy` 的檔案
    files = [f for f in os.listdir(directory) if f.startswith('n') and f.endswith('.npy')]

    # 提取檔案名稱中的數字部分，忽略後綴和其他非數字部分
    numbers = []
    for f in files:
        parts = f.split('_')[0]  # 取檔名 "_" 前的部分
        # 使用正規表達式提取前面的數字
        match = re.search(r'(\d+)', parts)
        if match:
            numbers.append(int(match.group(1)))  # 提取的數字轉為整數並加入到 numbers 列表中
            # print(match.group(1))

    # 確定新檔案的數字部分
    next_number = max(numbers, default=0) + 1

    # 生成隨機小寫字母後綴
    random_suffix = ''.join(random.choices(string.ascii_lowercase, k=2))

    # 返回新檔案名稱
    return f"n{next_number}{random_suffix}"


def process_audio_file(audio_file):
    """
    主邏輯函式，處理導入音檔並進行判斷。
    - 提取嵌入向量並與現有的 `.npy` 比較。
    - 判斷是否更新嵌入向量或創建新的說話者資料。

    參數:
    audio_file (str): 音檔的路徑
    """
    print(f"\nProcessing file: {audio_file}")  # 測試用，顯示導入的音檔名稱
    # 檢查檔案是否存在
    if not os.path.exists(audio_file):
        print(f"File {audio_file} does not exist. Cancel.")
        return  # 如果檔案不存在，就取消

    # 建立新音檔嵌入向量
    new_embedding = extract_embedding(audio_file)
    
    # 比較新嵌入與現有嵌入向量
    match_file, best_distance, all_distances = compare_all_npy(new_embedding)
    
    if best_distance < THRESHOLD_LOW:
        # 如果距離小於低閾值，視為過於相似，跳過更新
        print(f"Embedding too similar (distance = {best_distance:.4f}), skipping update.")
        print(f"此音檔與{match_file}是同一個人")
    elif best_distance < THRESHOLD_UPDATE:
        # 距離在合理範圍內，進行嵌入更新
        old_embedding = np.load(os.path.join(EMBEDDING_DIR, match_file))
        n_samples = int(match_file.split("_")[-1].split(".")[0])  # 提取樣本數量
        updated_embedding, updated_samples = update_embedding(old_embedding, new_embedding, n_samples)
        
        # 保存更新後的嵌入向量，並刪除舊檔案
        new_filename = f"{match_file.split('_')[0]}_{updated_samples}.npy"
        np.save(os.path.join(EMBEDDING_DIR, new_filename), updated_embedding)
        os.remove(os.path.join(EMBEDDING_DIR, match_file))
        print(f"Updated: {match_file} -> {new_filename}")
        print(f"此音檔與{match_file}是同一個人")
    elif best_distance < THRESHOLD_NEW:
        # 距離接近匹配但不更新
        print(f"Matched with {match_file}, but no update performed (distance = {best_distance:.4f}).")
        print(f"此音檔與{match_file}是同一個人")
    else:
        # 如果距離超過新說話者閾值，新增新檔案
        new_filename = f"{generate_unique_filename(EMBEDDING_DIR)}_1.npy"
        np.save(os.path.join(EMBEDDING_DIR, new_filename), new_embedding)
        print(f"New speaker detected, saved as: {new_filename}")
        print(f"發現新的人，將此聲音保存至{new_filename}資料")

def process_audio_directory(directory):
    """
    處理資料夾中的所有音檔。
    - 將資料夾內的所有 .wav 檔案逐一進行語音處理。

    參數:
    directory (str): 資料夾的路徑
    """
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist. Cancel.")
        return

    # 獲取資料夾內所有 .wav 檔案
    audio_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".wav")]

    if not audio_files:
        print(f"No .wav files found in directory {directory}.")
        return

    print(f"Found {len(audio_files)} audio files in directory {directory}. Starting processing...")

    # 逐一處理每個音檔
    for audio_file in audio_files:
        try:
            process_audio_file(audio_file)
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")

    print(f"\nFinished processing all audio files in directory {directory}.")


if __name__ == "__main__":
    # test_audio_file = "audiofile/5-1.wav"  # 新音檔路徑
    # process_audio_file(test_audio_file)
    # print()

    test_directory = "test_audioFile"  # 測試資料夾路徑
    process_audio_directory(test_directory)  # 處理資料夾內的所有音檔
    print()
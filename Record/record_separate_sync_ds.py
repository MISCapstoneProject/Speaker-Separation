import os
import pyaudio
import numpy as np
import threading
from datetime import datetime
from speechbrain.inference import SepformerSeparation as separator
import torch
import torchaudio
from concurrent.futures import ThreadPoolExecutor

"""
record_separated_sync_ds.py
在根目錄下執行: python Record/record_separate_sync_ds.py

與 record_separated_sync.py 類似，一樣是邊錄音邊進行分離
但切割時間變為非固定的方式。改以動態切割策略來分割音訊 frame
即基於靜音持續時間動態確定分段
靜音的閾值和最小持續時間 (SILENCE_THRESHOLD 和 MIN_SILENCE_LEN) 可以根據實驗調整

每次完成動態切割後，立即啟動一個新執行緒進行語者分離

優化方向：

靜音閾值調整(已實作):
可以使用工具（如 librosa 的 RMS 檢測）計算背景噪音的能量範圍，調整 SILENCE_THRESHOLD。

靜音過濾與調整:
如果某些語句被誤切，可以在後處理階段進行重新合併（如檢測無聲間隔是否小於某個閾值，將短音段拼接）。

即時性測試:
確保語者分離的執行緒不影響錄音的連續性。
"""

# 錄音參數設定
CHUNK = 1024  # 每次從輸入串流讀取的音框大小
FORMAT = pyaudio.paInt16  # 音訊格式
CHANNELS = 2  # 立體聲
RATE = 44100  # 取樣頻率
MIN_SILENCE_LEN = 1.5  # 靜音的最小持續時間（秒）
DEVICE_INDEX = None  # 音訊裝置索引

# 最大同時執行的語者分離執行緒數量
MAX_THREADS = 2
semaphore = threading.Semaphore(MAX_THREADS)

# 初始化語者分離模型
model = separator.from_hparams(
    source="speechbrain/sepformer-wsj02mix",
    savedir="pretrained_models/sepformer-wsj02mix"
)

# 用於存儲語者分離執行緒
threads = []
executor = ThreadPoolExecutor(max_workers=MAX_THREADS)

def calibrate_silence_threshold(audio_stream, calibration_duration=2):
    """
    根據背景噪音自動校準靜音能量閾值
    """
    print("校準靜音閾值...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * calibration_duration)):
        data = audio_stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.int16))

    # 計算背景噪音的平均能量
    background_noise = np.concatenate(frames)
    rms_energy = np.mean(np.abs(background_noise))
    
    # 動態設置靜音閾值
    silence_threshold = rms_energy * 1.5  # 可調整倍率
    print(f"校準完成，靜音能量閾值設置為：{silence_threshold}")
    return silence_threshold

def dynamic_cut_and_stop(output_dir, silence_threshold, max_silence_duration=5):
    """
    動態切割錄音，若長時間無語音則自動停止
    """
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=DEVICE_INDEX
    )
    print("開始錄音，按 Ctrl+C 停止。")

    frames = []  # 儲存當前段的音訊
    silent_duration = 0  # 無語音累計時間
    active_duration = 0  # 有效音訊累計時間
    segment_index = 1  # 音訊段索引

    try:
        while True:
            # 獲取一個 CHUNK 的音訊
            data = stream.read(CHUNK)
            frame = np.frombuffer(data, dtype=np.int16)
            frames.append(frame)

            # 判斷是否為靜音
            if np.mean(np.abs(frame)) < silence_threshold:
                silent_duration += CHUNK / RATE
            else:
                silent_duration = 0
                active_duration += CHUNK / RATE  # 累計有效音訊時長

            # 檢查是否達到靜音條件並且有效音訊累積到足夠長度
            if silent_duration >= MIN_SILENCE_LEN and active_duration >= MIN_SILENCE_LEN:
                print("偵測到靜音段，準備進行語者分離處理")
                audio_data = np.concatenate(frames)
                audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)

                # 語者分離處理
                # 提交語者分離任務到執行緒池
                executor.submit(process_and_save, audio_tensor, output_dir, segment_index)

                # 清空 frames，準備下一段錄音
                frames = []
                segment_index += 1
                active_duration = 0  # 重置有效音訊累積時間

            # 如果無語音時間超過限制，停止錄音
            # if silent_duration >= max_silence_duration:
            #     print(f"超過 {max_silence_duration} 秒無語音，錄音自動結束。")
            #     break

    except KeyboardInterrupt:
        print("\n錄音停止。")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

        # 等待所有語者分離執行緒完成
        print("等待語者分離執行緒完成...")
        executor.shutdown(wait=True)
        print("所有語者分離執行緒完成。程式結束。")

def process_and_save(audio_tensor, output_dir, segment_index):
    """
    處理語者分離並儲存結果
    """
    print("開始進行語者分離處理")
    try:
        # # 確保輸入格式正確
        # if len(audio_tensor.shape) == 1:
        #     audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # 調整為 [batch_size, time]

        # 語者分離
        est_sources = model.separate_batch(audio_tensor)

        # 儲存分離結果
        timestamp_str = datetime.now().strftime('%Y%m%d-%H_%M_%S')
        speaker1_file = os.path.join(output_dir, f"speaker1_{timestamp_str}_{segment_index}.wav")
        speaker2_file = os.path.join(output_dir, f"speaker2_{timestamp_str}_{segment_index}.wav")
        
        # 確保只生成兩個檔案
        if est_sources.size(2) == 2:
            torchaudio.save(speaker1_file, est_sources[:, :, 0].detach().cpu(), 8000)
            torchaudio.save(speaker2_file, est_sources[:, :, 1].detach().cpu(), 8000)
            print(f"語者分離完成：{speaker1_file}, {speaker2_file}")
        else:
            print(f"語者分離失敗，輸出數量異常：{est_sources.size(2)} 個音軌")

    except Exception as e:
        print(f"語者分離時出錯：{e}")

def main():
    output_dir = "Audios/output_separatedAudio"
    os.makedirs(output_dir, exist_ok=True)

    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=DEVICE_INDEX
    )
    
    silence_threshold = calibrate_silence_threshold(stream)
    dynamic_cut_and_stop(output_dir, silence_threshold)

if __name__ == "__main__":
    main()

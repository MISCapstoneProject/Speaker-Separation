import os
import pyaudio
import numpy as np
import threading
from datetime import datetime
from speechbrain.inference import SepformerSeparation as separator
import torch
import torchaudio

"""

record_separate_sync.py
在根目錄下執行: python Record/record_separate_sync.py
邊錄音邊分段，並即時進行語者分離

錄音與分段儲存：
每次錄音 RECORD_SECONDS 秒（預設 5 秒）
將錄音的音訊 frame 作為輸入，傳遞給語者分離模型進行處理

語者分離（多執行緒）：
輸入音訊 frame 
語者分離的結果以 speaker1_...wav 和 speaker2_...wav 的格式儲存。

並行處理：
錄音與語者分離在不同的執行緒中運行，確保錄音不會因語者分離的耗時而中斷。
若錄音速度過快（例如每秒生成檔案），可以考慮使用工作佇列(如 queue.Queue) 或非同步處理進行調度

"""

# 錄音參數
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5  # 每段錄音秒數
DEVICE_INDEX = None

# 初始化語者分離模型
model = separator.from_hparams(
    source="speechbrain/sepformer-wsj02mix",
    savedir="pretrained_models/sepformer-wsj02mix"
)

def record_audio_and_separate(output_dir):
    """
    錄音並以 frame 為單位同步進行語者分離，最後儲存每位語者的獨立音檔
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

    segment_index = 1
    try:
        while True:
            frames = []
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(np.frombuffer(data, dtype=np.int16))

            # 將音訊 frames 合併為 numpy array
            audio_data = np.concatenate(frames)
            audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)

            # 語者分離
            print(f"開始語者分離：段落 {segment_index}")
            threading.Thread(target=process_and_save, args=(audio_tensor, output_dir, segment_index)).start()
            segment_index += 1

    except KeyboardInterrupt:
        print("\n錄音停止。")
        stream.stop_stream()
        stream.close()
        p.terminate()

def process_and_save(audio_tensor, output_dir, segment_index):
    """
    處理語者分離並儲存結果
    """
    try:
        est_sources = model.separate_batch(audio_tensor, sample_rate=RATE)

        # 儲存分離結果
        timestamp_str = datetime.now().strftime('%Y%m%d-%H_%M_%S')
        speaker1_file = os.path.join(output_dir, f"speaker1_{timestamp_str}_{segment_index}.wav")
        speaker2_file = os.path.join(output_dir, f"speaker2_{timestamp_str}_{segment_index}.wav")

        torchaudio.save(speaker1_file, est_sources[:, :, 0].detach().cpu(), RATE)
        torchaudio.save(speaker2_file, est_sources[:, :, 1].detach().cpu(), RATE)

        print(f"語者分離完成，結果已儲存：{speaker1_file}, {speaker2_file}")
    except Exception as e:
        print(f"語者分離時出錯：{e}")

def main():
    output_dir = "Audios/output_separatedAudio"
    os.makedirs(output_dir, exist_ok=True)

    record_audio_and_separate(output_dir)

if __name__ == "__main__":
    main()

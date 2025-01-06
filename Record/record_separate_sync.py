import os
import pyaudio
import wave
import numpy as np
import threading
from datetime import datetime
from speechbrain.inference import SepformerSeparation as separator
import torchaudio

"""
record_separate_sync.py
邊錄音邊分段，並即時進行語者分離

錄音與分段儲存：

每次錄音 RECORD_SECONDS 秒（預設 5 秒），儲存為獨立的 .wav 檔案。
檔案名稱包含時間戳，確保不會覆蓋。
語者分離（多執行緒）：

每儲存一個音檔後，啟動一個執行緒進行語者分離。
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

def record_audio_and_save(output_dir):
    """
    錄音並將每段儲存為獨立檔案
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
                frames.append(data)

            # 儲存錄音檔
            timestamp_str = datetime.now().strftime('%Y%m%d-%H_%M_%S')
            raw_filename = os.path.join('Audios/rawAudioFile', f"raw_audio_{timestamp_str}_{segment_index}.wav")
            wf = wave.open(raw_filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            print(f"錄音檔已儲存：{raw_filename}")

            # 啟動語者分離的執行緒
            threading.Thread(target=separate_speakers, args=(raw_filename, output_dir)).start()
            segment_index += 1

    except KeyboardInterrupt:
        print("\n錄音停止。")
        stream.stop_stream()
        stream.close()
        p.terminate()

def separate_speakers(input_file, output_dir):
    """
    對錄製的音檔進行語者分離
    """
    try:
        print(f"開始語者分離：{input_file}")
        est_sources = model.separate_file(path=input_file)

        # 儲存分離結果
        speaker1_file = os.path.join(output_dir, f"speaker1_{os.path.basename(input_file)}")
        speaker2_file = os.path.join(output_dir, f"speaker2_{os.path.basename(input_file)}")
        torchaudio.save(speaker1_file, est_sources[:, :, 0].detach().cpu(), 8000)
        torchaudio.save(speaker2_file, est_sources[:, :, 1].detach().cpu(), 8000)

        print(f"語者分離完成，結果已儲存：{speaker1_file}, {speaker2_file}")
    except Exception as e:
        print(f"語者分離時出錯：{e}")

def main():
    output_dir = "Audios/output_separatedAudio"
    os.makedirs(output_dir, exist_ok=True)

    record_audio_and_save(output_dir)

if __name__ == "__main__":
    main()

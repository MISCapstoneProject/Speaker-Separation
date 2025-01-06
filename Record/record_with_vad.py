import pyaudio
import wave
import numpy as np
from datetime import datetime

"""
錄音程式, 新增VAD機制
即如果超過3秒沒有偵測到聲音訊號
就會停止錄音並輸出，再重新錄製新音訊
如果沒有超過3秒無聲訊, 就會錄滿10秒後輸出
"""

CHUNK = 1024             # 每次從輸入串流讀取的 frame 數量
FORMAT = pyaudio.paInt16 # 音訊格式 (Int16)
CHANNELS = 2             # 聲道數
RATE = 44100             # 取樣頻率
RECORD_SECONDS = 10       # 每段最長錄音秒數(若沒有靜音 3 秒，就會錄滿10秒)
DEVICE_INDEX = None      # 若有多個音訊裝置，可指定裝置編號，None 表示使用預設裝置

# ==== 簡易 VAD 參數 ====
THRESHOLD = 500      # 能量閾值，小於此值視為靜音(依實際環境調整)
MAX_SILENCE_SEC = 3   # 連續靜音達 3 秒就提早結束該段錄音
# ======================

def record_segment_with_vad(audio_stream):
    """
    從 audio_stream 連續錄音，
    1) 若連續靜音達 MAX_SILENCE_SEC 就停止並回傳 frames
    2) 否則錄到 RECORD_SECONDS 滿為止
    """
    frames = []
    # 總共要讀多少次 CHUNK (若沒有提前靜音結束)
    frames_to_read = int(RATE / CHUNK * RECORD_SECONDS)

    # 每個 CHUNK 時間 (秒數)
    chunk_duration = CHUNK / RATE

    silence_time = 0.0
    chunk_count = 0

    while chunk_count < frames_to_read:
        data = audio_stream.read(CHUNK)
        frames.append(data)

        # 把 bytes 轉成 numpy array 以計算音量
        audio_data = np.frombuffer(data, dtype=np.int16)
        max_amplitude = np.max(np.abs(audio_data))

        if max_amplitude < THRESHOLD:
            # 低於閾值 -> 累積靜音時間
            silence_time += chunk_duration
        else:
            # 大於閾值 -> 重置靜音計時
            silence_time = 0.0

        # 若連續靜音時間達到 MAX_SILENCE_SEC，提早結束該段錄音
        if silence_time >= MAX_SILENCE_SEC:
            print(f"偵測到連續 {MAX_SILENCE_SEC} 秒靜音，提早結束該段錄音")
            break

        chunk_count += 1

    return frames

def save_wav_file(filename, frames, p):
    """
    將 frames 寫入 .wav 檔案
    """
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def main():
    p = pyaudio.PyAudio()

    # 開啟輸入串流
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=DEVICE_INDEX
    )

    print(
        "開始連續錄音：\n"
        f" - 每段最多錄 {RECORD_SECONDS} 秒\n"
        f" - 若連續靜音 {MAX_SILENCE_SEC} 秒則提早結束並輸出\n"
        "按 Ctrl+C 停止程式。\n"
    )

    segment_index = 1

    try:
        while True:
            # 先錄製一段音訊 (最長5秒，若連續3秒靜音則提早結束)
            frames = record_segment_with_vad(stream)

            # 用當下時間做檔名，避免重複 (Windows 不建議用冒號 ':')
            timestamp_str = datetime.now().strftime('%Y%m%d-%H_%M_%S')
            filename = f"Audios/outputFile_VAD/audio_{timestamp_str}-{segment_index}.wav"

            # 存成 WAV
            save_wav_file(filename, frames, p)

            print(f"第 {segment_index} 段錄音已儲存：{filename}")
            segment_index += 1

            # 接著馬上進入下一迴圈，開始錄下一段

    except KeyboardInterrupt:
        print("\n停止錄音。")

    # 收尾動作
    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    main()

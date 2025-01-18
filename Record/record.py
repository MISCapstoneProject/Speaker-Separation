import pyaudio
import wave
from datetime import datetime
import os

"""
在根目錄下執行: python Record/record.py
基礎錄音程式, 每隔5秒切割一次音檔 (儲存至outputFile)
按下CTRL+C 停止錄音
"""

CHUNK = 1024                    # 每次從輸入串流讀取的 frame 數量
FORMAT = pyaudio.paInt16        # 音訊格式
CHANNELS = 2                    # 立體聲 (2 聲道)
RATE = 44100                    # 取樣頻率
RECORD_SECONDS = 30              # 每段錄音秒數
DEVICE_INDEX = None             # 若有多個音訊裝置，可指定裝置編號，None 表示使用預設裝置

def record_segment(audio_stream):
    """
    從 audio_stream 連續讀取 RECORD_SECONDS 秒的音訊，回傳 frames (bytes list)
    """
    frames = []
    frames_to_read = int(RATE / CHUNK * RECORD_SECONDS)

    for _ in range(frames_to_read):
        data = audio_stream.read(CHUNK)
        frames.append(data)

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

    print("開始連續錄音，每段 {} 秒...按 Ctrl+C 停止。".format(RECORD_SECONDS))

    segment_index = 1

    try:
        while True:
            # 1) 先錄製一段音訊 (5秒)
            frames = record_segment(stream)

            # 2) 將剛錄製好的音訊存檔 (wav)
            # 取得當前時間，格式化為 yyyyMMdd-HH:mm:ss
            timestamp_str = datetime.now().strftime('%Y%m%d-%H_%M_%S')
            filename = os.path.join("Audios", "outputFile", f"audio_{timestamp_str}-{segment_index}.wav")
            save_wav_file(filename, frames, p)

            print(f"第 {segment_index} 段錄音已儲存：{filename}")
            segment_index += 1

            # 馬上進入下一圈迴圈，開始錄下一段

    except KeyboardInterrupt:
        print("\n停止錄音。")

    # 收尾動作
    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    main()

import os
import pyaudio
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from speechbrain.inference import SepformerSeparation as separator
import torch
import torchaudio
import logging

# ========== 使用者可自行調整的參數 ==========

CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100

# 分段長度 (秒) 與重疊長度 (秒)
WINDOW_SIZE = 6.0     
OVERLAP = 1.0        

DEVICE_INDEX = None   # 如果要指定裝置索引，填數字；若不指定則用預設麥克風

# 重新取樣
TARGET_RATE = 8000    # Sepformer 原始預訓練使用 8kHz 或 16kHz，可自行測試

# 能量閾值 (過低音訊直接忽略，避免空段處理)
MIN_ENERGY_THRESHOLD = 0.005  

# 輸出資料夾
OUTPUT_DIR = "16K-model/Audios-16K"

# 日誌設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cross_fade_concat(existing_signal: torch.Tensor, new_signal: torch.Tensor, overlap_samples: int) -> torch.Tensor:
    """
    將 new_signal 與 existing_signal 做 cross-fade 拼接，其中最後 overlap_samples 與
    new_signal 前 overlap_samples 的區段做淡入淡出混合。
    皆為單聲道張量 (shape=[time])。
    
    回傳：拼接後的張量 (shape=[time])。
    """
    if existing_signal is None or len(existing_signal) == 0:
        return new_signal
    
    # 如果沒有重疊，直接串接
    if overlap_samples <= 0:
        return torch.cat((existing_signal, new_signal), dim=0)
    
    # 若現有訊號長度不夠 overlap_samples，則縮小 overlap
    overlap_samples = min(overlap_samples, existing_signal.shape[0], new_signal.shape[0])
    
    # 拆分 overlap 區段
    old_overlap = existing_signal[-overlap_samples:]   # 現有訊號的最後 overlap_samples
    new_overlap = new_signal[:overlap_samples]         # 新訊號的前 overlap_samples
    
    # 產生淡入淡出權重 (線性)
    fade_out = torch.linspace(1.0, 0.0, steps=overlap_samples)
    fade_in = 1.0 - fade_out
    
    # 混合 overlap
    mixed_overlap = old_overlap * fade_out + new_overlap * fade_in
    
    # 組合最終結果
    out_signal = torch.cat([
        existing_signal[:-overlap_samples],  # 不包含最後的overlap
        mixed_overlap,                       # 混合後的overlap區段
        new_signal[overlap_samples:]         # new_signal剩下部分
    ], dim=0)
    
    return out_signal

def postprocess_and_silence_empty_tracks(speaker_buffers, silence_threshold=0.005):
    """
    根據能量檢測，將近乎空白的音軌清零。
    speaker_buffers: dict, key=說話者index, value=Tensor shape=[time]
    silence_threshold: 能量門檻
    """
    for spk_idx, wave_tensor in speaker_buffers.items():
        if wave_tensor is None or len(wave_tensor) == 0:
            continue

        # 計算平均絕對值做為能量指標
        avg_amplitude = torch.mean(torch.abs(wave_tensor))
        
        if avg_amplitude < silence_threshold:
            print(f"Speaker {spk_idx} 低於閾值 ({avg_amplitude:.6f} < {silence_threshold}), 視為空白，直接靜音化")
            speaker_buffers[spk_idx] = torch.zeros_like(wave_tensor)
    return speaker_buffers


class AudioSeparator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用設備: {self.device}")

        # 載入 Sepformer 模型
        self.model = separator.from_hparams(
            source="speechbrain/sepformer-libri3mix",
            savedir='pretrained_models/sepformer-libri3mix',
            run_opts={"device": self.device}
        )

        # 重新取樣器
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=RATE,
            new_freq=TARGET_RATE
        ).to(self.device)

        # 多執行緒，避免錄音與分離互相阻塞
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.futures = []

        # 用來儲存「每個說話者」的最終音訊 (Tensor)
        # 例如: self.speaker_buffers[0] = (shape=[time])，對應 speaker1 的累積結果
        self.speaker_buffers = {}

        self.is_recording = False
        logger.info("AudioSeparator 初始化完成")

    def process_audio(self, audio_data):
        """ 基本音訊處理：轉 float32、檢查能量、維度調整、重取樣 """
        try:
            # 轉 float32
            audio_float = audio_data.astype(np.float32)

            # 能量檢測
            energy = np.mean(np.abs(audio_float))
            if energy < MIN_ENERGY_THRESHOLD:
                logger.debug(f"音訊能量 ({energy}) 低於閾值 ({MIN_ENERGY_THRESHOLD})")
                return None

            # shape => [num_samples, channels]
            if len(audio_float.shape) == 1:
                audio_float = audio_float.reshape(-1, CHANNELS)

            audio_tensor = torch.from_numpy(audio_float).T.float()  # [channels, time]

            # 若有雙聲道但模型只支援單聲道 -> 做平均
            if audio_tensor.shape[0] == 2:
                audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)

            # 移到裝置 & 重新取樣
            audio_tensor = audio_tensor.to(self.device)
            resampled = self.resampler(audio_tensor)  # [1, time]

            return resampled
        except Exception as e:
            logger.error(f"音訊處理錯誤：{e}")
            return None

    def record_and_process(self):
        """ 邊錄音邊分段分離，最後在記憶體做 cross-fade 拼接，程式結束再輸出檔案 """
        mixed_audio_buffer = []
        
        # 建立輸出資料夾
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=DEVICE_INDEX
            )
            logger.info("開始錄音")

            # 計算每個 window 與 overlap
            samples_per_window = int(WINDOW_SIZE * RATE)  # 6秒 * 44100 = 264600 samples
            window_frames = samples_per_window // CHUNK   # 264600 / 1024 ≈ 258.984 (約258)
            overlap_samples = int(OVERLAP * RATE)         # 1秒 * 44100 = 44100
            chunk_buffer = []

            segment_index = 0
            self.is_recording = True

            while self.is_recording:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frame = np.frombuffer(data, dtype=np.float32)
                    chunk_buffer.append(frame)
                    mixed_audio_buffer.append(frame.copy())
                except IOError as e:
                    logger.warning(f"錄音時發生IO錯誤：{e}")
                    continue

                # 收集滿一個 window_size 時做分離
                if len(chunk_buffer) >= window_frames:
                    segment_index += 1
                    audio_data = np.concatenate(chunk_buffer[:window_frames])
                    
                    audio_tensor = self.process_audio(audio_data)
                    if audio_tensor is not None:
                        logger.info(f"處理片段 {segment_index}")
                        future = self.executor.submit(
                            self.separate_and_accumulate,
                            audio_tensor,
                            overlap_samples,
                            segment_index
                        )
                        self.futures.append(future)

                    # 保留重疊區塊 (注意: 這裡是 PyAudio 的 frame buffer)
                    # 若你想在 time domain 上更精準，則需把剩餘的samples換算成 frames
                    slide_frames = window_frames - int(overlap_samples // CHUNK)
                    chunk_buffer = chunk_buffer[slide_frames:]

                    # 回收已完成的 future
                    self.futures = [f for f in self.futures if not f.done()]

        except Exception as e:
            logger.error(f"錄音過程中發生錯誤：{e}")
        finally:
            # 結束錄音
            stream.stop_stream()
            stream.close()
            p.terminate()

            # 等待所有分離任務完成
            for future in self.futures:
                try:
                    future.result(timeout=15.0)
                except Exception as e:
                    logger.error(f"處理任務發生錯誤：{e}")

            self.executor.shutdown(wait=True)

            # 若有錄到混合音頻，先儲存完整混合音
            self.save_mixed_audio(mixed_audio_buffer)

            self.speaker_buffers = postprocess_and_silence_empty_tracks(
                self.speaker_buffers,
                silence_threshold=0.005
            )
            
            # 最後儲存分離後 (累積的) 結果
            self.save_all_speakers()

            logger.info("錄音結束，資源已清理")

    def separate_and_accumulate(self, audio_tensor: torch.Tensor, overlap_samples: int, segment_index: int):
        """
        執行分離並將結果累加到 speaker_buffers，利用 cross-fade 與上一段做拼接。
        audio_tensor shape: [1, time]
        """
        try:
            with torch.no_grad():
                # 分離 (離線 Sepformer)
                separated = self.model.separate_batch(audio_tensor)  # shape: [batch=1, time, num_speakers]
                separated = separated.cpu()

            num_speakers = separated.shape[2]
            # 假設 batch=1 => separated[0, :, i] 為第 i 個說話者的 waveform

            # 對每個說話者做 cross-fade 累加
            for i in range(num_speakers):
                speaker_idx = i + 1  # speaker1, speaker2, ...
                wave = separated[0, :, i]  # shape=[time]
                
                # 簡易音量正規化，避免爆音
                max_val = torch.max(torch.abs(wave))
                if max_val > 1e-8:
                    wave = wave / max_val * 0.9

                if speaker_idx not in self.speaker_buffers:
                    self.speaker_buffers[speaker_idx] = None

                # cross-fade 與上一次累計的音訊做拼接
                self.speaker_buffers[speaker_idx] = cross_fade_concat(
                    self.speaker_buffers[speaker_idx],
                    wave,
                    overlap_samples
                )

            logger.info(f"片段 {segment_index} 分離累加完成")

        except Exception as e:
            logger.error(f"處理片段 {segment_index} 時發生錯誤：{e}")

    def save_all_speakers(self):
        """最終將每個說話者的累計音訊輸出成 wav 檔"""
        if not self.speaker_buffers:
            logger.warning("沒有任何說話者分離結果可儲存")
            return

        timestamp = datetime.now().strftime('%Y%m%d-%H_%M_%S')
        for spk_idx, wave_tensor in self.speaker_buffers.items():
            if wave_tensor is None or len(wave_tensor) == 0:
                continue

            # 轉成 [1, time] 的 Tensor
            wave_tensor = wave_tensor.unsqueeze(0)  # [1, time]

            out_path = os.path.join(OUTPUT_DIR, f"final_speaker{spk_idx}_{timestamp}.wav")
            try:
                torchaudio.save(out_path, wave_tensor, TARGET_RATE)
                logger.info(f"已儲存 Speaker {spk_idx} 音訊: {out_path}")
            except Exception as e:
                logger.error(f"儲存 Speaker {spk_idx} 音訊時發生錯誤：{e}")

    def save_mixed_audio(self, mixed_audio_buffer):
        """儲存整段混合音（原始錄音）"""
        if not mixed_audio_buffer:
            return
        try:
            mixed_audio = np.concatenate(mixed_audio_buffer)
            mixed_audio = mixed_audio.reshape(-1, CHANNELS)

            timestamp = datetime.now().strftime('%Y%m%d-%H_%M_%S')
            mixed_output_file = os.path.join(
                OUTPUT_DIR,
                f"mixed_audio_{timestamp}.wav"
            )

            mixed_tensor = torch.from_numpy(mixed_audio).T.float()  # shape=[1, time]
            torchaudio.save(
                mixed_output_file,
                mixed_tensor,
                RATE
            )
            logger.info(f"已儲存原始混合音訊：{mixed_output_file}")
        except Exception as e:
            logger.error(f"儲存混合音訊時發生錯誤：{e}")

    def stop_recording(self):
        """停止錄音"""
        self.is_recording = False
        logger.info("準備停止錄音...")


def main():
    separator_instance = AudioSeparator()
    try:
        separator_instance.record_and_process()
    except KeyboardInterrupt:
        logger.info("\n接收到停止信號，停止錄音。")
        separator_instance.stop_recording()
    except Exception as e:
        logger.error(f"程式執行時發生錯誤：{e}")
        separator_instance.stop_recording()


if __name__ == "__main__":
    main()

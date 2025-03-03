import os
import pyaudio
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from speechbrain.inference import SepformerSeparation as separator
import torch
import torchaudio
import logging

# 基本錄音參數
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
TARGET_RATE = 8000

# ★增大窗口與重疊比例：
WINDOW_SIZE = 6      # 6 秒
OVERLAP = 1.0        # 1 秒重疊
DEVICE_INDEX = None

# 音訊處理參數
MIN_ENERGY_THRESHOLD = 0.005  # 能量閾值，可依實際情況調整

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioSeparator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用設備: {self.device}")

        # 載入 Sepformer 模型 (離線分離)
        self.model = separator.from_hparams(
            source="speechbrain/sepformer-libri3mix",
            savedir='pretrained_models/sepformer-libri3mix',
            run_opts={"device": self.device}
        )

        # 重新取樣器 (44100 -> 8000)
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=RATE,
            new_freq=TARGET_RATE
        ).to(self.device)

        self.executor = ThreadPoolExecutor(max_workers=2)
        self.futures = []
        self.is_recording = False

        logger.info("AudioSeparator 初始化完成")

    def process_audio(self, audio_data):
        """處理音訊格式"""
        try:
            # 轉 float32
            if FORMAT == pyaudio.paInt16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data.astype(np.float32)

            # 能量檢測：過低則略過
            energy = np.mean(np.abs(audio_float))
            if energy < MIN_ENERGY_THRESHOLD:
                logger.debug(f"音訊能量 ({energy}) 低於閾值 ({MIN_ENERGY_THRESHOLD})")
                return None

            # 重塑為 [樣本數, channels]
            if len(audio_float.shape) == 1:
                audio_float = audio_float.reshape(-1, CHANNELS)

            # 製作 PyTorch tensor
            audio_tensor = torch.from_numpy(audio_float).T.float()

            # 如果是雙聲道而模型只支援單聲道則做平均
            if audio_tensor.shape[0] == 2:
                audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)

            # 移至 GPU 並重新取樣至 8kHz
            audio_tensor = audio_tensor.to(self.device)
            resampled = self.resampler(audio_tensor)

            # 確保形狀正確 [1, time]
            if len(resampled.shape) == 1:
                resampled = resampled.unsqueeze(0)

            return resampled

        except Exception as e:
            logger.error(f"音訊處理錯誤：{e}")
            return None

    def record_and_process(self, output_dir):
        """錄音並處理 (分段式)"""
        mixed_audio_buffer = []
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

            # 依照 window + overlap 準備好要收集的 buffer
            samples_per_window = int(WINDOW_SIZE * RATE)
            window_frames = int(samples_per_window / CHUNK)

            # 計算重疊的 frames
            overlap_frames = int((OVERLAP * RATE) / CHUNK)
            slide_frames = window_frames - overlap_frames

            buffer = []
            segment_index = 0
            self.is_recording = True

            while self.is_recording:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frame = np.frombuffer(data, dtype=np.float32 if FORMAT == pyaudio.paFloat32 else np.int16)
                    buffer.append(frame)
                    mixed_audio_buffer.append(frame.copy())
                except IOError as e:
                    logger.warning(f"錄音時發生IO錯誤：{e}")
                    continue

                # 一旦 buffer 足夠達到一個 window，就進行分離
                if len(buffer) >= window_frames:
                    segment_index += 1
                    audio_data = np.concatenate(buffer[:window_frames])
                    audio_tensor = self.process_audio(audio_data)

                    if audio_tensor is not None:
                        logger.info(f"處理片段 {segment_index}")
                        future = self.executor.submit(
                            self.separate_and_save,
                            audio_tensor,
                            output_dir,
                            segment_index
                        )
                        self.futures.append(future)

                    # 保留重疊區塊
                    buffer = buffer[slide_frames:]
                    self.futures = [f for f in self.futures if not f.done()]

        except Exception as e:
            logger.error(f"錄音過程中發生錯誤：{e}")
        finally:
            # 關閉錄音串流
            stream.stop_stream()
            stream.close()
            p.terminate()

            # 等待所有分離任務結束
            for future in self.futures:
                try:
                    future.result(timeout=10.0)
                except Exception as e:
                    logger.error(f"處理任務發生錯誤：{e}")

            self.executor.shutdown(wait=True)

            # 儲存原始混合音訊
            if mixed_audio_buffer:
                try:
                    mixed_audio = np.concatenate(mixed_audio_buffer)
                    mixed_audio = mixed_audio.reshape(-1, CHANNELS)

                    timestamp = datetime.now().strftime('%Y%m%d-%H_%M_%S')
                    mixed_output_file = os.path.join(
                        "16K-model/Audios-16K",
                        f"mixed_audio_{timestamp}.wav"
                    )

                    mixed_tensor = torch.from_numpy(mixed_audio).T.float()
                    torchaudio.save(
                        mixed_output_file,
                        mixed_tensor,
                        RATE  # 44100Hz 儲存
                    )
                    logger.info(f"已儲存原始混合音訊：{mixed_output_file}")
                except Exception as e:
                    logger.error(f"儲存混合音訊時發生錯誤：{e}")

            logger.info("錄音結束，資源已清理")

    def separate_and_save(self, audio_tensor, output_dir, segment_index):
        """分離並儲存音訊 (不再進行頻譜閘控或任何後處理)"""
        try:
            with torch.no_grad():
                separated = self.model.separate_batch(audio_tensor)

                # 這裡直接使用原始分離結果，不做降噪或其他處理
                # （可在此處實作簡單正規化，避免爆音）
                for i in range(separated.shape[2]):
                    speaker_audio = separated[:, :, i].cpu()

                    # 音量正規化
                    max_val = torch.max(torch.abs(speaker_audio))
                    if max_val > 1e-8:
                        speaker_audio = speaker_audio / max_val * 0.9

                    final_audio = speaker_audio[0].numpy()
                    final_tensor = torch.from_numpy(final_audio).unsqueeze(0)

                    timestamp = datetime.now().strftime('%Y%m%d-%H_%M_%S')
                    output_file = os.path.join(
                        output_dir,
                        f"speaker{i+1}_{timestamp}_{segment_index}.wav"
                    )

                    torchaudio.save(
                        output_file,
                        final_tensor,
                        TARGET_RATE
                    )

            logger.info(f"片段 {segment_index} 處理完成")

        except Exception as e:
            logger.error(f"處理片段 {segment_index} 時發生錯誤：{e}")

    def stop_recording(self):
        """停止錄音"""
        self.is_recording = False
        logger.info("準備停止錄音...")


def main():
    """主程式"""
    output_dir = "16K-model/Audios-16K"
    os.makedirs(output_dir, exist_ok=True)

    separator_instance = AudioSeparator()
    try:
        separator_instance.record_and_process(output_dir)
    except KeyboardInterrupt:
        logger.info("\n接收到停止信號")
        separator_instance.stop_recording()
    except Exception as e:
        logger.error(f"程式執行時發生錯誤：{e}")


if __name__ == "__main__":
    main()

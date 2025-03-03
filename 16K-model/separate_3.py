import os
import pyaudio
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from speechbrain.inference import SepformerSeparation as separator
import torch
import torchaudio
import logging
import noisereduce as nr


# 基本錄音參數
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
TARGET_RATE = 8000
WINDOW_SIZE = 6
OVERLAP = 0.5
DEVICE_INDEX = None

# 音訊處理參數
MIN_ENERGY_THRESHOLD = 0.005  # 能量閾值
NOISE_REDUCE_STRENGTH = 0.1  # 降噪強度

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
        
        self.model = separator.from_hparams(
            source="speechbrain/sepformer-libri3mix",
            savedir='pretrained_models/sepformer-libri3mix',
            run_opts={"device": self.device}
        )
        
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=RATE,
            new_freq=8000
        ).to(self.device)
        
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.futures = []
        self.is_recording = False
        logger.info("AudioSeparator 初始化完成")

    def spectral_gating(self, audio):
        """應用頻譜閘控降噪"""
        # 使用音訊開始部分作為噪音樣本，並確保噪音樣本不為空
        noise_sample = audio[:max(int(TARGET_RATE * 0.1), 1)]
        return nr.reduce_noise(
            y=audio,
            y_noise=noise_sample,
            sr=TARGET_RATE,
            prop_decrease=NOISE_REDUCE_STRENGTH,
            n_jobs=-1
        )

    def enhance_separation(self, separated_signals):
        """增強分離效果，僅應用一次降噪以避免過度處理"""
        enhanced_signals = torch.zeros_like(separated_signals) # 建立一個與分離結果同形狀的零張量
        
        for i in range(separated_signals.shape[2]):
            # 取得當前說話者的信號
            current_signal = separated_signals[0, :, i].cpu().numpy()
            
            # 應用頻譜閘控降噪
            denoised_signal = self.spectral_gating(current_signal)
            
            # 轉換回 tensor 並存儲（長度不足時取最小值）
            length = min(len(denoised_signal), separated_signals.shape[1])
            enhanced_signals[0, :length, i] = torch.from_numpy(denoised_signal).to(self.device)
        
        return enhanced_signals

    def process_audio(self, audio_data):
        """處理音訊格式"""
        try:
            # 轉換為 float32
            if FORMAT == pyaudio.paInt16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data.astype(np.float32)
            
            # 能量檢測：過低則略過
            energy = np.mean(np.abs(audio_float))
            if energy < MIN_ENERGY_THRESHOLD:
                logger.debug(f"音訊能量 ({energy}) 低於閾值 ({MIN_ENERGY_THRESHOLD})")
                return None
            
            # 重塑為正確形狀
            if len(audio_float.shape) == 1:
                audio_float = audio_float.reshape(-1, CHANNELS)

            # 直接使用原始音訊，不做前處理降噪
            # 調整形狀以符合模型輸入：[channels, time]
            audio_tensor = torch.from_numpy(audio_float).T.float()

            # 如果是雙聲道而模型只支援單聲道則取平均
            if audio_tensor.shape[0] == 2:
                audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)

            # *這裡可以考慮不進行重新採樣而是直接將44.1kHz的原始音訊直接丟入模型處理*
            # 移至 GPU 並重新取樣至 8kHz
            audio_tensor = audio_tensor.to(self.device)
            resampled = self.resampler(audio_tensor)
            
            # 確保形狀正確
            if len(resampled.shape) == 1:
                resampled = resampled.unsqueeze(0)
            
            return resampled
            
        except Exception as e:
            logger.error(f"音訊處理錯誤：{e}")
            return None

    def record_and_process(self, output_dir):
        """錄音並處理"""
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
            
            # 計算緩衝區大小與重疊數據
            samples_per_window = int(WINDOW_SIZE * RATE)
            window_frames = int(samples_per_window / CHUNK)
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
                    
                    # 保留重疊部分
                    buffer = buffer[slide_frames:]
                    self.futures = [f for f in self.futures if not f.done()]
                    
        except Exception as e:
            logger.error(f"錄音過程中發生錯誤：{e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            for future in self.futures:
                try:
                    future.result(timeout=10.0)
                except Exception as e:
                    logger.error(f"處理任務發生錯誤：{e}")
            
            self.executor.shutdown(wait=True)
            
            # 儲存原始混合音訊為單獨檔案
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
                        RATE  # 使用原始採樣率 44100Hz 儲存
                    )
                    logger.info(f"已儲存原始混合音訊：{mixed_output_file}")
                except Exception as e:
                    logger.error(f"儲存混合音訊時發生錯誤：{e}")
            
            logger.info("錄音結束，資源已清理")

    def separate_and_save(self, audio_tensor, output_dir, segment_index):
        """分離並儲存音訊"""
        try:
            with torch.no_grad():
                separated = self.model.separate_batch(audio_tensor)
                enhanced_separated = self.enhance_separation(separated) # *可以試試看從頭到尾都不進行降噪*
                
                timestamp = datetime.now().strftime('%Y%m%d-%H_%M_%S')
                for i in range(enhanced_separated.shape[2]):
                    speaker_audio = enhanced_separated[:, :, i].cpu()
                    
                    # 正規化音量
                    max_val = torch.max(torch.abs(speaker_audio))
                    if max_val > 0:
                        speaker_audio = speaker_audio / max_val * 0.9
                    
                    # 直接使用增強後的信號，不再進行重複降噪處理
                    final_audio = speaker_audio[0].numpy()
                    final_tensor = torch.from_numpy(final_audio).unsqueeze(0)
                    
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

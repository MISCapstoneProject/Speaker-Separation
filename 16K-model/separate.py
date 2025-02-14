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
CHANNELS = 2
RATE = 44100
TARGET_RATE = 16000
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
            source="speechbrain/sepformer-whamr16k",
            savedir='pretrained_models/sepformer-whamr16k',
            run_opts={"device": self.device}
        )
        
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=RATE,
            new_freq=TARGET_RATE
        ).to(self.device)
        
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.futures = []
        self.is_recording = False
        logger.info("AudioSeparator 初始化完成")

    def spectral_gating(self, audio):
        """應用頻譜閘控降噪"""
        # 使用音訊開始部分作為噪音樣本
        noise_sample = audio[:int(TARGET_RATE * 0.1)]  # 使用前0.1秒作為噪音樣本
        return nr.reduce_noise(
            y=audio,
            y_noise=noise_sample,
            sr=TARGET_RATE,
            prop_decrease=NOISE_REDUCE_STRENGTH,
            n_jobs=-1
        )

    def enhance_separation(self, separated_signals):
        """增強分離效果"""
        enhanced_signals = torch.zeros_like(separated_signals)
        
        for i in range(separated_signals.shape[2]):
            # 取得當前說話者的信號
            current_signal = separated_signals[0, :, i].cpu().numpy()
            
            # 應用頻譜閘控降噪
            denoised_signal = self.spectral_gating(current_signal)
            
            # 轉換回tensor並存儲
            enhanced_signals[0, :len(denoised_signal), i] = torch.from_numpy(denoised_signal).to(self.device)
        
        return enhanced_signals

    def process_audio(self, audio_data):
        try:
            # 轉換為float32
            if FORMAT == pyaudio.paInt16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data.astype(np.float32)
            
            # 改進能量檢測邏輯
            energy = np.mean(np.abs(audio_float))
            if energy < MIN_ENERGY_THRESHOLD:
                logger.debug(f"音訊能量 ({energy}) 低於閾值 ({MIN_ENERGY_THRESHOLD})")
                return None
            
            # 重塑為正確形狀
            if len(audio_float.shape) == 1:
                audio_float = audio_float.reshape(-1, CHANNELS)

            # ---【改動1：移除/註解前處理 spectral_gating】---
            # denoised_audio = np.stack([
            #     self.spectral_gating(audio_float[:, ch]) 
            #     for ch in range(CHANNELS)
            # ], axis=1)

            # 改為直接使用原始音訊（不做前處理降噪）
            # 這裡 audio_float shape = [samples, channels]
            # torch expects shape = [channels, time]
            audio_tensor = torch.from_numpy(audio_float).T.float()

            # 如果是雙聲道，且模型只支援單聲道，可取平均
            if audio_tensor.shape[0] == 2:
                audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)

            # 移至GPU並重新取樣
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
        # 創建混合音訊的緩衝區
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
            
            # 計算緩衝區大小
            samples_per_window = int(WINDOW_SIZE * RATE)
            window_frames = int(samples_per_window / CHUNK)
            overlap_frames = int((OVERLAP * RATE) / CHUNK)
            slide_frames = window_frames - overlap_frames
            
            buffer = []
            segment_index = 0  # 從0開始計數
            self.is_recording = True
            
            while self.is_recording:
                # 讀取音訊資料
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frame = np.frombuffer(data, dtype=np.float32 if FORMAT == pyaudio.paFloat32 else np.int16)
                    buffer.append(frame)
                    # 同時儲存到混合音訊緩衝區
                    mixed_audio_buffer.append(frame.copy())
                except IOError as e:
                    logger.warning(f"錄音時發生IO錯誤：{e}")
                    continue
                
                # 當緩衝區達到指定大小時處理音訊
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
                    
                    # 移動緩衝區
                    buffer = buffer[slide_frames:]
                    
                    # 清理已完成的任務
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
            
            # 儲存原始混合音訊
            if mixed_audio_buffer:
                try:
                    # 合併所有音訊片段
                    mixed_audio = np.concatenate(mixed_audio_buffer)
                    mixed_audio = mixed_audio.reshape(-1, CHANNELS)  # 確保形狀正確 (samples, channels)
                    
                    # 生成輸出檔案名稱
                    timestamp = datetime.now().strftime('%Y%m%d-%H_%M_%S')
                    mixed_output_file = os.path.join(
                        "Audios",
                        f"mixed_audio_{timestamp}.wav"
                    )
                    
                    # 轉換為tensor並確保形狀正確 (channels, samples)
                    mixed_tensor = torch.from_numpy(mixed_audio).T.float()
                    
                    # 儲存原始音訊
                    torchaudio.save(
                        mixed_output_file,
                        mixed_tensor,
                        RATE  # 使用原始採樣率 44100Hz
                    )
                    logger.info(f"已儲存原始混合音訊：{mixed_output_file}")
                except Exception as e:
                    logger.error(f"儲存混合音訊時發生錯誤：{e}")
            
            logger.info("錄音結束，資源已清理")

    def separate_and_save(self, audio_tensor, output_dir, segment_index):
        """分離並儲存音訊"""
        try:
            # 分離音訊
            with torch.no_grad():
                # 執行初始分離
                separated = self.model.separate_batch(audio_tensor)
                
                # 增強分離效果
                enhanced_separated = self.enhance_separation(separated)
                
                # 儲存結果
                timestamp = datetime.now().strftime('%Y%m%d-%H_%M_%S')
                for i in range(enhanced_separated.shape[2]):
                    # 提取當前說話者的音訊
                    speaker_audio = enhanced_separated[:, :, i].cpu()
                    
                    # 正規化音量
                    max_val = torch.max(torch.abs(speaker_audio))
                    if max_val > 0:
                        speaker_audio = speaker_audio / max_val * 0.9
                    
                    # 轉換為numpy進行最終處理
                    audio_np = speaker_audio.numpy()
                    
                    # 再次應用降噪
                    final_audio = self.spectral_gating(audio_np[0])
                    
                    # 轉回tensor並保存
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
    output_dir = "Audios"
    os.makedirs(output_dir, exist_ok=True)
    
    separator = AudioSeparator()
    try:
        separator.record_and_process(output_dir)
    except KeyboardInterrupt:
        logger.info("\n接收到停止信號")
        separator.stop_recording()
    except Exception as e:
        logger.error(f"程式執行時發生錯誤：{e}")

if __name__ == "__main__":
    main()
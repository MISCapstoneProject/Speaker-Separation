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

"""
record_separate_sync_sw.py
執行程式，輸入 python Record\record_separate_sync_sw.py

程式工作流程：

1. 持續錄音並將音訊存入緩衝區
2. 當緩衝區累積足夠的資料時(5秒):
    ‧進行初步的降噪處理
    ‧使用語音分離模型將不同說話者的聲音分開
    ‧對分離後的每個音訊再次進行降噪
    ‧儲存處理後的音訊檔案
3. 使用多執行緒處理音訊，確保錄音不會被中斷

"""

# 基本錄音參數
CHUNK = 1024    # 每次讀取的音訊區塊大小
FORMAT = pyaudio.paFloat32   # 音訊格式為32位元浮點數
CHANNELS = 2        # 雙聲道
RATE = 44100        # 原始採樣率
TARGET_RATE = 8000  # 降採樣後的採樣率
WINDOW_SIZE = 5     # 處理窗口大小(秒)
OVERLAP = 0.5       # 窗口重疊比例(秒)
DEVICE_INDEX = None

# 音訊處理參數
MIN_ENERGY_THRESHOLD = 0.005  # 能量閾值
NOISE_REDUCE_STRENGTH = 0.15  # 降噪強度

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioSeparator:
    def __init__(self):
        # 檢查是否可以使用GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用設備: {self.device}")
        
        # 載入語音分離模型 
        self.model = separator.from_hparams(
            source="speechbrain/sepformer-wsj02mix",
            savedir="pretrained_models/sepformer-wsj02mix",
            run_opts={"device": self.device}
        )
        
        # 初始化重採樣器，將採樣率從44100Hz降到8000Hz
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=RATE,
            new_freq=TARGET_RATE
        ).to(self.device)
        
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.futures = []   # 用於追蹤提交的任務
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
        """處理音訊資料"""
        try:
            # 轉換為float32
            if FORMAT == pyaudio.paInt16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data.astype(np.float32)
            
            # 能量檢測
            energy = np.mean(np.abs(audio_float))
            if energy < MIN_ENERGY_THRESHOLD:
                logger.debug(f"音訊能量 ({energy}) 低於閾值 ({MIN_ENERGY_THRESHOLD})")
                return None
            
            # 重塑為正確的形狀
            if len(audio_float.shape) == 1:
                audio_float = audio_float.reshape(-1, CHANNELS)
            
            # 應用頻譜閘控降噪進行預處理
            denoised_audio = np.stack([self.spectral_gating(audio_float[:, ch]) for ch in range(CHANNELS)], axis=1)
            
            # 轉換為PyTorch tensor並確保形狀正確
            audio_tensor = torch.from_numpy(denoised_audio).T.float()
            if audio_tensor.shape[0] == 2:
                audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
            
            # 移至GPU並重新取樣（如果沒有GPU就使用CPU）
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
            
            logger.info("開始錄音, CTRL+C 停止錄音")
            
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
                    
                    # 清理已完成的任務，保留未完成的工作
                    self.futures = [f for f in self.futures if not f.done()]
                    
        except Exception as e:
            logger.error(f"錄音過程中發生錯誤：{e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # 在結束時等待所有任務完成
            for future in self.futures:
                try:
                    future.result(timeout=10.0)  # 設定超時以避免永久等待
                except Exception as e:
                    logger.error(f"處理任務發生錯誤：{e}")
            
            self.executor.shutdown(wait=True)
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
    output_dir = "Audios/output_separatedAudio"
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

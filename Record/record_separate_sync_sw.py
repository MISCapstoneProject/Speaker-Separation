import os
import pyaudio
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from speechbrain.inference import SepformerSeparation as separator
import torch
import torchaudio
import logging
from scipy import signal

# 基本錄音參數
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 2
RATE = 48000
TARGET_RATE = 8000  # 降低採樣率以改善分離效果
WINDOW_SIZE = 5  # 縮短窗口大小以提高分離效果
OVERLAP = 0.5  # 減少重疊以避免邊界效應
DEVICE_INDEX = None

# 音訊處理參數
MIN_ENERGY_THRESHOLD = 0.01  # 最小能量閾值，用於靜音檢測

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioSeparator:
    def __init__(self):
        # 初始化模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用設備: {self.device}")
        
        # 使用較小的頻寬配置載入模型
        self.model = separator.from_hparams(
            source="speechbrain/sepformer-wsj02mix",
            savedir="pretrained_models/sepformer-wsj02mix",
            run_opts={
                "device": self.device,
            }
        )
        
        # 初始化重採樣器
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=RATE,
            new_freq=TARGET_RATE
        ).to(self.device)
        
        # 初始化執行緒池
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.futures = []
        self.is_recording = False
        logger.info("AudioSeparator 初始化完成")

    def enhance_waveform(self, waveform):
        """增強波形品質"""
        # 對信號進行標準化
        mean = torch.mean(waveform)
        std = torch.std(waveform)
        waveform = (waveform - mean) / (std + 1e-8)
        
        # 應用 pre-emphasis
        waveform = torch.cat([waveform[:, 0:1], waveform[:, 1:] - 0.97 * waveform[:, :-1]], dim=1)
        
        return waveform

    def process_audio(self, audio_data):
        """處理音訊資料"""
        try:
            # 轉換為float32
            if FORMAT == pyaudio.paInt16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data.astype(np.float32)
                
            # 檢查音訊能量
            energy = np.mean(np.abs(audio_float))
            if energy < MIN_ENERGY_THRESHOLD:
                return None
            
            # 重塑為正確的形狀
            if len(audio_float.shape) == 1:
                audio_float = audio_float.reshape(-1, CHANNELS)
            
            # 轉換為PyTorch tensor
            audio_tensor = torch.from_numpy(audio_float).T.float()
            
            # 轉換為單聲道
            if audio_tensor.shape[0] == 2:
                audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
            
            # 移至GPU並重新取樣
            audio_tensor = audio_tensor.to(self.device)
            resampled = self.resampler(audio_tensor)
            
            # 增強波形
            enhanced = self.enhance_waveform(resampled)
            
            # 確保形狀正確
            if len(enhanced.shape) == 1:
                enhanced = enhanced.unsqueeze(0)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"音訊處理錯誤：{e}")
            raise

    def improve_separation(self, separated_signals):
        """改進分離效果"""
        # 標準化每個分離的信號
        for i in range(separated_signals.shape[2]):
            signal = separated_signals[:, :, i]
            mean = torch.mean(signal)
            std = torch.std(signal)
            separated_signals[:, :, i] = (signal - mean) / (std + 1e-8)
        
        # 應用 Wiener 濾波進行增強
        for i in range(separated_signals.shape[2]):
            signal = separated_signals[:, :, i]
            # 計算信號功率譜
            spec = torch.stft(signal.squeeze(0), 
                            n_fft=512, 
                            hop_length=256, 
                            window=torch.hann_window(512).to(self.device),
                            return_complex=True)
            mag = torch.abs(spec)
            phase = torch.angle(spec)
            
            # 應用 Wiener 濾波
            power = mag ** 2
            mask = power / (power + 0.01)
            enhanced_mag = mag * mask
            
            # 重建信號
            enhanced_spec = enhanced_mag * torch.exp(1j * phase)
            enhanced_signal = torch.istft(enhanced_spec, 
                                        n_fft=512, 
                                        hop_length=256, 
                                        window=torch.hann_window(512).to(self.device))
            separated_signals[:, :enhanced_signal.shape[0], i] = enhanced_signal.unsqueeze(0)
        
        return separated_signals

    def separate_and_save(self, audio_tensor, output_dir, segment_index):
        """分離並儲存音訊"""
        try:
            logger.info(f"處理片段 {segment_index}")
            
            if audio_tensor is None:
                logger.info(f"片段 {segment_index} 能量太低，跳過處理")
                return
            
            # 分離音訊
            with torch.no_grad():
                separated = self.model.separate_batch(audio_tensor)
                # 改進分離效果
                separated = self.improve_separation(separated)
            
            # 儲存結果
            timestamp = datetime.now().strftime('%Y%m%d-%H_%M_%S')
            for i in range(separated.shape[2]):
                output_file = os.path.join(
                    output_dir,
                    f"speaker{i+1}_{timestamp}_{segment_index}.wav"
                )
                
                # 正規化音量
                audio_data = separated[:, :, i].cpu()
                max_val = torch.max(torch.abs(audio_data))
                if max_val > 0:
                    audio_data = audio_data / max_val * 0.9
                
                torchaudio.save(
                    output_file,
                    audio_data,
                    TARGET_RATE
                )
                
            logger.info(f"片段 {segment_index} 處理完成")
            
        except Exception as e:
            logger.error(f"處理片段 {segment_index} 時發生錯誤：{e}")

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
            
            logger.info("開始錄音")
            
            # 計算每個窗口需要的幀數
            samples_per_window = int(WINDOW_SIZE * RATE)
            window_frames = int(samples_per_window / CHUNK)
            overlap_frames = int((OVERLAP * RATE) / CHUNK)
            slide_frames = window_frames - overlap_frames
            
            buffer = []
            segment_index = 1
            self.is_recording = True
            
            while self.is_recording:
                # 讀取音訊資料
                for _ in range(slide_frames):
                    if not self.is_recording:
                        break
                    try:
                        data = stream.read(CHUNK, exception_on_overflow=False)
                        if FORMAT == pyaudio.paFloat32:
                            frame = np.frombuffer(data, dtype=np.float32)
                        else:
                            frame = np.frombuffer(data, dtype=np.int16)
                        buffer.append(frame)
                    except IOError as e:
                        logger.warning(f"錄音時發生IO錯誤：{e}")
                        continue
                
                # 當緩衝區足夠大時處理音訊
                if len(buffer) >= window_frames:
                    try:
                        # 取出一個完整窗口的資料
                        audio_data = np.concatenate(buffer[:window_frames])
                        audio_tensor = self.process_audio(audio_data)
                        
                        if audio_tensor is not None:
                            # 提交分離任務
                            future = self.executor.submit(
                                self.separate_and_save,
                                audio_tensor,
                                output_dir,
                                segment_index
                            )
                            self.futures.append(future)
                        
                        # 清理已完成的任務和緩衝區
                        self.futures = [f for f in self.futures if not f.done()]
                        buffer = buffer[slide_frames:]
                        segment_index += 1
                        
                    except Exception as e:
                        logger.error(f"音訊處理錯誤：{e}")
                        buffer = buffer[slide_frames:]
                        continue
                    
        except Exception as e:
            logger.error(f"錄音過程中發生錯誤：{e}")
        finally:
            # 清理資源
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # 等待所有處理任務完成
            for future in self.futures:
                try:
                    future.result(timeout=10.0)
                except Exception as e:
                    logger.error(f"處理任務發生錯誤：{e}")
            
            self.executor.shutdown(wait=True)
            logger.info("錄音結束，資源已清理")

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

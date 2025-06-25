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
import threading
import time

# 基本錄音參數
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
TARGET_RATE = 16000
WINDOW_SIZE = 6
OVERLAP = 0.5
DEVICE_INDEX = None

# 音訊處理參數
MIN_ENERGY_THRESHOLD = 0.001  # 能量閾值
NOISE_REDUCE_STRENGTH = 0.1  # 降噪強度
MAX_BUFFER_MINUTES = 5  # 最大緩衝區時長（分鐘）
SNR_THRESHOLD = 10  # SNR 閾值

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioSeparator:
    def __init__(self, enable_noise_reduction=True, snr_threshold=SNR_THRESHOLD):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enable_noise_reduction = enable_noise_reduction
        self.snr_threshold = snr_threshold
        
        logger.info(f"使用設備: {self.device}")
        logger.info(f"降噪功能: {'啟用' if enable_noise_reduction else '停用'}")
        logger.info(f"SNR 閾值: {snr_threshold} dB")
        
        try:
            self.model = separator.from_hparams(
            source="speechbrain/sepformer-whamr16k",
            savedir='pretrained_models/sepformer-whamr16k',
                run_opts={"device": self.device}
            )
        except Exception as e:
            logger.error(f"模型載入失敗: {e}")
            raise
        
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=RATE,
            new_freq=TARGET_RATE
        ).to(self.device)
        
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.futures = []
        self.is_recording = False
        self.processing_stats = {
            'segments_processed': 0,
            'segments_skipped': 0,
            'errors': 0
        }
        
        # 緩衝區大小限制
        self.max_buffer_size = int(RATE * MAX_BUFFER_MINUTES * 60 / CHUNK)
        
        logger.info("AudioSeparator 初始化完成")

    def estimate_snr(self, signal):
        """估算信號雜訊比"""
        try:
            # 計算信號功率
            signal_power = np.mean(signal ** 2)
            
            # 估算雜訊功率（使用高頻部分作為雜訊估計）
            if len(signal) > 1000:
                noise_estimate = np.std(signal[-1000:]) ** 2
            else:
                noise_estimate = np.std(signal) ** 2 * 0.1
            
            # 避免除零
            noise_estimate = max(noise_estimate, 1e-10)
            
            snr = 10 * np.log10(signal_power / noise_estimate)
            return snr
        except Exception as e:
            logger.warning(f"SNR 估算失敗: {e}")
            return 0

    def spectral_gating(self, audio):
        """應用頻譜閘控降噪"""
        try:
            # 使用音訊開始部分作為噪音樣本
            noise_sample_length = max(int(TARGET_RATE * 0.1), 1)
            noise_sample = audio[:noise_sample_length]
            
            return nr.reduce_noise(
                y=audio,
                y_noise=noise_sample,
                sr=TARGET_RATE,
                prop_decrease=NOISE_REDUCE_STRENGTH,
                n_jobs=1  # 減少並行度以避免資源競爭
            )
        except Exception as e:
            logger.warning(f"降噪處理失敗: {e}")
            return audio

    def enhance_separation(self, separated_signals):
        """增強分離效果 - 智能降噪"""
        if not self.enable_noise_reduction:
            return separated_signals
            
        enhanced_signals = torch.zeros_like(separated_signals)
        
        for i in range(separated_signals.shape[2]):
            current_signal = separated_signals[0, :, i].cpu().numpy()
            
            # 檢查信號品質，只在必要時進行降噪
            signal_snr = self.estimate_snr(current_signal)
            
            if signal_snr < self.snr_threshold:
                logger.debug(f"說話者 {i+1} SNR: {signal_snr:.2f} dB，進行降噪")
                denoised_signal = self.spectral_gating(current_signal)
            else:
                logger.debug(f"說話者 {i+1} SNR: {signal_snr:.2f} dB，跳過降噪")
                denoised_signal = current_signal
            
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
                logger.debug(f"音訊能量 ({energy:.6f}) 低於閾值 ({MIN_ENERGY_THRESHOLD})")
                return None
            
            # 重塑為正確形狀
            if len(audio_float.shape) == 1:
                audio_float = audio_float.reshape(-1, CHANNELS)

            # 調整形狀以符合模型輸入：[channels, time]
            audio_tensor = torch.from_numpy(audio_float).T.float()

            # 如果是雙聲道而模型只支援單聲道則取平均
            if audio_tensor.shape[0] == 2:
                audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)

            # 移至 GPU 並重新取樣至 8kHz
            audio_tensor = audio_tensor.to(self.device)
            resampled = self.resampler(audio_tensor)
            
            # 確保形狀正確
            if len(resampled.shape) == 1:
                resampled = resampled.unsqueeze(0)
            
            return resampled
            
        except Exception as e:
            logger.error(f"音訊處理錯誤：{e}")
            self.processing_stats['errors'] += 1
            return None

    def cleanup_futures(self):
        """清理已完成的任務"""
        completed_futures = []
        for future in self.futures:
            if future.done():
                try:
                    future.result()  # 獲取結果以捕獲任何異常
                except Exception as e:
                    logger.error(f"處理任務發生錯誤：{e}")
                    self.processing_stats['errors'] += 1
                completed_futures.append(future)
        
        # 移除已完成的任務
        for future in completed_futures:
            self.futures.remove(future)

    def record_and_process(self, output_dir):
        """錄音並處理"""
        mixed_audio_buffer = []
        p = None
        stream = None
        
        try:
            p = pyaudio.PyAudio()
            
            # 檢查設備可用性
            if DEVICE_INDEX is not None:
                device_info = p.get_device_info_by_index(DEVICE_INDEX)
                logger.info(f"使用音訊設備: {device_info['name']}")
            
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
            last_stats_time = time.time()
            
            while self.is_recording:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frame = np.frombuffer(data, dtype=np.float32 if FORMAT == pyaudio.paFloat32 else np.int16)
                    
                    buffer.append(frame)
                    
                    # 限制 mixed_audio_buffer 大小以防止記憶體耗盡
                    mixed_audio_buffer.append(frame.copy())
                    if len(mixed_audio_buffer) > self.max_buffer_size:
                        mixed_audio_buffer.pop(0)
                        
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
                        self.processing_stats['segments_processed'] += 1
                    else:
                        self.processing_stats['segments_skipped'] += 1
                    
                    # 保留重疊部分
                    buffer = buffer[slide_frames:]
                    
                    # 定期清理已完成的任務
                    if segment_index % 10 == 0:
                        self.cleanup_futures()
                    
                    # 每30秒報告一次統計資訊
                    current_time = time.time()
                    if current_time - last_stats_time > 30:
                        self._log_statistics()
                        last_stats_time = current_time
                    
        except Exception as e:
            logger.error(f"錄音過程中發生錯誤：{e}")
        finally:
            self._cleanup_resources(p, stream, mixed_audio_buffer)

    def _cleanup_resources(self, p, stream, mixed_audio_buffer):
        """清理資源"""
        # 停止並關閉音訊流
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
                logger.info("音訊流已關閉")
            except Exception as e:
                logger.error(f"關閉音訊流時發生錯誤：{e}")
        
        if p is not None:
            try:
                p.terminate()
                logger.info("PyAudio 已終止")
            except Exception as e:
                logger.error(f"終止 PyAudio 時發生錯誤：{e}")
        
        # 等待所有處理任務完成
        logger.info("等待處理任務完成...")
        for future in self.futures:
            try:
                future.result(timeout=15.0)
            except Exception as e:
                logger.error(f"處理任務發生錯誤：{e}")
        
        self.executor.shutdown(wait=True)
        logger.info("線程池已關閉")
        
        # 儲存原始混合音訊
        self._save_mixed_audio(mixed_audio_buffer)
        
        # 記錄最終統計
        self._log_final_statistics()
        
        # 清理GPU記憶體
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("錄音結束，所有資源已清理")

    def _save_mixed_audio(self, mixed_audio_buffer):
        """儲存混合音訊"""
        if not mixed_audio_buffer:
            return
            
        try:
            mixed_audio = np.concatenate(mixed_audio_buffer)
            mixed_audio = mixed_audio.reshape(-1, CHANNELS)
            
            timestamp = datetime.now().strftime('%Y%m%d-%H_%M_%S')
            mixed_output_file = os.path.join(
                "Audios_Storage",
                f"mixed_audio_{timestamp}.wav"
            )
            
            mixed_tensor = torch.from_numpy(mixed_audio).T.float()
            torchaudio.save(
                mixed_output_file,
                mixed_tensor,
                RATE
            )
            logger.info(f"已儲存原始混合音訊：{mixed_output_file}")
            
        except Exception as e:
            logger.error(f"儲存混合音訊時發生錯誤：{e}")

    def _log_statistics(self):
        """記錄統計資訊"""
        stats = self.processing_stats
        logger.info(f"統計 - 已處理: {stats['segments_processed']}, "
                   f"已跳過: {stats['segments_skipped']}, "
                   f"錯誤: {stats['errors']}, "
                   f"進行中任務: {len(self.futures)}")

    def _log_final_statistics(self):
        """記錄最終統計資訊"""
        stats = self.processing_stats
        total = stats['segments_processed'] + stats['segments_skipped']
        if total > 0:
            success_rate = (stats['segments_processed'] / total) * 100
            logger.info(f"最終統計 - 總片段: {total}, "
                       f"成功處理: {stats['segments_processed']} ({success_rate:.1f}%), "
                       f"跳過: {stats['segments_skipped']}, "
                       f"錯誤: {stats['errors']}")

    def separate_and_save(self, audio_tensor, output_dir, segment_index):
        """分離並儲存音訊"""
        try:
            with torch.no_grad():
                separated = self.model.separate_batch(audio_tensor)
                enhanced_separated = self.enhance_separation(separated)
                
                # 立即釋放原始分離結果
                del separated
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                timestamp = datetime.now().strftime('%Y%m%d-%H_%M_%S')
                
                for i in range(enhanced_separated.shape[2]):
                    speaker_audio = enhanced_separated[:, :, i].cpu()
                    
                    # 正規化音量
                    max_val = torch.max(torch.abs(speaker_audio))
                    if max_val > 0:
                        speaker_audio = speaker_audio / max_val * 0.9
                    
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
                
                logger.debug(f"片段 {segment_index} 處理完成")
                
        except Exception as e:
            logger.error(f"處理片段 {segment_index} 時發生錯誤：{e}")
            self.processing_stats['errors'] += 1
        finally:
            # 確保 GPU 記憶體釋放
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def stop_recording(self):
        """停止錄音"""
        self.is_recording = False
        logger.info("準備停止錄音...")


def main():
    """主程式"""
    output_dir = "Audios_Storage"
    os.makedirs(output_dir, exist_ok=True)
    
    # 可配置參數
    enable_noise_reduction = True  # 是否啟用降噪
    snr_threshold = 10  # SNR 閾值
    
    separator_instance = AudioSeparator(
        enable_noise_reduction=enable_noise_reduction,
        snr_threshold=snr_threshold
    )
    
    try:
        separator_instance.record_and_process(output_dir)
    except KeyboardInterrupt:
        logger.info("\n接收到停止信號")
        separator_instance.stop_recording()
    except Exception as e:
        logger.error(f"程式執行時發生錯誤：{e}")


if __name__ == "__main__":
    main()
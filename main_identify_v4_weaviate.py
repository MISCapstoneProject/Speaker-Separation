import os
import re
import sys
import uuid
import numpy as np
import torch
import soundfile as sf
from scipy.spatial.distance import cosine
from scipy.signal import resample_poly
import warnings
import logging
from datetime import datetime, timezone
from typing import Tuple, List, Dict, Optional, Union, Any
import weaviate  # type: ignore
from weaviate.classes.query import MetadataQuery # type: ignore
import sync_npy_username  # 用來呼叫 ffmpeg 或檢查更新

# 設定 httpx 的日誌層級為 WARNING 或更高，以關閉 INFO 層級的 HTTP 請求日誌
logging.getLogger("httpx").setLevel(logging.WARNING)

# 新增時區處理函數
def format_rfc3339(dt: Optional[datetime] = None) -> str:
    """
    將日期時間格式化為符合 RFC3339 標準的字串，包含時區信息
    
    Args:
        dt: 要格式化的 datetime 對象，若為 None 則使用當前時間
        
    Returns:
        str: RFC3339 格式的日期時間字串
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    elif dt.tzinfo is None:
        # 若沒有時區信息，則假設為 UTC
        dt = dt.replace(tzinfo=timezone.utc)
    
    # 格式化為 RFC3339 格式: YYYY-MM-DDThh:mm:ss.sssZ
    return dt.isoformat().replace('+00:00', 'Z')

# 隱藏多餘的警告與日誌
warnings.filterwarnings("ignore")
logging.getLogger("speechbrain").setLevel(logging.ERROR)

# 自定義 Tee 類別：同時輸出到螢幕和 log 檔案
class Tee:
    def __init__(self, file_name: str, mode: str = "w") -> None:
        """
        初始化 Tee 實例，用於同時輸出到螢幕和檔案
        
        Args:
            file_name: 輸出日誌檔名
            mode: 檔案開啟模式，預設為 'w' (覆寫)
        """
        self.file = open(file_name, mode, encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, message: str) -> None:
        """
        寫入訊息到檔案和螢幕
        
        Args:
            message: 要輸出的訊息
        """
        self.file.write(message)
        self.stdout.write(message)

    def flush(self) -> None:
        """
        強制刷新檔案和螢幕輸出緩衝
        """
        self.file.flush()
        self.stdout.flush()

# 將標準輸出重定向到 log 檔
sys.stdout = Tee("output_log.txt")

# 載入 SpeechBrain 語音辨識模型
from speechbrain.inference import SpeakerRecognition

# 全域參數設定
THRESHOLD_LOW = 0.2     # 過於相似，不更新
THRESHOLD_UPDATE = 0.34 # 更新嵌入向量
THRESHOLD_NEW = 0.36    # 判定為新說話者
DEFAULT_SPEAKER_NAME = "未命名說話者"  # 預設的說話者名稱

# Weaviate 連接參數（如果需要可以修改）
WEAVIATE_HOST = "localhost"
WEAVIATE_PORT = "8080"


class AudioProcessor:
    """音訊處理類別，負責音訊處理和嵌入向量提取"""
    
    def __init__(self) -> None:
        """初始化 SpeechBrain 模型"""
        try:
            self.model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="models/speechbrain_recognition"
            )
            print("SpeechBrain 模型加載成功！")
        except ImportError as e:
            print(f"SpeechBrain 未正確安裝: {e}")
            print("請運行: pip install speechbrain")
            raise
        except Exception as e:
            print(f"載入 SpeechBrain 模型時發生錯誤: {e}")
            raise
    
    def resample_audio(self, signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        使用 scipy 進行高品質重新採樣
        
        Args:
            signal: 原始音訊信號
            orig_sr: 原始取樣率
            target_sr: 目標取樣率
            
        Returns:
            np.ndarray: 重新採樣後的音訊信號
        """
        return resample_poly(signal, target_sr, orig_sr)
    
    def extract_embedding(self, audio_path: str) -> np.ndarray:
        """
        提取音檔的嵌入向量，根據音檔取樣率智能處理
        
        Args:
            audio_path: 音檔路徑
            
        Returns:
            np.ndarray: 音檔的嵌入向量
            
        處理流程:
            1. 若音檔為 16kHz，則直接使用
            2. 若音檔為 8kHz，則直接升頻到 16kHz
            3. 若音檔取樣率高於 16kHz，則降頻到 16kHz
            4. 其他取樣率，則重新採樣到 16kHz
        """
        try:
            signal, sr = sf.read(audio_path)
            
            # 處理立體聲轉單聲道
            if signal.ndim > 1:
                signal = signal.mean(axis=1)
                
            # 測試
            # signal_8k = self.resample_audio(signal, 16000, 8000)
            # signal_16k = self.resample_audio(signal_8k, 8000, 16000)

            # 根據取樣率處理
            if sr == 16000:
                # 已是 16kHz，直接使用
                signal_16k = signal
            elif sr == 8000:
                # 若為 8kHz，直接升頻到 16kHz
                signal_16k = self.resample_audio(signal, 8000, 16000)
            elif sr > 16000:
                # 若高於 16kHz，直接降頻到 16kHz
                signal_16k = self.resample_audio(signal, sr, 16000)
            else:
                # 其他取樣率，重新採樣到 16kHz
                signal_16k = self.resample_audio(signal, sr, 16000)
            
            # 轉換為 PyTorch 張量
            signal_16k = torch.tensor(signal_16k, dtype=torch.float32).unsqueeze(0)
            
            # 限制音檔長度（最多 10 秒）
            max_length = 16000 * 10
            if signal_16k.shape[1] > max_length:
                signal_16k = signal_16k[:, :max_length]
                
            # 提取嵌入向量
            embedding = self.model.encode_batch(signal_16k).squeeze().numpy()
            return embedding
            
        except Exception as e:
            print(f"提取嵌入向量時發生錯誤: {e}")
            raise


class WeaviateRepository:
    """Weaviate 資料存取庫類別，負責與 Weaviate 資料庫的交互"""
    
    def __init__(self) -> None:
        """初始化 Weaviate 連接"""
        try:
            self.client = weaviate.connect_to_local()
            print("成功連接到 Weaviate 資料庫！")
            
            # 檢查必要的集合是否存在
            if not self.client.collections.exists("VoicePrint") or not self.client.collections.exists("Speaker"):
                print("警告：Weaviate 中缺少必要的集合 (VoicePrint / Speaker)!")
                print("請先運行 weaviate_study/create_collections.py 建立所需的集合")
                print("正在嘗試繼續執行，但可能會發生錯誤...")
        
        except Exception as e:
            print(f"無法連接到 Weaviate 資料庫：{e}")
            print("請確認：")
            print("1. Docker 服務是否正在運行")
            print("2. Weaviate 容器是否已經啟動")
            print("3. weaviate_study/docker-compose.yml 中的配置是否正確")
            print("使用命令 'docker-compose -f weaviate_study/docker-compose.yml up -d' 啟動 Weaviate")
            raise
    
    def compare_embedding(self, new_embedding: np.ndarray) -> Tuple[Optional[str], Optional[str], float, List[Tuple[str, str, float, int]]]:
        """
        比較新的嵌入向量與資料庫中所有現有嵌入向量的相似度
        
        Args:
            new_embedding: 新的嵌入向量
            
        Returns:
            tuple: (最佳匹配ID, 最佳匹配說話者名稱, 最小距離, 所有距離列表)
        """
        try:
            voice_print_collection = self.client.collections.get("VoicePrint")
            
            # 計算新向量與數據庫中所有向量的距離
            results = voice_print_collection.query.near_vector(
                near_vector=new_embedding.tolist(),
                limit=5,  # 測試! 返回前 5 個最相似的結果
                return_properties=["speaker_name", "update_count", "create_time", "updated_time"],
                return_metadata=MetadataQuery(distance=True)
            )
            
            # 如果沒有找到任何結果
            if not results.objects:
                print("資料庫中尚無任何嵌入向量")
                return None, None, float('inf'), []
            
            # 處理結果，計算距離
            distances = []
            for obj in results.objects:
                # 距離信息可能在不同位置，根據您的 Weaviate 版本進行適配
                distance = None
                if hasattr(obj, 'metadata') and hasattr(obj.metadata, 'distance'):
                    # v4 API
                    distance = obj.metadata.distance
                
                # 處理 distance 可能是 None 的情況
                if distance is None:
                    distance = -1  # 使用預設值
                    print(f"警告：無法從結果中獲取距離信息，使用預設值 {distance}")
                
                object_id = obj.uuid
                speaker_name = obj.properties.get("speaker_name")
                update_count = obj.properties.get("update_count")
                
                # 使用安全的格式化方法，避免 None 值導致錯誤
                distance_str = f"{distance:.4f}" if distance is not None else "未知"
                print(f"比對 - 說話者: {speaker_name}, "
                      f"更新次數: {update_count}, 餘弦距離: {distance_str}")
                
                # 保存距離資訊
                distances.append((object_id, speaker_name, distance, update_count))
            
            # 找出最小距離
            if distances:
                best_match = min(distances, key=lambda x: x[2])
                best_id, best_name, best_distance, _ = best_match
                return best_id, best_name, best_distance, distances
            else:
                # 如果沒有有效的距離信息，返回空結果
                print("警告：未能獲取有效的距離信息")
                return None, None, float('inf'), []
            
        except Exception as e:
            print(f"比對嵌入向量時發生錯誤: {e}")
            raise
    
    def update_embedding(self, voice_print_id: str, new_embedding: np.ndarray, update_count: int) -> int:
        """
        使用加權移動平均更新現有的嵌入向量
        
        Args:
            voice_print_id: 要更新的聲紋向量 ID
            new_embedding: 新的嵌入向量
            update_count: 當前更新次數
            
        Returns:
            int: 更新後的次數
        """
        try:
            # 獲取現有的嵌入向量
            voice_print_collection = self.client.collections.get("VoicePrint")
            existing_object = voice_print_collection.query.fetch_object_by_id(
                uuid=voice_print_id,
                include_vector=True
            )
            
            if not existing_object:
                raise ValueError(f"找不到 ID 為 {voice_print_id} 的聲紋向量")
            
            # 獲取現有的嵌入向量            
            vec_dict = existing_object.vector   # 取出 Weaviate 回傳的 named vector
            raw_old = vec_dict["default"] if isinstance(vec_dict, dict) else vec_dict   # 如果是 dict，就用 "default" 這組；否則直接當 list 處理
            old_embedding = np.array(raw_old, dtype=float)
            
            # 使用加權移動平均更新嵌入向量
            updated_embedding = (old_embedding * update_count + new_embedding) / (update_count + 1)
            new_update_count = update_count + 1
            
            # 更新數據庫中的向量
            voice_print_collection.data.update(
                uuid=voice_print_id,
                properties={
                    "updated_time": format_rfc3339(),
                    "update_count": new_update_count
                },
                vector=updated_embedding.tolist()
            )
            
            print(f"(更新) 聲紋ID {voice_print_id} 已更新，新的更新次數: {new_update_count}")
            return new_update_count
            
        except Exception as e:
            print(f"更新嵌入向量時發生錯誤: {e}")
            raise
    
    def add_embedding_without_averaging(self, speaker_name: str, new_embedding: np.ndarray, speaker_id: Optional[str] = None) -> str:
        """
        為現有說話者添加新的嵌入向量（不進行加權平均）
        
        Args:
            speaker_name: 說話者名稱
            new_embedding: 新的嵌入向量
            speaker_id: 現有說話者 ID，如果為 None 則創建新說話者
            
        Returns:
            str: 新建立的聲紋向量 ID
        """
        try:
            # 如果沒有提供 speaker_id，則創建新的說話者
            if not speaker_id:
                speaker_id = self.create_new_speaker(speaker_name)
            
            # 添加新的嵌入向量
            voice_print_collection = self.client.collections.get("VoicePrint")
            voice_print_id = str(uuid.uuid4())
            
            # 創建新的聲紋向量
            voice_print_collection.data.insert(
                properties={
                    "create_time": format_rfc3339(),
                    "updated_time": format_rfc3339(),
                    "update_count": 1,
                    "speaker_name": speaker_name
                },
                uuid=voice_print_id,
                vector=new_embedding.tolist(),
                references={
                    "speaker": [speaker_id]
                }
            )
            
            print(f"(新嵌入) 為說話者 {speaker_name} 添加了新的聲紋向量 (ID: {voice_print_id})")
            return voice_print_id
            
        except Exception as e:
            print(f"添加嵌入向量時發生錯誤: {e}")
            raise
    
    def create_new_speaker(self, speaker_name: str = DEFAULT_SPEAKER_NAME) -> str:
        """
        創建新的說話者
        
        Args:
            speaker_name: 說話者名稱，默認為「未命名說話者」
            
        Returns:
            str: 新建立的說話者 ID
        """
        try:
            # 創建新的說話者
            speaker_collection = self.client.collections.get("Speaker")
            speaker_id = str(uuid.uuid4())
            
            # 如果是默認名稱，生成唯一的名稱 (類似 n1, n2, ...)
            if speaker_name == DEFAULT_SPEAKER_NAME:
                # 獲取所有說話者
                results = speaker_collection.query.fetch_objects(
                    limit=100,
                    return_properties=["name"],
                )
                
                # 提取所有以 'n' 開頭的數字
                numbers = []
                pattern = re.compile(r'^n(\d+)')
                for obj in results.objects:
                    name = obj.properties.get("name", "")
                    match = pattern.match(name)
                    if match:
                        numbers.append(int(match.group(1)))
                
                # 生成下一個編號
                next_number = max(numbers) + 1 if numbers else 1
                speaker_name = f"n{next_number}"
            
            # 創建說話者
            speaker_collection.data.insert(
                properties={
                    "name": speaker_name,
                    "create_time": format_rfc3339(),
                    "last_active_time": format_rfc3339(),
                    "voiceprint_ids": []  # 初始時沒有聲紋向量
                },
                uuid=speaker_id
            )
            
            print(f"(新說話者) 建立新說話者 {speaker_name} (ID: {speaker_id})")
            return speaker_id
            
        except Exception as e:
            print(f"創建新說話者時發生錯誤: {e}")
            raise
    
    def handle_new_speaker(self, new_embedding: np.ndarray) -> Tuple[str, str, str]:
        """
        處理全新的說話者：創建新說話者和嵌入向量
        
        Args:
            new_embedding: 新的嵌入向量
            
        Returns:
            tuple: (說話者ID, 聲紋向量ID, 說話者名稱)
        """
        try:
            # 創建新的說話者
            speaker_id = self.create_new_speaker()
            
            # 獲取說話者名稱
            speaker_collection = self.client.collections.get("Speaker")
            speaker_obj = speaker_collection.query.fetch_object_by_id(
                uuid=speaker_id,
                return_properties=["name"]
            )
            
            if not speaker_obj:
                raise ValueError(f"找不到剛剛創建的說話者 (ID: {speaker_id})")
            
            speaker_name = speaker_obj.properties["name"]
            
            # 創建新的嵌入向量，並與說話者建立關聯
            voice_print_collection = self.client.collections.get("VoicePrint")
            voice_print_id = str(uuid.uuid4())
            
            # 創建聲紋向量
            voice_print_collection.data.insert(
                properties={
                    "create_time": format_rfc3339(),
                    "updated_time": format_rfc3339(),
                    "update_count": 1,
                    "speaker_name": speaker_name
                },
                uuid=voice_print_id,
                vector=new_embedding.tolist(),
                references={
                    "speaker": [speaker_id]
                }
            )
            
            # 更新說話者的聲紋向量列表
            speaker_collection.data.update(
                uuid=speaker_id,
                properties={
                    "voiceprint_ids": [voice_print_id],
                    "last_active_time": format_rfc3339()
                }
            )
            
            print(f"(新說話者) 已建立新說話者 {speaker_name} 和對應的聲紋向量")
            return speaker_id, voice_print_id, speaker_name
            
        except Exception as e:
            print(f"處理新說話者時發生錯誤: {e}")
            raise
    
    def match_speaker(self, voice_print_id: str, speaker_name: str, best_distance: float) -> None:
        """
        匹配到說話者但不進行更新
        
        Args:
            voice_print_id: 匹配到的聲紋向量 ID
            speaker_name: 說話者名稱
            best_distance: 最佳匹配的距離
        """
        print(f"(匹配) 該音檔與說話者 {speaker_name} 的聲紋向量 (ID: {voice_print_id}) 相似 "
              f"(距離 = {best_distance:.4f})，但未進行更新。")
    
    def get_voice_print_properties(self, voice_print_id: str, properties: List[str]) -> Optional[Dict[str, Any]]:
        """
        獲取聲紋向量的屬性
        
        Args:
            voice_print_id: 聲紋向量 ID
            properties: 需要獲取的屬性列表
            
        Returns:
            Optional[Dict[str, Any]]: 屬性字典，若不存在則返回 None
        """
        try:
            voice_print_collection = self.client.collections.get("VoicePrint")
            result = voice_print_collection.query.fetch_object_by_id(
                uuid=voice_print_id,
                return_properties=properties
            )
            
            if not result:
                return None
                
            return result.properties
            
        except Exception as e:
            print(f"獲取聲紋向量屬性時發生錯誤: {e}")
            return None
    
    def update_speaker_voice_prints(self, speaker_id: str, voice_print_id: str) -> bool:
        """
        更新說話者的聲紋向量列表
        
        Args:
            speaker_id: 說話者 ID
            voice_print_id: 要添加的聲紋向量 ID
            
        Returns:
            bool: 是否更新成功
        """
        try:
            speaker_collection = self.client.collections.get("Speaker")
            speaker_obj = speaker_collection.query.fetch_object_by_id(
                uuid=speaker_id,
                return_properties=["voiceprint_ids"]
            )
            
            if not speaker_obj:
                return False
                
            voiceprint_ids = speaker_obj.properties.get("voiceprint_ids", [])
            if voice_print_id not in voiceprint_ids:
                voiceprint_ids.append(voice_print_id)
                
                speaker_collection.data.update(
                    uuid=speaker_id,
                    properties={
                        "voiceprint_ids": voiceprint_ids,
                        "last_active_time": format_rfc3339()
                    }
                )
            
            return True
            
        except Exception as e:
            print(f"更新說話者聲紋向量列表時發生錯誤: {e}")
            return False
    
    def close(self) -> None:
        """關閉 Weaviate 連接"""
        if hasattr(self, 'client'):
            self.client.close()
            print("已關閉 Weaviate 連接")


class SpeakerIdentifier:
    """說話者識別類，負責核心識別邏輯"""
    
    def __init__(self) -> None:
        """初始化說話者識別器"""
        self.audio_processor = AudioProcessor()
        self.database = WeaviateRepository()
        self.threshold_low = THRESHOLD_LOW
        self.threshold_update = THRESHOLD_UPDATE
        self.threshold_new = THRESHOLD_NEW
    
    def _handle_very_similar(self, best_id: str, best_name: str, best_distance: float) -> Tuple[str, str, float]:
        """
        處理過於相似的情況：不更新向量
        
        Args:
            best_id: 最佳匹配ID
            best_name: 最佳匹配說話者名稱
            best_distance: 最佳匹配距離
            
        Returns:
            Tuple[str, str, float]: (說話者ID, 說話者名稱, 相似度)
        """
        print(f"(跳過) 嵌入向量過於相似 (距離 = {best_distance:.4f})，不進行更新。")
        print(f"該音檔與說話者 {best_name} 的檔案相同。")
        return best_id, best_name, best_distance
    
    def _handle_update_embedding(self, best_id: str, best_name: str, best_distance: float, new_embedding: np.ndarray) -> Tuple[str, str, float]:
        """
        處理需要更新嵌入向量的情況
        
        Args:
            best_id: 最佳匹配ID
            best_name: 最佳匹配說話者名稱
            best_distance: 最佳匹配距離
            new_embedding: 新的嵌入向量
            
        Returns:
            Tuple[str, str, float]: (說話者ID, 說話者名稱, 相似度)
        """
        try:
            # 獲取當前更新次數
            properties = self.database.get_voice_print_properties(best_id, ["update_count"])
            if properties is None:
                raise ValueError(f"無法獲取聲紋向量 ID {best_id} 的屬性")
            
            update_count = properties["update_count"]
            
            # 更新嵌入向量
            self.database.update_embedding(best_id, new_embedding, update_count)
            print(f"該音檔與說話者 {best_name} 相符，且已更新嵌入檔案。")
            return best_id, best_name, best_distance
        except Exception as e:
            print(f"更新嵌入向量時發生錯誤: {e}")
            raise
    
    def _handle_match_only(self, best_id: str, best_name: str, best_distance: float) -> Tuple[str, str, float]:
        """
        處理僅匹配但不更新的情況
        
        Args:
            best_id: 最佳匹配ID
            best_name: 最佳匹配說話者名稱
            best_distance: 最佳匹配距離
            
        Returns:
            Tuple[str, str, float]: (說話者ID, 說話者名稱, 相似度)
        """
        self.database.match_speaker(best_id, best_name, best_distance)
        return best_id, best_name, best_distance
    
    def _handle_new_speaker(self, new_embedding: np.ndarray) -> Tuple[str, str, float]:
        """
        處理新說話者的情況：創建新說話者
        
        Args:
            new_embedding: 新的嵌入向量
            
        Returns:
            Tuple[str, str, float]: (說話者ID, 說話者名稱, 相似度)
        """
        speaker_id, voice_print_id, speaker_name = self.database.handle_new_speaker(new_embedding)
        return speaker_id, speaker_name, -1  # -1 表示全新的說話者
    
    def process_audio_file(self, audio_file: str) -> Optional[Tuple[str, str, float]]:
        """
        處理音檔並進行說話者識別
        
        Args:
            audio_file: 音檔路徑
            
        Returns:
            Optional[Tuple[str, str, float]]: (說話者ID, 說話者名稱, 相似度) 或 None 表示處理失敗
        """
        try:
            print(f"\n處理音檔: {audio_file}")
            if not os.path.exists(audio_file):
                print(f"音檔 {audio_file} 不存在，取消處理。")
                return None
            
            # 提取嵌入向量
            new_embedding = self.audio_processor.extract_embedding(audio_file)
            
            # 與 Weaviate 中的嵌入向量比對
            best_id, best_name, best_distance, _ = self.database.compare_embedding(new_embedding)
            
            # 根據距離進行判斷，使用輔助函數處理不同情況
            if best_id is None:
                # 資料庫為空，直接創建新說話者
                print("資料庫為空，創建新說話者")
                result = self._handle_new_speaker(new_embedding)
            elif best_distance < self.threshold_low:
                # 過於相似，不更新
                result = self._handle_very_similar(best_id, best_name, best_distance)
            elif best_distance < self.threshold_update:
                # 距離在允許的範圍內，更新嵌入向量
                result = self._handle_update_embedding(best_id, best_name, best_distance, new_embedding)
            elif best_distance < self.threshold_new:
                # 距離在匹配範圍內，但不更新
                result = self._handle_match_only(best_id, best_name, best_distance)
            else:
                # 判定為新說話者
                result = self._handle_new_speaker(new_embedding)
                
            return result
                
        except Exception as e:
            print(f"處理音檔時發生錯誤: {e}")
            return None
        finally:
            # 關閉 Weaviate 連接
            self.database.close()
    
    def process_audio_directory(self, directory: str) -> Dict[str, Any]:
        """
        處理指定資料夾內所有 .wav 檔案
        
        Args:
            directory: 資料夾路徑
            
        Returns:
            Dict[str, Any]: 處理結果統計
        """
        if not os.path.exists(directory):
            print(f"資料夾 {directory} 不存在，取消處理。")
            return {"success": False, "error": "資料夾不存在"}
            
        audio_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".wav")]
        if not audio_files:
            print(f"資料夾 {directory} 中沒有 .wav 檔案。")
            return {"success": False, "error": "無音檔"}
            
        print(f"發現 {len(audio_files)} 個音檔於 {directory}，開始處理...")
        
        results = {
            "total": len(audio_files),
            "processed": 0,
            "failed": 0,
            "new_speakers": 0,
            "updated_speakers": 0,
            "matched_speakers": 0,
            "details": []
        }
        
        try:
            for audio_file in audio_files:
                try:
                    result = self.process_audio_file(audio_file)
                    results["processed"] += 1
                    
                    if result:
                        speaker_id, speaker_name, distance = result
                        results["details"].append({
                            "file": audio_file,
                            "speaker_id": speaker_id,
                            "speaker_name": speaker_name,
                            "distance": distance
                        })
                        
                        # 計算統計
                        if distance == 1.0:
                            results["new_speakers"] += 1
                        elif distance < self.threshold_update:
                            results["updated_speakers"] += 1
                        else:
                            results["matched_speakers"] += 1
                        
                except Exception as e:
                    print(f"處理 {audio_file} 時發生錯誤: {e}")
                    results["failed"] += 1
                    results["details"].append({
                        "file": audio_file,
                        "error": str(e)
                    })
        finally:
            # 關閉 Weaviate 連接
            self.database.close()
                
        print(f"\n完成處理資料夾 {directory} 中所有音檔。")
        print(f"處理總數: {results['processed']}/{results['total']}, 失敗: {results['failed']}")
        print(f"新增說話者: {results['new_speakers']}, 更新說話者: {results['updated_speakers']}, 匹配說話者: {results['matched_speakers']}")
        
        return results
    
    def add_embedding_to_existing_speaker(self, audio_file: str, speaker_id: str) -> bool:
        """
        添加音檔的嵌入向量到現有的說話者，但不進行加權平均更新
        
        Args:
            audio_file: 音檔路徑
            speaker_id: 說話者 ID
            
        Returns:
            bool: 是否成功添加
        """
        try:
            print(f"\n添加音檔嵌入向量到說話者 (ID: {speaker_id}): {audio_file}")
            if not os.path.exists(audio_file):
                print(f"音檔 {audio_file} 不存在，取消處理。")
                return False
            
            # 檢查說話者是否存在
            speaker_collection = self.database.client.collections.get("Speaker")
            speaker_obj = speaker_collection.query.fetch_object_by_id(
                uuid=speaker_id,
                return_properties=["name"]
            )
            
            if not speaker_obj:
                print(f"說話者 ID {speaker_id} 不存在，取消處理。")
                return False
                
            speaker_name = speaker_obj.properties["name"]
            
            # 提取嵌入向量
            new_embedding = self.audio_processor.extract_embedding(audio_file)
            
            # 添加新的嵌入向量到說話者
            voice_print_id = self.database.add_embedding_without_averaging(
                speaker_name, new_embedding, speaker_id)
            
            # 更新說話者的聲紋向量列表
            success = self.database.update_speaker_voice_prints(speaker_id, voice_print_id)
            
            if success:
                print(f"已成功將音檔嵌入向量添加到說話者 {speaker_name}")
            
            return success
                
        except Exception as e:
            print(f"添加嵌入向量時發生錯誤: {e}")
            return False
        finally:
            # 關閉 Weaviate 連接
            self.database.close()


if __name__ == "__main__":
    # 創建說話者識別器
    identifier = SpeakerIdentifier()
    
    # 主程式執行: 若要處理單一檔案或資料夾，可解除下列註解
    identifier.process_audio_file("testFiles/test_audioFile/0770/770-2.wav")
    # identifier.process_audio_directory("path_to_directory")
    
    # 如果需要將嵌入向量添加到現有說話者，可解除下列註解
    # identifier.add_embedding_to_existing_speaker("path_to_audio.wav", "speaker_uuid")
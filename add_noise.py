import numpy as np
import soundfile as sf

def add_noise_to_audio(input_file, output_file, noise_level=0.01):
    """
    為音訊檔案添加隨機雜訊
    :param input_file: 原始音訊檔案路徑 (如 'input.wav')
    :param output_file: 添加雜訊後的音訊檔案路徑 (如 'output_with_noise.wav')
    :param noise_level: 雜訊強度 (建議範圍 0.01 ~ 0.1)
    """
    # 讀取音訊檔案
    audio, samplerate = sf.read(input_file)

    # 確保音訊是 NumPy 陣列
    audio = np.asarray(audio)

    # 生成隨機雜訊
    noise = np.random.normal(0, noise_level, audio.shape)

    # 添加雜訊到音訊中
    audio_with_noise = audio + noise

    # 確保音訊數據不超過 [-1, 1] 範圍
    audio_with_noise = np.clip(audio_with_noise, -1, 1)

    # 將處理後的音訊存成檔案
    sf.write(output_file, audio_with_noise, samplerate)
    print(f"已將雜訊添加至音訊，輸出檔案：{output_file}")


add_noise_to_audio('mixed_voice.wav', 'mixed_voice_with_noise.wav', noise_level=0.01)

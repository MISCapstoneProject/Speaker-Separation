import whisper

def audio_to_text_whisper(audio_file, model_size="small"):
    """
    讀取音檔 audio_file，使用OpenAI Whisper進行語音轉文字。
    model_size 可選: tiny, base, small, medium, large
    """
    # 如果有GPU，可以用 'cuda'；沒有GPU就用CPU
    model = whisper.load_model(model_size, device="cpu")
    
    # language可以指定"zh"、"en"或讓它自動偵測
    result = model.transcribe(audio_file, fp16=False, language="en")
    return result["text"]

if __name__ == "__main__":
    input_audio = "Audios/speaker1_20250211-18_12_39_1.wav"
    recognized_text = audio_to_text_whisper(input_audio, model_size="small")
    print("Whisper辨識結果：", recognized_text)

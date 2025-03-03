import whisper

def audio_to_text_whisper(audio_file, model_size="small"):
    """
    讀取音檔 audio_file，使用OpenAI Whisper進行語音轉文字。
    model_size 可選: tiny, base, small, medium, large
    """
    # 如果有GPU，可以用 'cuda'；沒有GPU就用CPU
    model = whisper.load_model(model_size, device="cpu")
    
    # language可以指定"zh"、"en"或讓它自動偵測
    result = model.transcribe(audio_file, fp16=False, language="zh")
    return result["text"]

if __name__ == "__main__":
    input_audio = "16K-model\Audios-16K\\final_speaker2_20250303-17_23_00.wav"
    recognized_text = audio_to_text_whisper(input_audio, model_size="small")
    print("Whisper辨識結果：", recognized_text)

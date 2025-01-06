import wenet

model = wenet.load_model('chinese')
result = model.transcribe('audio.wav')
print(result['text'])
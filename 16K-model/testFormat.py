import pyaudio

p = pyaudio.PyAudio()

device_index = 2  # 假設要檢測第0號裝置，可自行調整
sample_rate = 44100
channels = 1

try:
    supported = p.is_format_supported(
        rate=sample_rate,
        input_device=device_index,
        input_channels=channels,
        input_format=pyaudio.paFloat32
    )
    if supported:
        print("裝置支援 paFloat32")
    else:
        print("裝置不支援 paFloat32")
except ValueError as e:
    print("裝置不支援 paFloat32 或其他參數不相容", e)

p.terminate()

from speechbrain.inference import SepformerSeparation as seperator
import torchaudio


model = seperator.from_hparams(
    source='speechbrain/sepformer-libri3mix',
    savedir='pretrained_models/sepformer-libri3mix'
)


est_sources = model.separate_file(path= "16K-model/Audios-16K/mixed_audio_20250218-17_45_26.wav")


print(f"分離後音頻形狀: {est_sources.shape}")
print(f"分離後數據範圍: {est_sources.min().item()}, {est_sources.max().item()}")


# 分離語者
torchaudio.save("separate_voice/speaker1-0223.wav", est_sources[:, :, 0].detach().cpu(), 8000)
torchaudio.save("separate_voice/speaker2-0223.wav", est_sources[:, :, 1].detach().cpu(), 8000)
torchaudio.save("separate_voice/speaker3-0223.wav", est_sources[:, :, 2].detach().cpu(), 8000)

print("語者分離完成")
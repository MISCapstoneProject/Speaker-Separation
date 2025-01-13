from speechbrain.inference import SepformerSeparation as separator
import torch
import time

model = separator.from_hparams(
    source="speechbrain/sepformer-wsj02mix",
    savedir="pretrained_models/sepformer-wsj02mix"
)

audio_tensor = torch.randn(1, 8000 * 5)  # 5 秒假音訊
start_time = time.time()
est_sources = model.separate_batch(audio_tensor)
print(f"分離完成，耗時：{time.time() - start_time:.2f} 秒")
import os
import torch

from compressai.zoo import image_models

model_name = "mbt2018-mean"

for quality in range(1, 9):
    model = image_models[model_name](quality=quality, pretrained=True)
    os.makedirs(os.path.dirname(f"./models/mbt2018_mean/quality_{quality}/g_a"), exist_ok=True)
    torch.save(model.g_a.state_dict(), f"./models/mbt2018_mean/quality_{quality}/g_a")
    torch.save(model.g_s.state_dict(), f"./models/mbt2018_mean/quality_{quality}/g_s")
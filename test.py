from pathlib import Path

from models_config import DATASET_CLASS_COUNT, build_timm_model_str
import timm
import numpy as np

weights_path = Path("/data/weights")

weight_paths = list(weights_path.iterdir())

def load_state_dict(weight_path):
    pass

def load_model(weight_path):
    weight_name = weight_path.stem
    config_name = weight_name.split('-')[0]
    dataset_name = weight_name.split('-')[1]
    drop_rate = float(weight_name.split('-')[-2].split('_')[1])
    drop_path_rate = float(weight_name.split('-')[-1].split('_')[1])
    num_classes = DATASET_CLASS_COUNT[dataset_name]

    model_str = build_timm_model_str(config_name, dataset_name)

    model = timm.create_model(
        model_str, 
        num_classes=num_classes, 
        drop_rate=drop_rate, 
        attn_drop_rate=drop_rate, 
        drop_path_rate=drop_path_rate
    )

    print(weight_path, model.patch_embed.proj.weight.shape)

    timm.models.load_checkpoint(model, weight_path)
    
    return model

for weight_path in weight_paths:
    model = load_model(weight_path)

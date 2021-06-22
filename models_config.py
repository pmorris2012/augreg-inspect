DATASET_CLASS_COUNT = {
    "i1k": 1000,
    "i21k": 21843,
    "i21k_30": 21843
}

MODEL_CONFIG_DICT = {
    "Ti_16": "vit_tiny_r_s16_p8_224",#"vit_tiny_patch16_224",
    "S_16": "vit_small_patch16_224",
    "S_32": "vit_small_r26_s32_224",#"vit_small_patch32_224",
    "B_16": "vit_base_patch16_224",
    "B_32": "vit_base_patch32_224",
    "L_16": "vit_large_patch16_224",
    "R_Ti_16": "vit_tiny_r_s16_p8_224",
    "R26_S_32": "vit_small_r26_s32_224",
    "R50_L_32": "vit_large_r50_s32_224"
}

def build_timm_model_str(config_name, dataset_name):
    model_str = MODEL_CONFIG_DICT[config_name]
    num_classes = DATASET_CLASS_COUNT[dataset_name]
    
    if num_classes == DATASET_CLASS_COUNT["i21k"]:
        model_str += "_in21k"

    return model_str

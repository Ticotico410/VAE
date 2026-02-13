import os
import kagglehub

# TODO:
# API Token: KGAT_f869f111b67dbb6b5d671b151be9e2a1
# export KAGGLE_API_TOKEN=KGAT_f869f111b67dbb6b5d671b151be9e2a1
# kaggle competitions list

output_dir = "/home/ycb410/ycb_ws/vae/datasets"
os.makedirs(output_dir, exist_ok=True)
path = kagglehub.dataset_download(
    "paultimothymooney/chest-xray-pneumonia",
    output_dir=output_dir
)

print("Path to dataset files:", path)
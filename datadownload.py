import kagglehub
import os

# 指定数据集保存路径
output_dir = "/home/ycb410/ycb_ws/vae/datasets"

# 如果目录不存在，创建它
os.makedirs(output_dir, exist_ok=True)

# 下载数据集到指定目录
path = kagglehub.dataset_download(
    "paultimothymooney/chest-xray-pneumonia",
    output_dir=output_dir
)

print("Path to dataset files:", path)


# TODO:
# API Token: KGAT_f869f111b67dbb6b5d671b151be9e2a1
# export KAGGLE_API_TOKEN=KGAT_f869f111b67dbb6b5d671b151be9e2a1
# kaggle competitions list
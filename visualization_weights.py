import os
import sys
import json

sys.path.append("..")

import matplotlib.pyplot as plt

import Model

if not os.path.exists("images"):
    os.makedirs("images")

model_path = "models/final_model.pkl"
nn_architecture = json.load(open(model_path.replace(".pkl", ".json"), "r"))

# 初始化的模型
model = Model.MLPModel(nn_architecture)
for i, layer in enumerate(model.layers):
    if isinstance(layer, Model.Linear):
        # 绘制各层参数的热力图
        plt.figure()
        plt.imshow(layer.W, cmap="hot", interpolation="nearest")
        plt.title(f"Layer {i // 2 + 1} Weight Matrix Initialization")
        plt.colorbar()
        plt.savefig(f"images/layer_{i // 2 + 1}_weight_matrix_init.png")

# 训练后的模型
model.load_model_dict(path=model_path)
for i, layer in enumerate(model.layers):
    if isinstance(layer, Model.Linear):
        # 绘制各层参数的热力图
        plt.figure()
        plt.imshow(layer.W, cmap="hot", interpolation="nearest")
        plt.title(f"Layer {i // 2 + 1} Weight Matrix After Training")
        plt.colorbar()
        plt.savefig(f"images/layer_{i // 2 + 1}_weight_matrix.png")

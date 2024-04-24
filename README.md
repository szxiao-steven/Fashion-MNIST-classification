# Fashion-MNIST Classification

Fashion-MNIST classification with 3-layer neural network based on numpy.

## 安装依赖

```bash
pip install -r requirements.txt
```

## 模型训练

可自定义修改 [`train.py`](train.py) 中的以下部分：

* 数据加载器参数：可指定数据集所在路径、验证集比例、训练的batch size，例如：

```python
dataloader_kwargs = {
    "path_dir": "./Fashion-MNIST",
    "val_size": 0.1,
    "batch_size": 16,
}
```

* 神经网络模型结构，即各个线性层的输入维度、输出维度和激活函数类型，例如：

```python
nn_architecture = [
    {"input_dim": 784, "output_dim": 512, "activation": "relu"},
    {"input_dim": 512, "output_dim": 128, "activation": "relu"},
    {"input_dim": 128, "output_dim": 10, "activation": "softmax"},
]
```

其中，激活函数目前支持的指定类型包括`"sigmoid"`、`"relu"`、`"leakyrelu"`和`"softmax"`；也可在[`Model.py`](Model.py)中修改`Activation`类实现其他自定义的激活函数。

* SGD优化器参数：可指定学习率、L2正则化系数、学习率衰减系数、学习率衰减步数，例如：

```python
optimizer_kwargs = {
    "lr": 0.01,
    "ld": 0.001,
    "decay_rate": 0.95,
    "decay_step": 200,
}
```

* 训练器参数：可指定训练的轮次、验证集评估轮次的频率，例如：

```python
trainer_kwargs = {
    "n_epochs": 50,
    "eval_step": 1,
}
```

设定好上述参数后，运行train.py即可进行模型训练。训练过程的日志`train_log.txt`将被保存至自动生成的`logs/`目录下；根据验证集指标（默认为accuracy，也可修改为loss）自动保存最优的模型权重，同时还会保存最后一个epoch的模型权重。

## 模型测试

将模型权重存放至`models/`目录下；可自定义修改 [`test.py`](test.py) 中的以下部分：

* 指定数据集所在路径，例如：

```python
dataloaders_kwargs = {
    "path_dir": "./Fashion-MNIST"
}
```

* 指定模型权重文件的路径，例如：

```python
model_path = "models/final_model.pkl"
```

运行`test.py`即可即可完成测试集评测。

import matplotlib.pyplot as plt
from Dataloader import FashionMNISTDataloader
from Model import MLPModel, CrossEntropyLoss, SGDOptimizer
from Trainer import Trainer


dataloader_kwargs = {
    "val_size": 0.1,
    "batch_size": 16
}

nn_architecture = [
    {"input_dim": 784, "output_dim": 512, "activation": "relu"},
    {"input_dim": 512, "output_dim": 128, "activation": "relu"},
    {"input_dim": 128, "output_dim": 10, "activation": "softmax"}
]

optimizer_kwargs = {
    "lr": 0.01,
    "ld": 0.001,
    "decay_rate": 0.95,
    "decay_step": 200
}

trainer_kwargs = {
    "n_epochs": 50,
    "eval_step": 1
}


if __name__ == "__main__":
    dataloader = FashionMNISTDataloader(**dataloader_kwargs)
    model = MLPModel(nn_architecture)
    optimizer = SGDOptimizer(**optimizer_kwargs)
    loss = CrossEntropyLoss()

    trainer = Trainer(model, optimizer, loss, dataloader, **trainer_kwargs)
    trainer.train(save_ckpt=True, verbose=True)  # 训练模型
    trainer.save_log("./logs/")  # 保存训练日志
    trainer.save_best_model("./models/", metric="acc", n=1, keep_last=True)  # 保存最优模型
    trainer.clear_cache()

    plt.show(block=True)

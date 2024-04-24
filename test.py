import numpy as np
from Dataloader import FashionMNISTDataloader
from Model import MLPModel, CrossEntropyLoss

dataloaders_kwargs = {
    "path_dir": "./Fashion-MNIST"
}

model_path = "models/final_model.pkl"


if __name__ == "__main__":
    dataloader = FashionMNISTDataloader(**dataloaders_kwargs)
    model = MLPModel()
    model.load_model_dict(model_path)  # 加载模型权重
    loss = CrossEntropyLoss()

    total_loss = 0
    total_acc = 0

    for x_batch, y_batch in dataloader.generate_test_batch():
        y_pred = model.forward(x_batch)
        total_acc += np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))
        ce_loss = loss.forward(y_pred, y_batch)
        total_loss += ce_loss * len(x_batch)

    test_loss = total_loss / len(dataloader.y_test)
    test_acc = total_acc / len(dataloader.y_test)

    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} |")

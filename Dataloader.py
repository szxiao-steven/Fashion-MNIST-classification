import os
import gzip
import numpy as np


class FashionMNISTDataloader:
    def __init__(self, path_dir="./Fashion-MNIST", val_size=0.1, batch_size=32, random_seed=2024):
        X, y = self.load_mnist(path_dir, kind="train")
        self.x_train, self.y_train, self.x_val, self.y_val = self.train_val_split(X, y, val_size, random_seed)
        self.x_test, self.y_test = self.load_mnist(path_dir, kind="t10k")
        self.batch_size = batch_size

    @staticmethod
    def load_mnist(path, kind='train'):
        labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
        images_path = os.path.join(path,
                                '%s-images-idx3-ubyte.gz'
                                % kind)

        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                   offset=8)
            labels = np.eye(10)[labels]

        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                   offset=16).reshape(len(labels), 784)
            images = images.astype(np.float32) / 255.0

        return images, labels

    @staticmethod
    def train_val_split(x_train, y_train, val_size, random_seed):
        n_samples = x_train.shape[0]
        np.random.seed(random_seed)
        indices = np.random.permutation(n_samples)
        n_val = int(n_samples * val_size)
        valid_indices = indices[:n_val]
        train_indices = indices[n_val:]
        return x_train[train_indices], y_train[train_indices], x_train[valid_indices], y_train[valid_indices]

    def generate_train_batch(self):
        n_samples = self.x_train.shape[0]
        indices = np.random.permutation(self.x_train.shape[0])
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.x_train[batch_indices], self.y_train[batch_indices]

    def generate_val_batch(self):
        n_samples = self.x_val.shape[0]
        indices = np.arange(n_samples)
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.x_val[batch_indices], self.y_val[batch_indices]

    def generate_test_batch(self):
        n_samples = self.x_test.shape[0]
        indices = np.arange(n_samples)
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.x_test[batch_indices], self.y_test[batch_indices]


if __name__ == "__main__":
    dataloader_kwargs = {
        # "path_dir": "./Fashion-MNIST",
        "val_size": 0.1,
        "batch_size": 16,
    }
    dataloader = FashionMNISTDataloader(**dataloader_kwargs)
    print(dataloader.x_train.shape)
    print(dataloader.x_val.shape)
    print(dataloader.x_test.shape)
    print(dataloader.y_test)
    
    # import matplotlib.pyplot as plt
    # data=dataloader.x_train[0,:].reshape(28,28)
    # plt.imshow(data,cmap='Greys',interpolation=None)
    # plt.show()
    
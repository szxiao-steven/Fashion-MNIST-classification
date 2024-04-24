import numpy as np
import json
import pickle


class Layer:
    def __init__(self):
        self.input_cache = None


class Activation(Layer):
    def __init__(self, activation_type):
        super().__init__()
        self.type = activation_type
        self.leakyrelu_alpha = 0.01 if activation_type == "leakyrelu" else None

    def forward(self, z):
        self.input_cache = z
        if self.type == "relu":
            return np.maximum(0, z)
        elif self.type == "leakyrelu":
            return np.where(z > 0, z, z * self.leakyrelu_alpha)
        elif self.type == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif self.type == "softmax":
            exps = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exps / np.sum(exps, axis=1, keepdims=True)

    def backward(self, dA, z):
        if self.type == "relu":
            return dA * (z > 0)
        elif self.type == "leakyrelu":
            dz = np.ones_like(z)
            dz[z < 0] = self.leakyrelu_alpha
            return dA * dz
        elif self.type == "sigmoid" or self.type == "softmax":
            sig = self.forward(z)
            return dA * sig * (1 - sig)


class Linear(Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.W = np.random.normal(0, pow(input_dim, -0.5), (input_dim, output_dim))
        self.b = np.zeros((1, output_dim))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.input_cache = x
        return np.dot(x, self.W) + self.b

    def backward(self, dZ, x):
        dW = np.dot(x.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        dx = np.dot(dZ, self.W.T)
        return dW, db, dx

    def zero_grad(self):
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)


class CrossEntropyLoss:
    def __init__(self):
        self.epsilon = 1e-12  # 防止取 log 时出现无穷大
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        self.y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        self.y_true = y_true
        return -np.sum(self.y_true * np.log(self.y_pred), axis=1).mean()

    def backward(self):
        grad = (self.y_pred - self.y_true) / self.y_pred.shape[0]
        return grad


class SGDOptimizer:
    def __init__(self, lr=0.01, ld=0.001, decay_rate=0.95, decay_step=200):
        self.lr = lr
        self.ld = ld
        self.decay_rate = decay_rate if 0 < decay_rate < 1 else None
        self.decay_step = decay_step if 0 < decay_rate < 1 else None
        self.iterations = 0

    def step(self, model):
        for layer in model.layers:
            if isinstance(layer, Linear):
                layer.W -= self.lr * (layer.dW + self.ld * layer.W)
                layer.b -= self.lr * layer.db
                layer.zero_grad()
        self.iterations += 1
        if self.decay_rate and self.iterations % self.decay_step == 0:
            self.lr *= self.decay_rate if self.lr > 0.01 else 1


class MLPModel:
    def __init__(self, nn_architecture=None):
        self.layers = []
        self.nn_architecture = nn_architecture if nn_architecture else []
        for layer in self.nn_architecture:
            self.add(Linear(layer["input_dim"], layer["output_dim"]))
            self.add(Activation(layer["activation"]))

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            if isinstance(layer, Linear):
                dW, db, grad = layer.backward(grad, layer.input_cache)
                layer.dW = dW
                layer.db = db
            elif isinstance(layer, Activation):
                grad = layer.backward(grad, layer.input_cache)
        return grad

    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)
    
    def deep_copy(self):
        model_copy = MLPModel(self.nn_architecture)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                model_copy.layers[i].W = layer.W.copy()
                model_copy.layers[i].b = layer.b.copy()
        return model_copy

    def save_model_dict(self, path):
        model_dict = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                model_dict[f"layer_{i}_W"] = layer.W
                model_dict[f"layer_{i}_b"] = layer.b
        with open(path, "wb") as f:
            pickle.dump(model_dict, f)
        with open(path.replace(".pkl", ".json"), "w") as f:
            json.dump(self.nn_architecture, f)

    def load_model_dict(self, path):
        with open(path.replace(".pkl", ".json"), "r") as f:
            nn_architecture = json.load(f)
        self.__init__(nn_architecture)
        with open(path, "rb") as f:
            model_dict = pickle.load(f)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                layer.W = model_dict[f"layer_{i}_W"]
                layer.b = model_dict[f"layer_{i}_b"]
                layer.zero_grad()
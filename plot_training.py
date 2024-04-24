import numpy as np
import matplotlib.pyplot as plt

file_path = './logs/train_log.txt'

epochs = []
train_losses = []
valid_losses = []
valid_accs = []

with open(file_path, 'r') as f:
    for line in f.readlines():
        if 'Epoch' in line:
            parts = line.split('|')
            epoch = int(parts[0].split()[1])
            epochs.append(epoch)
            
            train_loss = float(parts[2].split(':')[1])
            train_losses.append(train_loss)
            
            valid_loss = float(parts[4].split(':')[1])
            valid_losses.append(valid_loss)
            
            valid_acc = float(parts[5].split(':')[1])
            valid_accs.append(valid_acc)

# 转换为numpy数组
epochs = np.array(epochs)
train_losses = np.array(train_losses)
valid_losses = np.array(valid_losses)
valid_accs = np.array(valid_accs)

# 绘制loss曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss', marker='o')
plt.plot(epochs, valid_losses, label='Validation Loss', marker='x')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./images/loss_curve.png')

# 绘制accuracy曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, valid_accs, label='Validation Accuracy', marker='o', color='r')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend()
plt.savefig('./images/accuracy_curve.png')

# # 显示图形
# plt.tight_layout()
# plt.show()
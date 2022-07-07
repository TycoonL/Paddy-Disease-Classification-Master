import random

import torch
import torch.nn as nn
import torch.optim as optim
from model import CNN
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
# 模型代码编写，forward 就是直接调用 model(x) 时执行的计算流程
from torchvision.transforms import Lambda

# 测试能否使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数设定
input_size = 784
in_channel = 3
num_classes = 10
learning_rate = 0.001
batch_size = 32
num_epochs = 20

model_path =None
# 读取数据集

transforms = transforms.Compose([
    transforms.Resize(64),    # 将图片短边缩放至256，长宽比保持不变：
    #这里如果不裁剪会出现RuntimeError: stack expects each tensor to be equal size, but got [3, 85, 64] at entry 0 and [3, 64, 85] 错误
    #可能因为有些图片在裁剪前是翻转的
    transforms.CenterCrop((64,42)),   #将图片从中心切剪成3*224*224大小的图片
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.ToTensor()          #把图片进行归一化，并把数据转换成Tensor类型
])
#onehot 转换
target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

path = r'.\Paddy Doctor Dataset\train_images'
data = datasets.ImageFolder(path,transforms)
n = len(data)  # total number of examples
n_test = random.sample(range(1, n), int(0.1 * n))  # take ~10% for test
test_set = torch.utils.data.Subset(data, n_test)  # take 10%
train_set = torch.utils.data.Subset(data,list(set(range(1, n)).difference(set(n_test))))  # take the rest

data_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
data_test=DataLoader(test_set, batch_size=batch_size, shuffle=False)

# for batch_idex, (data, targets) in enumerate(data_test):
#     print(batch_idex,targets)

# 实例化模型
model = CNN(in_channels=in_channel,num_classes=num_classes)
if model_path:
    model.load_state_dict(torch.load(model_path))
model.to(device)

# 设定损失函数和优化器
criterion = nn.CrossEntropyLoss()  #label不需要onehot，不需要softmax层
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 下面部分是训练，有时可以单独拿出来写函数
for epoch in range(num_epochs):
    for batch_idex, (data, targets) in enumerate(data_train):
        # 如果模型在 GPU 上，数据也要读入 GPU
        data = data.to(device=device)
        targets = targets.to(device=device)
        # print(data.shape)   # [64,1,28,28] Batch 大小 64 , 1 channel, 28*28 像素
        # forward 前向模型计算输出，然后根据输出算损失
        scores = model(data)
        loss = criterion(scores, targets)
        # backward 反向传播计算梯度
        optimizer.zero_grad()
        loss.backward()
        # 梯度下降，优化参数
        optimizer.step()
        if batch_idex % 20 == 0:
            print('Train Epoch: {} {} Loss: {:.6f}'.format(epoch,batch_idex,loss.item()))
model_pth = 'model.pth'
torch.save(model.state_dict(), model_pth)

# 评估准确度的函数
def check_accuracy(loader, model):
    # if loader.dataset.train:
    #     print("Checking acc on training data")
    # else:
    #     print("Checking acc on testing data")
    num_correct = 0
    num_samples = 0
    model.eval()  # 将模型调整为 eval 模式，具体可以搜一下区别
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            # 64*10
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        acc = float(num_correct) / float(num_samples) * 100
        print(f'Got {num_correct} / {num_samples} with accuracy {acc:.2f}')

    model.train()
    return acc

check_accuracy(data_train, model)
check_accuracy(data_test, model)
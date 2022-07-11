import random
from os import walk

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import DataLoader
from torchvision import transforms, datasets,models
from torchvision.transforms import Lambda
from check_accuracy import check_accuracy

# 测试能否使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数设定
input_size = 784
in_channel = 3
num_classes = 10
learning_rate = 0.001
batch_size = 32
num_epochs = 100
h=224

model_path =None
# 读取数据集
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomChoice([
        transforms.Pad(padding=10),
        transforms.CenterCrop(480),
        transforms.RandomRotation(20),
        transforms.CenterCrop((576, 432)),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.1
        )
    ]),
    transforms.Resize((h, h)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
#onehot 转换
target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

path = r'.\Paddy Doctor Dataset\train_images'
data = datasets.ImageFolder(path,transform)
print(data.class_to_idx,data.classes)

n = len(data)  # total number of examples
n_test = random.sample(range(1, n), int(0.01 * n))  # take ~10% for test
test_set = torch.utils.data.Subset(data, n_test)  # take 10%
train_set = torch.utils.data.Subset(data,list(set(range(1, n)).difference(set(n_test))))  # take the rest

data_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
data_test=DataLoader(test_set, batch_size=batch_size, shuffle=False)


# # 实例化模型
# model = CNN(H=h,in_channels=in_channel,num_classes=num_classes)
# if model_path:
#     model.load_state_dict(torch.load(model_path))
# model.to(device)
model = models.resnet101(pretrained=False)
model.fc = nn.Sequential(
    nn.Dropout(0.1),
    nn.Linear(model.fc.in_features, 10)
)
model_path=None#'resnet32_model.pth'
if model_path:
     model.load_state_dict(torch.load(model_path))
model = model.to(device)

# 设定损失函数和优化器
criterion = nn.CrossEntropyLoss()  #label不需要onehot，不需要softmax层
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#动态学习率
scheduler = ReduceLROnPlateau(optimizer, mode='max',verbose=1,patience=3)

model.train()
# 下面部分是训练
for epoch in range(num_epochs):
    for batch_idex, (data, targets) in enumerate(data_train):
        # 如果模型在 GPU 上，数据也要读入 GPU
        data = data.to(device=device)
        targets = targets.to(device=device)
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
    val_acc=check_accuracy(data_test, model)
    scheduler.step(val_acc)
model_pth = 'resnet101_model.pth'
torch.save(model.state_dict(), model_pth)


check_accuracy(data_train, model)
check_accuracy(data_test, model)

submission_dir = r'./Paddy Doctor Dataset/test_images/'
dataset_file = './Paddy Doctor Dataset/train.csv'
submission_sample = './Paddy Doctor Dataset/sample_submission.csv'
submission_output = './Paddy Doctor Dataset/submission_resnet101.csv'

df = pd.read_csv(dataset_file)

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 测试能否使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet34(pretrained=False)
model.fc = nn.Sequential(
    nn.Dropout(0.1),
    nn.Linear(model.fc.in_features, 10)
)
model_path='resnet101_model.pth'
if model_path:
     model.load_state_dict(torch.load(model_path))
model = model.to(device)

idx_to_label= ['bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight', 'blast', 'brown_spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro']
model.eval()
image_ids, labels = [], []
for (dirpath, dirname, filenames) in walk(submission_dir):
    for filename in filenames:
        image = Image.open(dirpath+filename)
        image = test_transform(image)
        image = image.unsqueeze(0).to(device)
        image_ids.append(filename)
        labels.append(idx_to_label[model(image).argmax().item()])

submission = pd.DataFrame({
    'image_id': image_ids,
    'label': labels,
})
submission['label'].value_counts()
submission.to_csv(submission_output, index=False, header=True)
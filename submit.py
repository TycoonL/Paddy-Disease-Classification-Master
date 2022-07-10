import random
from os import walk

import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from PIL import Image

from model import CNN
from torch.utils.data import DataLoader
from torchvision import transforms, datasets,models
from torchvision.transforms import Lambda

submission_dir = r'./Paddy Doctor Dataset/test_images/'
dataset_file = './Paddy Doctor Dataset/train.csv'
submission_sample = './Paddy Doctor Dataset/sample_submission.csv'
submission_output = './Paddy Doctor Dataset/submission.csv'

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
model_path='resnet32_model.pth'
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

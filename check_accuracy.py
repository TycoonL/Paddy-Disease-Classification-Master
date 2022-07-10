import torch

# 测试能否使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 评估准确度的函数
def check_accuracy(loader, model):
    # if loader.dataset.train:
    #     print("Checking acc on training data")
    # else:
    #     print("Checking acc on testing data")
    num_correct = 0
    num_samples = 0
    model.eval()  # 将模型调整为 eval 模式
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
# Paddy Disease Classification

题目链接：https://www.kaggle.com/competitions/paddy-disease-classification

## 一、问题描述

大米（Oryza sativa）是世界范围内的主食之一。稻谷是去壳前的粗粮，主要在亚洲国家在热带气候中种植。水稻种植需要持续监督，因为多种疾病和害虫可能会影响水稻作物，导致高达 70% 的产量损失。通常需要专家监督来减轻这些疾病并防止作物损失。由于作物保护专家的可用性有限，人工疾病诊断既繁琐又昂贵。因此，通过利用在各个领域取得可喜成果的基于计算机视觉的技术来自动化疾病识别过程变得越来越重要。

## 二、数据集

提供了一个包含 10,407 个 (75%) 标记图像的训练数据集，涵盖 10 个类别（9 个疾病类别和正常叶片）。此外，我们还为每个图像提供额外的元数据（稻谷品种和年龄）。

将给定测试数据集中的 3,469 个（25%）图像中的每个水稻图像分类为九种疾病类别之一或正常叶子。

**train.csv** - 训练集

- `image_id`- 唯一图像标识符对应于**train_images**目录中的图像文件名 (.jpg)。
- `label`- 水稻病害类型，也是目标类别。有十类，包括正常的叶子。
- `variety`- 水稻品种的名称。
- `age`- 以天为单位的稻谷年龄。

**sample_submission.csv** - 样本提交文件。

**train_images** - 该目录包含 10,407 张训练图像，存储在对应于 10 个目标类的不同子目录下。文件名对应`image_id`于`train.csv`.

**test_images** - 此目录包含 3,469 个测试集图像。

## 三、项目文件结构

```
Paddy-Disease-Classification-Master/
│
├── train.py - main script to start training
├── test.py - evaluation of trained model
│
├── model.py
│  
└── Paddy Doctor Dataset/ - dateset
    ├── test_images
    └── train_images
```

## 四、数据集加载

![img](https://img-blog.csdnimg.cn/20200414110352377.png?)

[pytorch](https://so.csdn.net/so/search?q=pytorch&spm=1001.2101.3001.7020)加载图片数据集有两种方法，根据数据集结构选择了ImageFolder 方法。

1. ImageFolder 适合于分类[数据集](https://so.csdn.net/so/search?q=数据集&spm=1001.2101.3001.7020)，并且每一个类别的图片在同一个文件夹, ImageFolder加载的数据集， 训练数据为文件件下的图片， 训练标签是对应的文件夹， 每个文件夹为一个类别
2. 根据pytorch提供的Dataset类创建自己的数据集加载类。

```

```

## 五、模型

模型A: Conv（in：3，out：8，3×3）→ MaxPool（2×2）→ Conv（in：8，out：16，3×3）→ MaxPool（2×2）→ FC

模型B：Conv（in：3，out：8，3×3）→ MaxPool（2×2）→ Conv（in：8，out：4，3×3）→ MaxPool（2×2）→ FC

模型C：Conv（in：3，out：8，3×3）→ MaxPool（2×2）→Conv（in：8，out：6，3×3）→ MaxPool（2×2）→  Conv（in：6，out：4，3×3）→ FC

模型D：Conv（in：3，out：8，3×3）→ MaxPool（2×2）→Conv（in：8，out：6，3×3）→ MaxPool（2×2）→  Conv（in：6，out：8，3×3）→ FC

## 六、实验结果

64：

![image-20220707221933455](C:\Users\Tycoon\AppData\Roaming\Typora\typora-user-images\image-20220707221933455.png)

64 normalize：

![image-20220707224401101](C:\Users\Tycoon\AppData\Roaming\Typora\typora-user-images\image-20220707224401101.png)

128：

![image-20220707234901429](C:\Users\Tycoon\AppData\Roaming\Typora\typora-user-images\image-20220707234901429.png)

320：

![image-20220708003106270](C:\Users\Tycoon\AppData\Roaming\Typora\typora-user-images\image-20220708003106270.png)



B_128_:

![image-20220708125727929](C:\Users\Tycoon\AppData\Roaming\Typora\typora-user-images\image-20220708125727929.png)

B_128_con5:

![image-20220708133622176](C:\Users\Tycoon\AppData\Roaming\Typora\typora-user-images\image-20220708133622176.png)

C：

![image-20220708140927290](C:\Users\Tycoon\AppData\Roaming\Typora\typora-user-images\image-20220708140927290.png)

D：

![image-20220708164307070](C:\Users\Tycoon\AppData\Roaming\Typora\typora-user-images\image-20220708164307070.png)

D 640:

![image-20220708181123542](C:\Users\Tycoon\AppData\Roaming\Typora\typora-user-images\image-20220708181123542.png)

## 后记

- ```
  用nn.CrossEntropyLoss()作为损失函数，label不需要onehot，不需要softmax层！
  ```

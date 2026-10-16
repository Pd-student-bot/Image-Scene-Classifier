# Image‑Scene‑Classifier

> 
> PyTorch实现的风景 / 美食 / 人像三分类图像训练&推理完整工具
> Complete PyTorch image classification toolkit for landscape, food, portrait.

## 📖 项目简介

本项目是基于 ResNet‑18 的图像分类完整Python代码。
包含**数据集加载、模型训练、测试评估、预测可视化、单张图片推理、批量图片自动分类**全套功能。专门用于区分三类图片：`风景`、`美食`、`人像`。

- 使用ResNet‑18预训练权重，采用迁移学习方式训练；
- 自动检测GPU(CUDA) / CPU运行环境；
- 训练过程输出损失、准确率，支持进度条显示；
- 自动保存训练得到的最优权重 `best_model.pth`；
- 可生成预测结果可视化图片；
- 支持**单张图片预测**、**整个文件夹批量自动分类**，会按照预测类别自动建立子文件夹并拷贝图片。

> 
> This is a full‑featured Python image classification project based on ResNet‑18 with transfer‑learning.
> It covers dataset loading, model training, evaluation, prediction visualization, single‑image inference and batch folder classification for three categories: `landscape`, `food`, `portrait`.
> 
> 
> - Pre‑trained ResNet‑18 backbone for transfer learning
> - Auto detect CUDA GPU or CPU device
> - Print loss & accuracy during training with progress bar
> - Auto‑save best weight as `best_model.pth`
> - Generate prediction visualization figure
> - Support single image prediction and batch classification; images will be copied into category sub‑folders automatically.

> 
> ⚠️ 本项目仅用于技术学习、课程练习，不提供预训练权重文件，需要使用者自行准备数据集进行训练。
> ⚠️ For technical learning & practice only. No pre‑trained weights are provided. You need to prepare your own dataset for training.

## ✨ 功能特性｜Features

- ✅ 自动环境检测，优先使用NVIDIA GPU，无GPU自动降级CPU
- ✅ 数据集加载：使用`ImageFolder`，按子文件夹名称作为分类标签
- ✅ 迁移学习：冻结ResNet18主干，仅训练最后全连接层
- ✅ 训练循环：损失计算、准确率统计、tqdm训练进度条
- ✅ 自动保存最优模型权重到 `saved_models/best_model.pth`
- ✅ 生成预测可视化图保存至 `visualizations/predictions.png`
- ✅ 单张图片预测，输出类别与置信度
- ✅ 批量处理文件夹：自动把图片复制到 `classified_images/类别名/`
- ✅ 支持 jpg / jpeg / png / bmp / gif 常见图片格式

> 
> Features

- ✅ Auto environment detection: use CUDA GPU if available, fallback to CPU
- ✅ Dataset loading via ImageFolder, sub‑folder names act as class labels
- ✅ Transfer‑learning: freeze ResNet‑18 backbone, only train final fully‑connected layer
- ✅ Training loop with loss, accuracy and tqdm progress bar
- ✅ Auto save best checkpoint: `saved_models/best_model.pth`
- ✅ Export prediction visualization: `visualizations/predictions.png`
- ✅ Single‑image inference: output predicted class and confidence score
- ✅ Batch folder processing: copy images into `classified_images/<class_name>/`
- ✅ Support jpg / jpeg / png / bmp / gif image formats

## 🗂 数据集目录结构

你的数据集根目录下，必须存在 `train` 和 `test` 两个子目录，每个内部建立三个类别文件夹：

```
你的数据集总文件夹/
├─ train/
│   ├─ 风景/
│   ├─ 美食/
│   └─ 人像/
└─ test/
    ├─ 风景/
    ├─ 美食/
    └─ 人像/
```

> 
> Dataset folder structure

```
dataset_root/
├─ train/
│   ├─ 风景/
│   ├─ 美食/
│   └─ 人像/
└─ test/
    ├─ 风景/
    ├─ 美食/
    └─ 人像/
```

## 🛠 环境依赖｜Requirements

Python >= 3.9

```
pip install torch torchvision numpy pillow matplotlib tqdm opencv‑python
```

> 
> Environment

```
pip install torch torchvision numpy pillow matplotlib tqdm opencv‑python
```

## 🚀 使用步骤｜Usage

1. 克隆仓库到本地

```
git clone [https://github.com/](https://github.com/)你的用户名/Image‑Scene‑Classifier.git
cd Image‑Scene‑Classifier
```

2. 准备好上面格式的数据集，打开 `图像识别代码v2.PY`
修改代码中两处路径：

```
# 修改为你的数据集总路径
data_dir = r'D:\python练习\应用赛'
# 修改为你的待测试单张图片路径
test_image_path = r'D:\python练习\应用赛\待分类\example.jpg'
```

3. 运行脚本

```
python 图像识别代码v2.PY
```

运行后程序会依次执行：

1. 打印硬件环境信息（GPU/CPU）
2. 加载训练、测试数据集
3. ResNet18迁移学习训练（默认5轮epoch）
4. 保存最优模型到 `saved_models/best_model.pth`
5. 生成预测样例可视化图片 `visualizations/predictions.png`
6. 执行单张图片预测
7. 控制台询问是否批量处理文件夹；输入`y`即可开启批量分类

> 
> Usage Steps

1. Clone repository

```
git clone [https://github.com/YourUserName/Image](https://github.com/YourUserName/Image)‑Scene‑Classifier.git
cd Image‑Scene‑Classifier
```

2. Prepare dataset following folder structure above.
Edit `图像识别代码v2.PY`, change these two paths to match your local environment:

```
data_dir = r'Your‑dataset‑root‑path'
test_image_path = r'path/to/your/test/example.jpg'
```

3. Run script

```
python 图像识别代码v2.PY
```

Program workflow:

1. Print hardware info(GPU / CPU)
2. Load train & test dataset
3. Transfer‑learning training with ResNet‑18(default 5 epochs)
4. Save best checkpoint to `saved_models/best_model.pth`
5. Export prediction visualization `visualizations/predictions.png`
6. Run single‑image prediction
7. Prompt for batch‑folder classification; input `y` to start batch inference.

### 输出文件夹说明

- `saved_models/`：存放训练输出的模型权重 best_model.pth
- `visualizations/`：预测样例可视化图片
- `classified_images/`：批量推理输出，图片按预测类别分到不同子文件夹

> 
> Output directories

- `saved_models/`: trained checkpoint `best_model.pth`
- `visualizations/`: prediction sample plot
- `classified_images/`: batch inference output, images grouped by predicted class.

## ⚠️ 注意事项｜Notes

1. Windows路径使用原始字符串 `r""`，避免转义报错；
2. 数据集子文件夹名称必须是 `风景`、`美食`、`人像`，`ImageFolder`会把文件夹名当做标签；
3. 如果没有NVIDIA显卡，会自动使用CPU训练，速度会很慢；
4. 本仓库**不提供训练数据集与预训练pth权重**，需要使用者自行收集图片训练；
5. 训练轮数epochs在代码 `train_model(..., epochs=5)` 修改。

> 
> Notes

1. Use raw string `r""` for Windows file path to avoid escape‑character error.
2. Sub‑folder names inside train/test must exactly be `风景`, `美食`, `人像`. ImageFolder uses folder names as class labels.
3. Without NVIDIA GPU, training will run on CPU and be very slow.
4. This repository **does not contain dataset or pre‑trained weights**. You need to collect your own images for training.
5. Adjust training epochs in function call `train_model(..., epochs=5)`.

## ⚖️ 免责声明｜Disclaimer

项目仅用于个人学习与技术练习。代码按原样提供，不做任何担保。使用者自行承担运行本代码产生的全部风险。请勿直接用于商业产品。

> 
> Disclaimer
> This project is for personal study and technical practice only. The code is provided as‑is without warranty. End users bear all risks arising from usage. Do not use directly for commercial products.

## 📄 License

MIT License

---

### 配套 `.gitignore` 建议（你可以新建一个`.gitignore`上传到仓库，避免把本地产物上传GitHub）

```
# python缓存
__pycache__/
*.pyc
*.pyo
*.pyd

# 模型权重、输出产物
saved_models/
visualizations/
classified_images/

# 本地数据集、图片
*.jpg
*.jpeg
*.png
*.bmp
*.gif

# IDE
.vscode/
.idea/
*.swp
*.swo
```

#添加库并重命名
import os#添加os模块，使用户能与系统交互
import torch#PyTorch 深度学习框架的核心库，提供了张量计算、自动微分、神经网络构建等功能
import numpy as np#用于科学计算的核心库，提供高效的多维数组对象（ndarray）和处理这些数组的工具
import torch.nn as nn#帮助用户高效搭建和训练深度学习模型
import torch.optim as optim#用于更新模型参数以最小化损失函数
from torchvision import datasets, models, transforms
#datasets用于常用数据集加载，如图片数据
#models用于预训练的深度学习模型
#transforms用于图像变换函数，处理图片
from torch.utils.data import DataLoader
#dataloader批量加载数据
from tqdm import tqdm#进度条库，更直观
import matplotlib.pyplot as plt
#常用绘图库，创建可视化图表
from PIL import Image

def setup_environment():#def:建立函数
    """设置环境并检查GPU"""
    print("===== 环境初始化 =====")
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)#seed:生成随机种子，用于训练
    # 检查GPU可用性
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device:指定cpu为运行的硬件环境
    print(f"运行设备: {device}")
    if device.type == "cuda":#指定运行硬件类型
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024:.0f} MB")
    return device

def load_datasets(data_dir):
    """加载数据集并创建数据加载器"""
    print(f"\n===== 加载数据集: {data_dir} =====")
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(256),#调整图像大小
        transforms.CenterCrop(224),#裁剪中心区域
        transforms.ToTensor(),#转换为tensor数组，用于加速
        transforms.Normalize(#设定数据集的数据
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    # 加载数据集
    try:
        train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform)
        test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform)
    except Exception as e:#exception，报错
        print(f"数据集加载失败: {e}")
        print("请确保目录结构为:")
        print(f"{data_dir}/train/风景")
        print(f"{data_dir}/train/美食")
        print(f"{data_dir}/train/人像")
        print(f"{data_dir}/test/风景")
        print(f"{data_dir}/test/美食")
        print(f"{data_dir}/test/人像")
        exit(1)
    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    # 打印数据集信息
    class_names = train_dataset.classes
    print(f"分类类别: {class_names} ({len(class_names)} 类)")
    print(f"训练集大小: {len(train_dataset)} 张")
    print(f"测试集大小: {len(test_dataset)} 张")
    return train_loader, test_loader, class_names

def create_model(device, num_classes):
    """创建并配置模型"""
    print("\n===== 初始化模型 =====")
    # 加载预训练模型
    model = models.resnet18(pretrained=True)#创建一个模型
    # 冻结大部分参数
    for param in model.parameters():
        param.requires_grad = False
    # 修改最后一层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)#nn.linear，数据向量化
    # 移至设备
    model = model.to(device)
    return model

def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=5):
    """训练模型并返回最佳模型"""
    print(f"\n===== 开始训练 ({epochs} 轮) =====")
    best_acc = 0.0
    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, "best_model.pth")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print('-' * 20)
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        # 使用tqdm显示进度条
        for inputs, labels in tqdm(train_loader, desc="训练中"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f"训练损失: {epoch_loss:.4f}, 准确率: {epoch_acc:.4f}")
        # 测试阶段
        model.eval()
        test_loss = 0.0
        test_corrects = 0
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="测试中"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                test_corrects += torch.sum(preds == labels.data)
        test_loss = test_loss / len(test_loader.dataset)
        test_acc = test_corrects.double() / len(test_loader.dataset)
        print(f"测试损失: {test_loss:.4f}, 准确率: {test_acc:.4f}")
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"模型已保存至: {best_model_path}")
    print(f"\n训练完成! 最佳测试准确率: {best_acc:.4f}")
    return model, best_model_path

def visualize_predictions(model, test_loader, class_names, device, num_images=6):
    """可视化模型预测结果"""
    print("\n===== 生成预测可视化 =====")
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(15, 10))
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                # 构建标题，包含预测类别和置信度
                probs = torch.nn.functional.softmax(outputs[j], dim=0)
                conf = probs[preds[j]].item() * 100
                ax.set_title(f'预测: {class_names[preds[j]]} ({conf:.1f}%)\n真实: {class_names[labels[j]]}')
                # 反归一化并显示图像
                img = inputs[j].cpu().numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                plt.imshow(img)
                if images_so_far == num_images:
                    viz_dir = "visualizations"
                    os.makedirs(viz_dir, exist_ok=True)
                    viz_path = os.path.join(viz_dir, "predictions.png")
                    plt.tight_layout()
                    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
                    print(f"可视化结果已保存至: {viz_path}")
                    return

def predict_image(image_path, model, class_names, device, transform, output_dir='classified_images'):
    """预测单张图片并保存结果"""
    model.eval()
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"无法打开图片: {image_path}, 错误: {e}")
        return None
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
    predicted_class = class_names[pred.item()]
    # 创建分类文件夹
    class_folder = os.path.join(output_dir, predicted_class)
    os.makedirs(class_folder, exist_ok=True)
    # 保存图片
    filename = os.path.basename(image_path)
    output_path = os.path.join(class_folder, filename)
    img.save(output_path)
    return {
        '预测类别': predicted_class,
        '置信度': f'{conf.item():.4f}',
        '保存路径': output_path
    }

def batch_process_images(input_dir, model, class_names, device, transform, output_dir='classified_images'):
    """批量处理文件夹中的所有图片"""
    if not os.path.exists(input_dir):
        print(f"输入文件夹不存在: {input_dir}")
        return
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    processed_count = 0
    print(f"\n===== 开始批量处理文件夹: {input_dir} =====")
    for filename in os.listdir(input_dir):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in image_extensions:
            image_path = os.path.join(input_dir, filename)
            result = predict_image(image_path, model, class_names, device, transform, output_dir)
            if result:
                print(f"已处理: {filename} -> 预测类别: {result['预测类别']}, 置信度: {result['置信度']}")
                processed_count += 1
    print(f"批量处理完成! 共处理 {processed_count} 张图片")
    print(f"分类结果保存在: {os.path.abspath(output_dir)}")

def main():
    # 设置环境
    device = setup_environment()
    # 数据集路径
    data_dir = r'D:\python练习\应用赛'#将此文件夹地址改为训练，测试，实践，实践结果四个文件所在的总文件地址
    # 加载数据集
    train_loader, test_loader, class_names = load_datasets(data_dir)
    # 创建模型
    model = create_model(device, len(class_names))
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    # 训练模型
    model, best_model_path = train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=5)
    # 可视化预测结果
    visualize_predictions(model, test_loader, class_names, device)
    # 数据预处理（与训练时相同）
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    # 示例：单张图片预测
    test_image_path = r'D:\python练习\应用赛\待分类\example.jpg'  # 替换为你的实践图片路径
    if os.path.exists(test_image_path):
        result = predict_image(test_image_path, model, class_names, device, transform)
        if result:
            print(f"\n单张图片预测结果: {result}")
    else:
        print(f"\n测试图片不存在: {test_image_path}")
    # 示例：批量处理整个文件夹
    batch_process_option = input("\n是否要批量处理文件夹中的图片? (y/n): ").lower().strip()
    if batch_process_option == 'y':
        input_dir = input("请输入要处理的文件夹路径: ")
        if os.path.exists(input_dir):
            batch_process_images(
                input_dir=input_dir,  # 使用用户输入的路径
                model=model,
                class_names=class_names,
                device=device,          # 补充缺失的device参数
                transform=transform,    # 补充缺失的transform参数
                output_dir=r'D:\python练习\应用赛\分类好的照片'  # 使用原始字符串避免转义问题
            )
        else:
            print(f"文件夹不存在: {input_dir}")

if __name__ == "__main__":#仅运行本文件程序里面编程的代码
    main()
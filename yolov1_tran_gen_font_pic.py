import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random,time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.optim.lr_scheduler import ReduceLROnPlateau


def resize_image(img,scale):
    width, height = img.size
    width = int(width * scale)
    height = int(height * scale)
    resized_image = img.resize((width, height), Image.LANCZOS)
    return resized_image

# 加载背景图
def load_background_image(background_path, scale=0.5):
    background = Image.open(background_path)
    background  = resize_image(background,scale)
    return background

# 加载数字单图
def load_digit_images(digit_folder, scale=0.5):
    digit_images = []

    all_digit_image = Image.open(digit_folder)

    w = int(all_digit_image.size[0]/5)+2
    h = int(all_digit_image.size[0]/5)
    for row in range(0,3):
        for col in range(0,4):
            left = col * w
            top = row * h
            right = left + w
            bottom = top + h

            digit_image  = all_digit_image.crop((left, top, right, bottom))
            digit_image  = resize_image(digit_image,scale)
            digit_images.append(digit_image)

    digit_images.pop(0)
    digit_images.pop()
    return digit_images

background_path = "./background.png"  # 替换为实际背景图路径
digit_folder = "./hurtFont_0.png"  # 替换为实际数字集合图片路径

g_background = load_background_image(background_path, 0.2)
g_digit_images = load_digit_images(digit_folder, 0.2)

# 在背景图上写数字
def write_numbers_on_background( num_strings=10):
    global g_background, g_digit_images
    background = g_background.copy()

    width, height = background.size
    digit_info_list = []  # 用于存储每个数字的信息，格式为 (num, x1, y1, x2, y2)

    used_areas = []  # 用于存储已经使用的区域，避免数字重叠

    max_scale = 1.6
    min_scale = 0.8

    max_digit_width =  5 * int(g_digit_images[0].width * max_scale)
    max_digit_height = int(g_digit_images[0].height * max_scale)

    ran_num = random.randint(1, num_strings+1)
    for _ in range(ran_num):
        # 随机生成1 - 5个数字
        num = random.randint(1, 10000)
        number = list(str(num))

        # 随机生成数字大小（缩放比例）
        scale = random.uniform(min_scale, max_scale)
        while True:
            norm_x = random.uniform(0, 0.85)  # 确保有足够的空间放置数字
            norm_y = random.uniform(0, 0.95)
            x = int(norm_x * width)
            y = int(norm_y * height)
            
            # 检查是否重叠
            overlap = False
            x2 = x + max_digit_width
            y2 = y + max_digit_height
            for area in used_areas:
                area_x1, area_y1, area_x2, area_y2 = area

                if not (x2 < area_x1 or x > area_x2 or y2 < area_y1 or y > area_y2):
                    overlap = True
                    break

            if not overlap:
                break

        xo = x 
        for digit in number:
            digit_img = g_digit_images[int(digit)]
            new_width = int(digit_img.width * scale)
            new_height = int(digit_img.height * scale)
            resized_digit = digit_img.resize((new_width, new_height), Image.LANCZOS)
            background.paste(resized_digit, (xo, y), resized_digit)            
               
            #digit_info_list.append((int(digit), xo/width, y/height, (xo+new_width)/width, (y+new_height)/height)) #单个数字输出
            xo += new_width

        digit_info_list.append((num, x/width, y/height, xo/width, (y+new_height)/height)) #多个数字输出
        used_areas.append((x, y, x + max_digit_width, y + max_digit_height))

    #由于卷积是有序的，所以需要对生成的数字进行排序，不然训练的时候卷积对应的特征位置没办法和标签的位置对应上，就会让学习没办法达到效果
    sorted_digit_info_list = sorted(digit_info_list, key=lambda item: (item[2], item[1]))  
    return background,sorted_digit_info_list




def show_numbers_on_background(img,digit_info_list):
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()  # 使用默认字体

    for digit_info in digit_info_list:
        digit,x1,y1,x2,y2 = digit_info

        x1 = int(x1*img.size[0])
        y1 = int(y1*img.size[1])
        x2 = int(x2*img.size[0])  
        y2 = int(y2*img.size[1])

        draw.text((x2, y2), f"({digit})", fill=(255, 0, 0), font=font)

        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)

    return img


# 主函数
def test_data():
    for i in range(10):
        begin_time = time.time()
        img,digit_info_list = write_numbers_on_background()
        end_time = time.time()
        print(f"Time taken to generate image: {end_time - begin_time:.4f} seconds")

    
    show_img = show_numbers_on_background(img,digit_info_list)
    show_img.show()

    wait = input("Press Enter to continue.")


def iou(bbox_pre, bbox_true, grid_i, grid_j, img_size=448, grid_num=7):
    """
    :param bbox_pre: Tensor, shape:[4], 4:bbox(cx, cy, w, h)
    :param bbox_true: Tensor, shape:[4], 4:bbox(cx, cy, w, h)
    :param grid_i: The row index (or y index) of the grid
    :param grid_j: The col index (or x index) of the grid
    :param img_size: The input image's shape of the model is (img_size, img_size, 3)
    :param grid_num: In the paper, this num is 7
    :return: Tensor, shape:[batch_size], iou
    """
    # 栅格大小
    grid_size = img_size / grid_num

    # 将归一化的预测值转化为实际的预测值
    cx_pre = bbox_pre[0] * grid_size + grid_j * grid_size
    cy_pre = bbox_pre[1] * grid_size + grid_i * grid_size
    w_pre = bbox_pre[2] * img_size
    h_pre = bbox_pre[3] * img_size

    # 将预测值(cx, cy, w, h)转成(x_min, y_min, x_max, y_max)
    x_min_pre = cx_pre - w_pre / 2
    y_min_pre = cy_pre - h_pre / 2
    x_max_pre = cx_pre + w_pre / 2
    y_max_pre = cy_pre + h_pre / 2

    # 将归一化的标定值转化为实际的标定值
    cx_true = bbox_true[0] * grid_size + grid_j * grid_size
    cy_true = bbox_true[1] * grid_size + grid_i * grid_size
    w_true = bbox_true[2] * img_size
    h_true = bbox_true[3] * img_size

    # 标定值(x_min, y_min, x_max, y_max)
    x_min_true = cx_true - w_true / 2
    y_min_true = cy_true - h_true / 2
    x_max_true = cx_true + w_true / 2
    y_max_true = cy_true + w_true / 2

    # 相交区域矩形(x_min, y_min, x_max, y_max)
    union_x_min = max(x_min_pre, x_min_true)
    union_x_max = min(x_max_pre, x_max_true)
    union_y_min = max(y_min_pre, y_min_true)
    union_y_max = min(y_max_pre, y_max_true)

    # 无相交区域，交并比为0
    if union_x_min >= union_x_max or union_y_min >= union_y_max:
        return 0

    # 相交区域矩形面积，预测区域矩形面积，标定区域矩形面积
    area_union = (union_x_max - union_x_min) * (union_y_max - union_y_min)
    area_pre = (x_max_pre - x_min_pre) * (y_max_pre - y_min_pre)
    area_true = (x_max_true - x_min_true) * (y_max_true - y_min_true)

    # 计算交并比
    eps = 1e-12 # 避免除零错误
    res = area_union / (area_pre + area_true - area_union + eps)
    return res


# 定义简化后的 YOLOv1 损失函数类，继承自 nn.Module
class SimplifiedYOLOv1Loss(nn.Module):
    def __init__(self, device, grid_num=7, img_size=448, lambda_coord=5, lambda_noobj=0.1):
        # 调用父类的构造函数
        super(SimplifiedYOLOv1Loss, self).__init__()
        # 设备，用于指定计算是在 CPU 还是 GPU 上进行
        self.device = device
        # 网格数量，YOLOv1 将图像划分为 SxS 的网格，这里 S 即为 grid_num
        self.grid_num = grid_num
        # 输入图像的大小
        self.img_size = img_size
        # 坐标损失的权重，用于平衡坐标损失在总损失中的比重
        self.lambda_coord = lambda_coord
        # 无物体的置信度损失的权重，用于平衡无物体时置信度损失在总损失中的比重
        self.lambda_noobj = lambda_noobj

    def forward(self, y_pre, y_true):
        """
        :param y_pre: Tensor, [batch_size, S, S, 10], 10: ((cx, cy, w, h, c)*2)
                      模型的预测输出，每个网格有两个边界框，每个边界框包含中心坐标 (cx, cy)、宽高 (w, h)、置信度 (c)
        :param y_true:  Tensor, [batch_size, S, S, 5],
                        5: (cx, cy, w, h, has_obj)
                        真实标签，包含是否有物体的标志、中心坐标、宽高
        :return: Scaler, loss
                 标量，总损失
        """
        # 一个极小的数，用于避免开方时出现除零错误
        eps = 1e-12
        # 初始化置信度损失，将其放在指定设备上
        loss_confidence = torch.tensor([0], dtype=torch.float32).to(self.device)
        # 初始化坐标损失，将其放在指定设备上
        loss_coordinate = torch.tensor([0], dtype=torch.float32).to(self.device)
        # 初始化尺度损失，将其放在指定设备上
        loss_scale = torch.tensor([0], dtype=torch.float32).to(self.device)
        # 获取当前批次的样本数量
        batch_size = y_pre.shape[0]

        # 遍历批次中的每个样本
        for bid in range(batch_size):
            # 遍历网格的行
            for grid_i in range(self.grid_num):
                # 遍历网格的列
                for grid_j in range(self.grid_num):
                    # 如果该网格中有物体
                    if y_true[bid, grid_i, grid_j, 4] == 1.0:
                        # 选择 IOU 较大的预测框
                        confidence_true = []
                        # 计算第一个预测框与真实框的 IOU
                        confidence_true.append(iou(
                            y_pre[bid, grid_i, grid_j, 0:4],
                            y_true[bid, grid_i, grid_j, 0:4],
                            grid_i=grid_i,
                            grid_j=grid_j,
                            img_size=self.img_size,
                            grid_num=self.grid_num
                        ))
                        # 计算第二个预测框与真实框的 IOU
                        confidence_true.append(iou(
                            y_pre[bid, grid_i, grid_j, 5:9],
                            y_true[bid, grid_i, grid_j, 0:4],
                            grid_i=grid_i,
                            grid_j=grid_j,
                            img_size=self.img_size,
                            grid_num=self.grid_num
                        ))
                        # 选择 IOU 较大的预测框
                        if confidence_true[0] > confidence_true[1]:
                            choose_bbox = 0
                        else:
                            choose_bbox = 1

                        # 计算置信度损失
                        # 获取选中预测框的置信度预测值
                        confidence_pre = y_pre[bid, grid_i, grid_j, 5 * choose_bbox + 4]
                        # 计算预测置信度与真实 IOU 的均方误差，并累加到置信度损失中
                        loss_confidence += torch.pow(confidence_pre - confidence_true[choose_bbox], 2)

                        # 计算中心坐标损失
                        # 计算选中预测框的中心坐标与真实中心坐标的均方误差，并乘以坐标损失权重，累加到坐标损失中
                        loss_coordinate += self.lambda_coord * torch.pow(
                            y_pre[bid, grid_i, grid_j, (5 * choose_bbox):(5 * choose_bbox + 2)] -
                            y_true[bid, grid_i, grid_j, 0:2], 2
                        ).sum()

                        # 计算尺度损失
                        # 计算选中预测框的宽高平方根与真实宽高平方根的均方误差，并乘以坐标损失权重，累加到尺度损失中
                        # 确保输入值非负
                        pre_w_h = torch.clamp(y_pre[bid, grid_i, grid_j, (5 * choose_bbox + 2):(5 * choose_bbox + 4)], min=eps)
                        true_w_h = torch.clamp(y_true[bid, grid_i, grid_j, 2:4], min=eps)
                        loss_scale += self.lambda_coord * torch.pow(
                            torch.sqrt(pre_w_h) -
                            torch.sqrt(true_w_h), 2
                        ).sum()
                    else:
                        # 该网格没有物体
                        # 真实置信度为 0
                        confidence_true = 0
                        # 第一个预测框
                        # 获取第一个预测框的置信度预测值
                        confidence_pre = y_pre[bid, grid_i, grid_j, 4]
                        # 计算预测置信度与真实置信度的均方误差，并乘以无物体置信度损失权重，累加到置信度损失中
                        loss_confidence += self.lambda_noobj * torch.pow(confidence_pre - confidence_true, 2)
                        # 第二个预测框
                        # 获取第二个预测框的置信度预测值
                        confidence_pre = y_pre[bid, grid_i, grid_j, 9]
                        # 计算预测置信度与真实置信度的均方误差，并乘以无物体置信度损失权重，累加到置信度损失中
                        loss_confidence += self.lambda_noobj * torch.pow(confidence_pre - confidence_true, 2)

        # 对每个损失进行归一化，除以批次大小
        loss_confidence /= batch_size
        loss_coordinate /= batch_size
        loss_scale /= batch_size
        # 计算总损失
        loss = loss_confidence + loss_coordinate + loss_scale

        return loss


def test_loss():
    y_pre = torch.zeros((5, 7, 7, 10))
    y_true = torch.zeros((5, 7, 7, 5))
    loss_func = SimplifiedYOLOv1Loss(device=torch.device("cpu"))
    loss = loss_func(y_pre, y_true)
    print(loss.item())


# 自定义数据集类
class NumberDataset(Dataset):
    # 类的初始化方法，在创建数据集对象时调用
    def __init__(self, num_samples):
        # 用于存储生成的样本，每个样本是一个元组，包含图像和对应的目标框编码信息
        self.samples = []
        # 循环生成指定数量的样本
        for _ in range(num_samples):
            # 调用 write_numbers_on_background 函数生成一张图像和对应的边界框信息
            # img 是 PIL 图像对象，boxes 是包含边界框信息的列表
            img, boxes = write_numbers_on_background()
            # 将图像的大小调整为 448x448 像素，以适应模型输入要求
            img = img.resize((448, 448))
            # 将 PIL 图像对象转换为 NumPy 数组，并将像素值归一化到 [0, 1] 范围
            img = np.array(img) / 255.0
            # 将 NumPy 数组转换为 PyTorch 张量，并调整维度顺序为 (C, H, W)，其中 C 是通道数，H 是高度，W 是宽度
            # 最后将数据类型转换为 float
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            # 调用 encode_boxes 方法对边界框信息进行编码，将其转换为模型可接受的目标张量
            target = self.encode_boxes(boxes)
            # 将处理后的图像和目标张量作为一个样本添加到 samples 列表中
            self.samples.append((img, target))

    # 对边界框信息进行编码的方法，将边界框信息转换为 YOLOv1 模型所需的目标张量
    def encode_boxes(self, boxes):
        # 定义网格的数量，这里将图像划分为 7x7 的网格
        S = 7
        # 每个网格预测的边界框数量，这里每个网格预测 2 个边界框
        B = 2
        # 类别数量，这里假设只有一个类别
        C = 1
        # 初始化目标张量，形状为 (S, S, 5 * B + C)
        # 每个网格的输出包含 5 * B 个边界框相关信息（每个边界框 5 个值：中心坐标 x、y，宽 w，高 h，置信度）和 C 个类别概率
        target = torch.zeros((S, S, 5))
        # 遍历每个边界框信息
        for number, x1, y1, x2, y2 in boxes:
            # 计算边界框的中心点 x 坐标
            cx = (x1 + x2) / 2
            # 计算边界框的中心点 y 坐标
            cy = (y1 + y2) / 2
            # 计算边界框的宽度
            w = x2 - x1
            # 计算边界框的高度
            h = y2 - y1
            # 确定边界框中心点所在的网格的 x 索引
            grid_x = int(cx * S)
            # 确定边界框中心点所在的网格的 y 索引
            grid_y = int(cy * S)
            # 计算边界框中心点相对于所在网格左上角的偏移量（x 方向）
            cx_cell = cx * S - grid_x
            # 计算边界框中心点相对于所在网格左上角的偏移量（y 方向）
            cy_cell = cy * S - grid_y
            # 如果当前网格的第一个边界框的置信度为 0，表示该网格还没有分配边界框
            if target[grid_y, grid_x, 4] == 0:
                # 将边界框的中心点偏移量、宽度和高度赋值给目标张量中第一个边界框的对应位置
                target[grid_y, grid_x, 0:4] = torch.tensor([cx_cell, cy_cell, w, h])
                # 将第一个边界框的置信度设置为 1，表示该边界框包含目标
                target[grid_y, grid_x, 4] = 1
        # 返回编码后的目标张量
        return target

    # 返回数据集的样本数量，是 Dataset 类的必要方法
    def __len__(self):
        return len(self.samples)

    # 根据索引获取数据集中的一个样本，是 Dataset 类的必要方法
    def __getitem__(self, idx):
        return self.samples[idx]


# YOLOv1 模型
class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=1):
        super(YOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.conv_layers = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),  # 添加批量归一化
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            
            # 第二层卷积
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),  # 添加批量归一化
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            

            # 第三层卷积
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),  # 添加批量归一化
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第四层卷积
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),  # 添加批量归一化
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 14 * 14, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, S * S * (5 * B + C))
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        x = x.view(-1, self.S, self.S, 5 * self.B + self.C)
        return x

# 训练函数
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_loss = float('inf')
    counter = 0  # 用于记录连续没有改进的轮次
    patience = 10  # 当连续 patience 轮次没有改进时，停止训练
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3,min_lr=1e-8)


    model.to(device)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss},LR: {optimizer.param_groups[0]["lr"]}')
        scheduler.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                model_path = f'yolov1_epoch_{epoch + 1}_{round(epoch_loss,6)}.pth'
                torch.save(model.state_dict(), model_path)
                print(f'Early stopping after {epoch + 1} epochs.')
                break

# 测试函数
def test_model(model, img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    img = img.resize((448, 448))
    img = np.array(img) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)

    show_output(img,output)

def show_output(img,output):
    S = 7
    B = 2
    C = 1
    output = output.cpu().squeeze(0)
    fig, ax = plt.subplots(1)
    ax.imshow(img.cpu().squeeze(0).permute(1, 2, 0))
    for i in range(S):
        for j in range(S):
            if output[i, j].shape[0] == 5:
                B = 1
            for b in range(B):
                conf = output[i, j, 4 + b * 5].item()
                if conf > 0.05:
                    cx_cell = output[i, j, b * 5 + 0].item()
                    cy_cell = output[i, j, b * 5 + 1].item()
                    w = output[i, j, b * 5 + 2].item()
                    h = output[i, j, b * 5 + 3].item()
                    cx = (j + cx_cell) / S
                    cy = (i + cy_cell) / S
                    x1 = (cx - w / 2) * 448
                    y1 = (cy - h / 2) * 448
                    x2 = (cx + w / 2) * 448
                    y2 = (cy + h / 2) * 448
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=conf * 5, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
    plt.show()

# 主函数
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    '''
    bbox_pre = torch.tensor([0.5, 0.5, 0.2, 0.2])
    bbox_true = torch.tensor([0.7, 0.7, 0.4, 0.8])
    grid_i = 3
    grid_j = 3
    iou_value = iou(bbox_pre, bbox_true, grid_i, grid_j)
    print(f"IoU: {iou_value.item()}")
    exit()
    '''
    #test_loss()
    
    # 数据集和数据加载器
    dataset = NumberDataset(num_samples=300)
    dataloader = DataLoader(dataset, batch_size=30, shuffle=True)

    #show_output(dataset[0][0],dataset[0][1])
    #exit()

    # 模型、损失函数和优化器
    model = YOLOv1()
    criterion = SimplifiedYOLOv1Loss(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(model, dataloader, criterion, optimizer, num_epochs=100)

    # 测试模型
    test_img, _ = write_numbers_on_background()
    test_model(model, test_img)
    input("Press Enter to continue.")
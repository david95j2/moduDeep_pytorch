import numpy as np
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm
from torchvision import transforms as transforms
from torchsummary import summary
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(50)
if device == 'cuda':
    torch.cuda.manual_seed_all(50)

# train_ds = torchvision.datasets.ImageFolder(
#     root="pytorch/Lab_05_(inception)/baby_data/train", transform=transforms.ToTensor())
# val_ds = torchvision.datasets.ImageFolder(
#     root="pytorch/Lab_05_(inception)/baby_data/val", transform=transforms.ToTensor())
# test_ds = torchvision.datasets.ImageFolder(
#     root="pytorch/Lab_05_(inception)/baby_data/test", transform=transforms.ToTensor())


# def get_Mean(data):
#     RGB = [np.mean(x.numpy(), axis=(1, 2)) for x, _ in data]
#     mean_R = np.mean([m[0] for m in RGB])
#     mean_G = np.mean([m[1] for m in RGB])
#     mean_B = np.mean([m[2] for m in RGB])

#     return mean_R, mean_G, mean_B


# def get_Std(data):
#     RGB = [np.std(x.numpy(), axis=(1, 2)) for x, _ in data]
#     std_R = np.mean([s[0] for s in RGB])
#     std_G = np.mean([s[1] for s in RGB])
#     std_B = np.mean([s[2] for s in RGB])

#     return std_R, std_G, std_B


# # RGB 평균 구하기
# train_meanR, train_meanG, train_meanB = get_Mean(train_ds)
# val_meanR, val_meanG, val_meanB = get_Mean(val_ds)
# test_meanR, test_meanG, test_meanB = get_Mean(test_ds)

# # RGB 표준편차 구하기
# train_stdR, train_stdG, train_stdB = get_Std(train_ds)
# val_stdR, val_stdG, val_stdB = get_Std(val_ds)
# test_stdR, test_stdG, test_stdB = get_Std(test_ds)

# print("train Mean : ", train_meanR, train_meanG, train_meanB)
# print("train Std : ", train_stdR, train_stdG, train_stdB)


# def get_trans(width, height, m_r, m_g, m_b, s_r, s_g, s_b):
#     transformation = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Resize((width, height)),
#         transforms.Normalize([m_r, m_g, m_b], [s_r, s_g, s_b]),
#     ])
#     return transformation


# train_transformation = get_trans(224, 224, train_meanR, train_meanG, train_meanB,
#                                  train_stdR, train_stdG, train_stdB)

# val_transformation = get_trans(224, 224, val_meanR, val_meanG, val_meanB,
#                                val_stdR, val_stdG, val_stdB)

# test_transformation = get_trans(224, 224, test_meanR, test_meanG, test_meanB,
#                                 test_stdR, test_stdG, test_stdB)

# train_ds.transform = train_transformation
# val_ds.transform = val_transformation
# test_ds.transform = test_transformation

# train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
# val_dl = DataLoader(val_ds, batch_size=32, shuffle=True)
# test_dl = DataLoader(test_ds, batch_size=32, shuffle=True)

# classes = ('bgs', 'crying')


# # display sample images
# def show(img, y=None, color=True):
#     npimg = img.numpy()
#     npimg_tr = np.transpose(npimg, (1, 2, 0))
#     plt.imshow(npimg_tr)

#     if y is not None:
#         plt.title('labels: ' + str(y))


# np.random.seed(0)
# torch.manual_seed(0)

# grid_size = 1
# rnd_inds = np.random.randint(0, len(train_ds), grid_size)

# x_grid = [train_ds[i][0] for i in rnd_inds]
# y_grid = [train_ds[i][1] for i in rnd_inds]

# x_grid = torchvision.utils.make_grid(x_grid, nrow=5, padding=0)

# # call helper function
# plt.figure(figsize=(10, 10))
# show(x_grid, y_grid)

# print(x_grid.shape)


class BasicBlock(nn.Module):
    def __init__(self, input_ch, output_ch, stride=1):
        super().__init__()

        self.basick_layer = nn.Sequential(
            nn.Conv2d(input_ch, output_ch, kernel_size=3,
                      stride=stride, padding=1),
            nn.BatchNorm2d(output_ch),
            nn.ReLU(),
            nn.Conv2d(output_ch, output_ch,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_ch),
        )

        self.shortcut = nn.Sequential()

        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_ch, output_ch, kernel_size=1, stride=2),
                nn.BatchNorm2d(output_ch)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.basick_layer(x) + self.shortcut(x)
        x = self.relu(x)
        return x


class Resnet34(nn.Module):
    def __init__(self, block, num_classes=2, init_weight=True):
        super().__init__()
        self.in_channel = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x = self.make_block(
            block, out_channel=64, num_block=3, stride=1)
        self.conv3_x = self.make_block(
            block, out_channel=128, num_block=4, stride=2)
        self.conv4_x = self.make_block(
            block, out_channel=256, num_block=6, stride=2)
        self.conv5_x = self.make_block(
            block, out_channel=512, num_block=3, stride=2)
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

        if init_weight:
            self.weights_initialize()

    def make_block(self, block, out_channel, num_block, stride):
        stride_list = [stride] + [1] * (num_block - 1)
        layer = []
        for i in stride_list:
            layer.append(block(self.in_channel, out_channel, i))
            self.in_channel = out_channel
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avgpool1(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def weights_initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


model = Resnet34(BasicBlock).to(device)
print(model)

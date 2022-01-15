# import package

# model
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchsummary import summary
from torch import optim
from torch.optim.lr_scheduler import StepLR


# dataset and transformation
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# display images
from torchvision import utils
import matplotlib.pyplot as plt

# utils
import numpy as np
from torchsummary import summary
import time
import copy


# load dataset
train_ds = torchvision.datasets.ImageFolder(
    root="pytorch/Lab_05_(inception)/baby_data/train", transform=transforms.ToTensor())
val_ds = torchvision.datasets.ImageFolder(
    root="pytorch/Lab_05_(inception)/baby_data/val", transform=transforms.ToTensor())

print(len(train_ds))
print(len(val_ds))


# To normalize the dataset, calculate the mean and std
train_meanRGB = [np.mean(x.numpy(), axis=(1, 2)) for x, _ in train_ds]
train_stdRGB = [np.std(x.numpy(), axis=(1, 2)) for x, _ in train_ds]

train_meanR = np.mean([m[0] for m in train_meanRGB])
train_meanG = np.mean([m[1] for m in train_meanRGB])
train_meanB = np.mean([m[2] for m in train_meanRGB])
train_stdR = np.mean([s[0] for s in train_stdRGB])
train_stdG = np.mean([s[1] for s in train_stdRGB])
train_stdB = np.mean([s[2] for s in train_stdRGB])


val_meanRGB = [np.mean(x.numpy(), axis=(1, 2)) for x, _ in val_ds]
val_stdRGB = [np.std(x.numpy(), axis=(1, 2)) for x, _ in val_ds]

val_meanR = np.mean([m[0] for m in val_meanRGB])
val_meanG = np.mean([m[1] for m in val_meanRGB])
val_meanB = np.mean([m[2] for m in val_meanRGB])

val_stdR = np.mean([s[0] for s in val_stdRGB])
val_stdG = np.mean([s[1] for s in val_stdRGB])
val_stdB = np.mean([s[2] for s in val_stdRGB])

print(train_meanR, train_meanG, train_meanB)
print(val_meanR, val_meanG, val_meanB)


# define the image transformation
train_transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(228),
    transforms.Normalize([train_meanR, train_meanG, train_meanB], [
                         train_stdR, train_stdG, train_stdB]),
    transforms.RandomHorizontalFlip(),
])

val_transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(228),
    transforms.Normalize([train_meanR, train_meanG, train_meanB], [
                         train_stdR, train_stdG, train_stdB]),
])

# apply transforamtion
train_ds.transform = train_transformation
val_ds.transform = val_transformation

# create DataLoader
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=True)


# display sample images
def show(img, y=None, color=True):
    npimg = img.numpy()
    npimg_tr = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg_tr)

    if y is not None:
        plt.title('labels: ' + str(y))


np.random.seed(0)
torch.manual_seed(0)

grid_size = 4
rnd_inds = np.random.randint(0, len(train_ds), grid_size)
print("image indices:", rnd_inds)

x_grid = [train_ds[i][0] for i in rnd_inds]
y_grid = [train_ds[i][1] for i in rnd_inds]

x_grid = utils.make_grid(x_grid, nrow=4, padding=2)
print(x_grid.shape)

# call helper function
plt.figure(figsize=(10, 10))
show(x_grid, y_grid)


class GoogLeNet(nn.Module):
    def __init__(self, aux_logits=True, num_classes=10, init_weights=True):
        super(GoogLeNet, self).__init__()
        assert aux_logits == True or aux_logits == False
        self.aux_logits = aux_logits

        # conv_block takes in_channels, out_channels, kernel_size, stride, padding
        # Inception block takes out1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool

        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, 2, 1)
        self.conv2 = conv_block(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, 2, 1)
        self.inception3a = Inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, 2, 1)
        self.inception4a = Inception_block(480, 192, 96, 208, 16, 48, 64)

        # auxiliary classifier

        self.inception4b = Inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_block(512, 112, 144, 288, 32, 64, 64)

        # auxiliary classifier

        self.inception4e = Inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, 2, 1)
        self.inception5a = Inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_block(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(7, 1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, num_classes)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        else:
            self.aux1 = self.aux2 = None

        # weight initialization
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        print(f"처음 x.shape : {x.shape}\n")
        x = self.conv1(x)
        print(f"ConV1 x.shape : {x.shape}\n")
        x = self.maxpool1(x)
        print(f"MaxP1 x.shape : {x.shape}\n")
        x = self.conv2(x)
        print(f"ConV2 x.shape : {x.shape}\n")
        x = self.maxpool2(x)
        print(f"MaxP2 x.shape : {x.shape}\n")
        x = self.inception3a(x)
        print(f"Incep3A x.shape : {x.shape}\n")
        x = self.inception3b(x)
        print(f"Incep3B x.shape : {x.shape}\n")
        x = self.maxpool3(x)
        print(f"MaxP3 x.shape : {x.shape}\n")
        x = self.inception4a(x)
        print(f"Incep4A x.shape : {x.shape}\n")

        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        print(f"Incep4B x.shape : {x.shape}\n")
        x = self.inception4c(x)
        print(f"IncepC x.shape : {x.shape}\n")
        x = self.inception4d(x)
        print(f"IncepD x.shape : {x.shape}\n")

        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        print(f"Incep4E x.shape : {x.shape}\n")
        x = self.maxpool4(x)
        print(f"Maxp4 x.shape : {x.shape}\n")
        x = self.inception5a(x)
        print(f"Incep5A x.shape : {x.shape}\n")
        x = self.inception5b(x)
        print(f"Incep5B x.shape : {x.shape}\n")
        x = self.avgpool(x)
        print(f"AvgP x.shape : {x.shape}\n")

        x = x.view(x.shape[0], -1)
        print(f"View x.shape : {x.shape}\n")

        x = self.dropout(x)
        print(f"Drop x.shape : {x.shape}\n")
        x = self.fc1(x)
        print(f"Fc1 x.shape : {x.shape}\n")

        if self.aux_logits and self.training:
            print(f"X/AUX1/AUX2 x.shape : {x.shape} and {aux1} and {aux2}\n")
            return x, aux1, aux2

        else:
            print(f"x x.shape : {x.shape}\n")
            return x

    # define weight initialization function
    def _initialize_weights(self):
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


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv_layer(x)


class Inception_block(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(Inception_block, self).__init__()

        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=1),
            conv_block(red_3x3, out_3x3, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=1),
            conv_block(red_5x5, out_5x5, kernel_size=5, padding=2),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, out_1x1pool, kernel_size=1)
        )

    def forward(self, x):
        # 0차원은 batch이므로 1차원인 filter 수를 기준으로 각 branch의 출력값을 묶어줍니다.
        x = torch.cat([self.branch1(x), self.branch2(
            x), self.branch3(x), self.branch4(x)], 1)
        return x

# auxiliary classifier의 loss는 0.3이 곱해지고, 최종 loss에 추가합니다. 정규화 효과가 있습니다.


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()

        self.conv = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            conv_block(in_channels, 128, kernel_size=1),
        )

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = GoogLeNet(aux_logits=True, num_classes=10,
                  init_weights=True).to(device)
print(model)


x = torch.randn(3, 3, 224, 224).to(device)
output = model(x)
print(output)

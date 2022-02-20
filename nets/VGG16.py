# -*- coding = utf-8 -*-
# @Time : 2022/1/8 15:41
# @Author : 戎昱
# @File : VGG16.py
# @Software : PyCharm
# @Contact : sekirorong@gmail.com
# @github : https://github.com/SekiroRong
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


#-------------------------------------------------#
#   MISH激活函数
#-------------------------------------------------#
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

#---------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   Conv2d + BatchNormalization + Mish
#---------------------------------------------------#
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.layer1 = BasicConv(3, 32, 3)
        self.layer2 = BasicConv(32, 64, 3)
        self.layer3 = BasicConv(64, 128, 3)
        self.layer4 = BasicConv(128, 256, 3)

        self.maxpool = nn.MaxPool2d(2)

    def forward(self,x):
        x = self.maxpool(self.layer1(x))
        x = self.maxpool(self.layer2(x))
        x = self.maxpool(self.layer3(x))
        x = self.layer4(x)

        return x

# import torch
# from torchsummary import summary
#
# if __name__ == "__main__":
#     # 需要使用device来指定网络在GPU还是CPU运行
#     device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     m       = VGG16().to(device)
#     summary(m, input_size=(1, 640, 480))
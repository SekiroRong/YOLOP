from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import Upsample as uupsample

from nets.CSPdarknet import darknet53
from nets.common import BottleneckCSP

from config import depth_img


def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

#---------------------------------------------------#
#   SPP结构，利用不同大小的池化核进行池化
#   池化后堆叠
#---------------------------------------------------#
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features

#---------------------------------------------------#
#   卷积 + 上采样
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x

#---------------------------------------------------#
#   三次卷积块
#---------------------------------------------------#
def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

#---------------------------------------------------#
#   五次卷积块
#---------------------------------------------------#
def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

#---------------------------------------------------#
#   最后获得yolov4的输出
#---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m

#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes):
        super(YoloBody, self).__init__()
        #---------------------------------------------------#   
        #   生成CSPdarknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        #---------------------------------------------------#
        self.backbone = darknet53(None)

        self.conv1      = make_three_conv([512,1024],1024)
        self.SPP        = SpatialPyramidPooling()
        self.conv2      = make_three_conv([512,1024],2048)

        self.upsample1          = Upsample(512,256)
        self.conv_for_P4        = conv2d(512,256,1)
        self.make_five_conv1    = make_five_conv([256, 512],512)

        self.upsample2          = Upsample(256,128)
        if depth_img:
            self.conv_for_P3        = conv2d(256*2,128,1)
        else:
            self.conv_for_P3 = conv2d(256, 128, 1)
        self.make_five_conv2    = make_five_conv([128, 256],256)

        # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
        self.yolo_head3         = yolo_head([256, len(anchors_mask[0]) * (5 + num_classes)],128)

        self.down_sample1       = conv2d(128,256,3,stride=2)
        self.make_five_conv3    = make_five_conv([256, 512],512)

        # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
        self.yolo_head2         = yolo_head([512, len(anchors_mask[1]) * (5 + num_classes)],256)

        self.down_sample2       = conv2d(256,512,3,stride=2)
        self.make_five_conv4    = make_five_conv([512, 1024],1024)

        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        self.yolo_head1         = yolo_head([1024, len(anchors_mask[2]) * (5 + num_classes)],512)

        # # ll和da的seg部分
        # self.conv3              = conv2d(256,128,3)
        # self.uupsample1          = uupsample(None, 2, 'nearest')
        # self.BottleneckCSP1     = BottleneckCSP(128, 64, 1, False)
        # self.conv4              = conv2d(64, 32, 3)
        # self.uupsample2          = uupsample(None, 2, 'nearest')
        # self.conv5              = conv2d(32, 16, 3)
        # self.BottleneckCSP2     = BottleneckCSP(16, 8, 1, False)
        # self.uupsample3          = uupsample(None, 2, 'nearest')
        # self.conv6              = conv2d(8, 3, 3)




    def forward(self, x, y=None):
        #  backbone
        x2, x1, x0 = self.backbone(x, y)

        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,2048 
        P5 = self.conv1(x0)
        P5 = self.SPP(P5)
        # 13,13,2048 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        P5 = self.conv2(P5)

        # 13,13,512 -> 13,13,256 -> 26,26,256
        P5_upsample = self.upsample1(P5)
        # 26,26,512 -> 26,26,256
        P4 = self.conv_for_P4(x1)
        # 26,26,256 + 26,26,256 -> 26,26,512
        P4 = torch.cat([P4,P5_upsample],axis=1)
        # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        P4 = self.make_five_conv1(P4)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        P4_upsample = self.upsample2(P4)
        # 52,52,256 -> 52,52,128
        P3 = self.conv_for_P3(x2)
        # 52,52,128 + 52,52,128 -> 52,52,256
        P3_upsample = torch.cat([P3,P4_upsample],axis=1)
        # 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        P3 = self.make_five_conv2(P3_upsample)

        # 52,52,128 -> 26,26,256
        P3_downsample = self.down_sample1(P3)
        # 26,26,256 + 26,26,256 -> 26,26,512
        P4 = torch.cat([P3_downsample,P4],axis=1)
        # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        P4 = self.make_five_conv3(P4)

        # 26,26,256 -> 13,13,512
        P4_downsample = self.down_sample2(P4)
        # 13,13,512 + 13,13,512 -> 13,13,1024
        P5 = torch.cat([P4_downsample,P5],axis=1)
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        P5 = self.make_five_conv4(P5)

        #---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size,75,52,52)
        #---------------------------------------------------#
        out2 = self.yolo_head3(P3)
        #---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size,75,26,26)
        #---------------------------------------------------#
        out1 = self.yolo_head2(P4)
        #---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size,75,13,13)
        #---------------------------------------------------#
        out0 = self.yolo_head1(P5)

        # # P3 上采样三次(最近邻插值)
        # P3_1 = self.uupsample1(self.conv3(P3_upsample))
        # P3_1 = self.BottleneckCSP1(P3_1)
        # P3_1 = self.uupsample2(self.conv4(P3_1))
        # P3_1 = self.conv5(P3_1)
        # P3_1 = self.BottleneckCSP2(P3_1)
        # P3_1 = self.uupsample3(P3_1)
        # out3 = self.conv6(P3_1)
        #
        # P3_2 = self.uupsample1(self.conv3(P3_upsample))
        # P3_2 = self.BottleneckCSP1(P3_2)
        # P3_2 = self.uupsample2(self.conv4(P3_2))
        # P3_2 = self.conv5(P3_2)
        # P3_2 = self.BottleneckCSP2(P3_2)
        # P3_2 = self.uupsample3(P3_2)
        # out4 = self.conv6(P3_2)

        return out0, out1, out2, P3_upsample

    # def _init_parameters(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             nn.init.constant_(m.bias, 0)

class yoloR(nn.Module):
    def __init__(self, anchors_mask, num_classes):
        super(yoloR, self).__init__()
        self.backbone_DetectHead = YoloBody(anchors_mask, num_classes)

        # ll和da的seg部分
        self.ll_conv3 = conv2d(256, 128, 3)
        self.ll_uupsample1 = uupsample(None, 2, 'nearest')
        self.ll_BottleneckCSP1 = BottleneckCSP(128, 64, 1, False)
        self.ll_conv4 = conv2d(64, 32, 3)
        self.ll_uupsample2 = uupsample(None, 2, 'nearest')
        self.ll_conv5 = conv2d(32, 16, 3)
        self.ll_BottleneckCSP2 = BottleneckCSP(16, 8, 1, False)
        self.ll_uupsample3 = uupsample(None, 2, 'nearest')
        self.ll_conv6 = conv2d(8, 3, 3)

        self.conv3 = conv2d(256, 128, 3)
        self.uupsample1 = uupsample(None, 2, 'nearest')
        self.BottleneckCSP1 = BottleneckCSP(128, 64, 1, False)
        self.conv4 = conv2d(64, 32, 3)
        self.uupsample2 = uupsample(None, 2, 'nearest')
        self.conv5 = conv2d(32, 16, 3)
        self.BottleneckCSP2 = BottleneckCSP(16, 8, 1, False)
        self.uupsample3 = uupsample(None, 2, 'nearest')
        self.conv6 = conv2d(8, 3, 3)

    def forward(self,x, y = None):

        out0, out1, out2, P3_upsample = self.backbone_DetectHead(x, y)
        # P3 上采样三次(最近邻插值)
        P3_1 = self.ll_uupsample1(self.ll_conv3(P3_upsample))
        P3_1 = self.ll_BottleneckCSP1(P3_1)
        P3_1 = self.ll_uupsample2(self.ll_conv4(P3_1))
        P3_1 = self.ll_conv5(P3_1)
        P3_1 = self.ll_BottleneckCSP2(P3_1)
        P3_1 = self.ll_uupsample3(P3_1)
        out3 = self.ll_conv6(P3_1)

        P3_2 = self.uupsample1(self.conv3(P3_upsample))
        P3_2 = self.BottleneckCSP1(P3_2)
        P3_2 = self.uupsample2(self.conv4(P3_2))
        P3_2 = self.conv5(P3_2)
        P3_2 = self.BottleneckCSP2(P3_2)
        P3_2 = self.uupsample3(P3_2)
        out4 = self.conv6(P3_2)

        return out0, out1, out2, out3, out4






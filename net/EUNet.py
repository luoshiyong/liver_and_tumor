from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class AttentionBlock(nn.Module):
    def __init__(self, inch, size):
        super(AttentionBlock, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(512, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple([6, 12, 18])
        modules.append(ASPPConv(512, 1, rate1))
        modules.append(ASPPConv(512, 1, rate2))
        modules.append(ASPPConv(512, 1, rate3))
        modules.append(ASPPPooling(512, 1))

        self.convs = nn.ModuleList(modules)
        self.transconv = nn.Sequential(
            nn.Conv2d(5, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True))

        self.project = nn.Sequential(
            nn.Conv2d(512, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.smothconv = nn.Conv3d(1, 1, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.weights = nn.Sequential(nn.Linear(inch, int(inch / 2)),
                                     nn.LeakyReLU(0.05),
                                     nn.Linear(int(inch / 2), inch),
                                     nn.Sigmoid()
                                     )
        self.ac = nn.Sigmoid()
        self.size = size
        self.inch = inch

    def forward(self, x):
        bs, ch, ww, hh = x.size()
        # channel attention
        re_weights = self.pool(x).view(bs, ch)  # [1, 256, 1, 1]
        re_weights = self.weights(re_weights).view(bs, ch, 1, 1)
        # print("re_weights shape = ",re_weights.shape)

        # spatial info
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        res = self.transconv(res)
        # print("spatial info shape = ",res.shape)
        # get 3d module
        out = torch.bmm(re_weights.view(bs, ch, -1), res.view(bs, 1, -1)).view(bs, ch, ww, hh).view(bs, 1, ch, ww, hh)
        out = self.smothconv(out).view(bs, ch, ww, hh)
        # print("out shape = ",out.shape)
        return self.ac(out)


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=8):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SE_Conv(nn.Module):
    def __init__(self, ch_in, reduction=8):
        super(SE_Conv, self).__init__()
        self.se = SE_Block(ch_in)
        self.bconv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in, ch_in // 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_in // 2),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.se(x)
        x = self.bconv(x)
        return x


class conv_block(nn.Module):
    """
    Convolution Block 
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class SEBlock(nn.Module):
    def __init__(self, f_in, mid_ch):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.weights = nn.Sequential(
            nn.Linear(f_in, mid_ch),
            nn.LeakyReLU(0.05),
            nn.Linear(mid_ch, f_in),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, ch = x.size()[:2]
        re_weights = self.pool(x).view(bs, ch)
        re_weights = self.weights(re_weights).view(bs, ch, 1, 1)
        return x * re_weights


class ASPPup(nn.Module):
    def __init__(self, in_channels, bs=4, ww=28, atrous_rates=[6, 12, 18]):
        super(ASPPup, self).__init__()
        out_channels = int(in_channels / 2)
        self.out_channels = out_channels
        # self.se = SEBlock(out_channels,int(out_channels/2))
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            # self.se
        )
        self.uptensor1 = torch.zeros([bs, self.out_channels, ww * 2, ww * 2])
        self.uptensor2 = torch.zeros([bs, self.out_channels, ww * 2, ww * 2])
        self.uptensor3 = torch.zeros([bs, self.out_channels, ww * 2, ww * 2])
        self.uptensor4 = torch.zeros([bs, self.out_channels, ww * 2, ww * 2])
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        for i in range(ww):
            for j in range(ww):
                self.uptensor1[:, :, 2 * i, 2 * j] = 1
                self.uptensor2[:, :, 2 * i + 1, 2 * j] = 1
                self.uptensor3[:, :, 2 * i, 2 * j + 1] = 1
                self.uptensor4[:, :, 2 * i + 1, 2 * j + 1] = 1
        self.uptensor1 = self.uptensor1.cuda()
        self.uptensor2 = self.uptensor2.cuda()
        self.uptensor3 = self.uptensor3.cuda()
        self.uptensor4 = self.uptensor4.cuda()

    def forward(self, x):
        bss, ch, ww, hh = x.size()
        res = []
        for conv in self.convs:
            res.append(self.up(conv(x)))
        uptensor = self.uptensor1 * res[0] + self.uptensor2 * res[1] + self.uptensor3 * res[2] + self.uptensor4 * res[3]

        # print("res[0] shape = ",res[0].shape)

        return self.project(uptensor)


# input = torch.randn(8,512,28,28)
# model = ASPPup(512)
# out = model(input)
# print(out.shape)

class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1), )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class MSFM(nn.Module):
    def __init__(self):
        super(MSFM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2)
        )
        self.hconv1 = nn.Sequential(
            nn.Conv2d(96, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2)
        )
        self.outconv = nn.Conv2d(48, 1, 1, bias=False)
        self.acc = torch.nn.Sigmoid()

    def forward(self, f224, f112, f56):
        # print("f224 shape = ",f224.shape)
        # print("f112 shape = ", f112.shape)
        # print("f56 shape = ", f56.shape)
        f224 = self.conv1(f224)
        f112 = self.conv2(f112)
        f56 = self.conv3(f56)
        # print("f56 shape = ",f56.shape)
        h112 = torch.cat([f112, f56], dim=1)
        h112 = self.hconv1(h112)

        h224 = torch.cat([h112, f224], dim=1)
        out = self.outconv(h224)
        return self.acc(out)


class EAU_Net2(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, args):
        super(EAU_Net2, self).__init__()
        self.args = args
        in_ch = 3
        out_ch = 1

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.msfm = MSFM()

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])
        self.se_conv1 = SE_Conv(512)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])
        self.se_conv2 = SE_Conv(256)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])
        self.se_conv3 = SE_Conv(128)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])
        self.se_conv4 = SE_Conv(64)
        self.atten_block = AttentionBlock(512, 28)
        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)
        e2 = self.Maxpool1(e1)

        e2 = self.Conv2(e2)
        e3 = self.Maxpool2(e2)

        e3 = self.Conv3(e3)
        e4 = self.Maxpool3(e3)

        e4 = self.Conv4(e4)
        e5 = self.Maxpool4(e4)

        edge_224 = self.msfm(e2, e3, e4)
        edge_112 = self.pp1(edge_224)
        edge_56 = self.pp2(edge_112)
        edge_448 = F.interpolate(edge_112, size=(448, 448), mode='bilinear', align_corners=False)
        # print(edge_224.shape)  [4, 1, 224, 224]
        e5 = self.Conv5(e5)
        e5 = e5 + e5 * self.atten_block(e5)

        d5 = self.Up5(e5)
        # print("d5 shape = ",d5.shape)  # [4, 256, 56, 56]
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        d5 = torch.cat([d5, d5 * edge_56], dim=1)
        d5 = self.se_conv1(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        d4 = torch.cat([d4, d4 * edge_112], dim=1)
        d4 = self.se_conv2(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        d3 = torch.cat([d3, d3 * edge_224], dim=1)
        d3 = self.se_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d2 = torch.cat([d2, d2 * edge_448], dim=1)
        d2 = self.se_conv4(d2)

        out = self.Conv(d2)

        # d1 = self.active(out)

        return out, edge_448

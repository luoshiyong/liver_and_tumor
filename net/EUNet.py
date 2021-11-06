from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch


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


class ASPPup(nn.Module):
    def __init__(self, in_channels, atrous_rates=[6, 12, 18]):
        super(ASPPup, self).__init__()
        self.out_channels = int(in_channels / 2)
        # self.se = SEBlock(out_channels,int(out_channels/2))
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, self.out_channels, 1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, self.out_channels, rate1))
        modules.append(ASPPConv(in_channels, self.out_channels, rate2))
        modules.append(ASPPConv(in_channels, self.out_channels, rate3))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            # self.se
        )
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        bs, ch, ww, hh = x.size()
        uptensor1 = torch.zeros([bs, self.out_channels, ww * 2, ww * 2])
        uptensor2 = torch.zeros([bs, self.out_channels, ww * 2, ww * 2])
        uptensor3 = torch.zeros([bs, self.out_channels, ww * 2, ww * 2])
        uptensor4 = torch.zeros([bs, self.out_channels, ww * 2, ww * 2])
        for i in range(ww):
            for j in range(ww):
                uptensor1[:, :, 2 * i, 2 * j] = 1
                uptensor2[:, :, 2 * i, 2 * j + 1] = 1
                uptensor3[:, :, 2 * i + 1, 2 * j] = 1
                uptensor4[:, :, 2 * i + 1, 2 * j + 1] = 1
        uptensor1 = uptensor1.cuda()
        uptensor2 = uptensor2.cuda()
        uptensor3 = uptensor3.cuda()
        uptensor4 = uptensor4.cuda()
        res = []
        for conv in self.convs:
            res.append(self.up(conv(x)))
        uptensor = uptensor1 * res[0] + uptensor2 * res[1] + uptensor3 * res[2] + uptensor4 * res[3]
        # print("res[0] shape = ",res[0].shape)

        return self.project(uptensor)


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
        # print("x before aspp = ",x.shape)
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
            # print("conv(x) shape = ",conv(x).shape) # [1, 256, 28, 28]
            # print(conv)
            res.append(conv(x))
            # print(len(res))
        res = torch.cat(res, dim=1)
        # print("concat shape = ",res.shape)
        return self.project(res)
class EU2_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, args):
        super(EU2_Net, self).__init__()
        self.args = args
        in_ch = 3
        out_ch = 1

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        # self.Conv5 = conv_block(filters[3], filters[4])

        # self.Up5 = up_conv(filters[4], filters[3])
        # self.Up5 = ASPPup(in_channels=512)
        # self.Up_conv5 = conv_block(512, 256)

        # self.Up4 = up_conv(filters[3], filters[2])
        self.Up4 = ASPPup(256)
        self.Up_conv4 = conv_block(256, 128)

        # self.Up3 = up_conv(filters[2], filters[1])
        self.Up3 = ASPPup(128)
        self.Up_conv3 = conv_block(128, 64)

        self.Up2 = up_conv(64, 32)
        # self.Up2 = ASPPup(32)
        self.Up_conv2 = conv_block(64, 16)

        self.Conv = nn.Conv2d(16, out_ch, kernel_size=1, stride=1, padding=0)

        # self.aspp = ASPP(512, [6, 12, 18])
        # self.downconv = nn.Conv2d(1024,256,1)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        #e5 = self.Maxpool4(e4)
        #e5 = self.Conv5(e5)
        # print("e5 shape = ", e5.shape) [4,512,28,28]
        # e5 = self.aspp(e5)
        # e5 = self.downconv(e5)
        # print("e5 shape = ",e5.shape)
        #d5 = self.Up5(e5)

        #d5 = torch.cat((e4, d5), dim=1)
        # print("d5 shape = ", d5.shape)
        #d5 = self.Up_conv5(d5)

        d4 = self.Up4(e4)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        # print("d2 shape = ",d2.shape)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        # d1 = self.active(out)

        return out

class EU1_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, args):
        super(EU1_Net, self).__init__()
        self.args = args
        in_ch = 3
        out_ch = 1

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        # self.Up5 = up_conv(filters[4], filters[3])
        self.Up5 = ASPPup(in_channels=512)
        self.Up_conv5 = conv_block(512, 256)

        # self.Up4 = up_conv(filters[3], filters[2])
        self.Up4 = ASPPup(256)
        self.Up_conv4 = conv_block(256, 128)

        # self.Up3 = up_conv(filters[2], filters[1])
        self.Up3 = ASPPup(128)
        self.Up_conv3 = conv_block(128, 64)

        self.Up2 = up_conv(64, 32)
        # self.Up2 = ASPPup(32)
        self.Up_conv2 = conv_block(64, 16)

        self.Conv = nn.Conv2d(16, out_ch, kernel_size=1, stride=1, padding=0)

        # self.aspp = ASPP(512, [6, 12, 18])
        # self.downconv = nn.Conv2d(1024,256,1)

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
        e5 = self.Conv5(e5)
        # print("e5 shape = ", e5.shape) [4,512,28,28]
        # e5 = self.aspp(e5)
        # e5 = self.downconv(e5)
        # print("e5 shape = ",e5.shape)
        d5 = self.Up5(e5)

        d5 = torch.cat((e4, d5), dim=1)
        # print("d5 shape = ", d5.shape)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        # print("d2 shape = ",d2.shape)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        # d1 = self.active(out)

        return out
class EU_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, args):
        super(EU_Net, self).__init__()
        self.args = args
        in_ch = 3
        out_ch = 1

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        # self.Up5 = up_conv(filters[4], filters[3])
        self.Up5 = ASPPup(in_channels=256)
        self.Up_conv5 = conv_block(384, 128)

        # self.Up4 = up_conv(filters[3], filters[2])
        self.Up4 = ASPPup(128)
        self.Up_conv4 = conv_block(192, 64)

        # self.Up3 = up_conv(filters[2], filters[1])
        self.Up3 = ASPPup(64)
        self.Up_conv3 = conv_block(96, 32)

        self.Up2 = up_conv(32, 16)
        # self.Up2 = ASPPup(32)
        self.Up_conv2 = conv_block(48, 16)

        self.Conv = nn.Conv2d(16, out_ch, kernel_size=1, stride=1, padding=0)

        self.aspp = ASPP(512, [6, 12, 18])
        # self.downconv = nn.Conv2d(1024,256,1)

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
        e5 = self.Conv5(e5)
        # print("e5 shape = ", e5.shape)
        e5 = self.aspp(e5)
        # e5 = self.downconv(e5)
        # print("e5 shape = ",e5.shape)
        d5 = self.Up5(e5)

        d5 = torch.cat((e4, d5), dim=1)
        # print("d5 shape = ", d5.shape)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        # d1 = self.active(out)

        return out

input = torch.randn(4,3,448,448)
input = input.cuda()
model = EU1_Net("")
model = model.cuda()
out = model(input)
print(out.shape)
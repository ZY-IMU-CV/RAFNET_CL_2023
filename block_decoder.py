import torch.nn as nn
import torch
import math
import time
from RAFNet.blocks import *
from torch.nn import functional as F


class BiFPN(nn.Module):
    def __init__(self, fpn_sizes):
        super(BiFPN, self).__init__()

        P3_channels, P4_channels, P5_channels, P6_channels, classnum = fpn_sizes
        self.W_bifpn = 64

        # self.p6_td_conv  = nn.Conv2d(P6_channels, self.W_bifpn, kernel_size=3, stride=1, groups=self.W_bifpn, bias=True, padding=1)
        self.p6_td_conv = nn.Conv2d(P6_channels, self.W_bifpn, kernel_size=3, stride=1, bias=True, padding=1)

        self.p6_td_act = nn.ReLU()
        self.p6_td_conv_bn = nn.BatchNorm2d(self.W_bifpn)
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p6_td_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p6_td_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)

        self.p5_td_conv = nn.Conv2d(P5_channels, self.W_bifpn, kernel_size=3, stride=1, bias=True, padding=1)
        self.p5_td_conv_2 = nn.Conv2d(self.W_bifpn, self.W_bifpn, kernel_size=3, stride=1, groups=self.W_bifpn,
                                      bias=True, padding=1)
        self.p5_td_act = nn.ReLU()
        self.p5_td_conv_bn = nn.BatchNorm2d(self.W_bifpn)
        self.p5_td_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p5_td_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)

        self.p4_td_conv = nn.Conv2d(P4_channels, self.W_bifpn, kernel_size=3, stride=1, bias=True, padding=1)
        self.p4_td_conv_2 = nn.Conv2d(self.W_bifpn, self.W_bifpn, kernel_size=3, stride=1, groups=self.W_bifpn,
                                      bias=True, padding=1)
        self.p4_td_act = nn.ReLU()
        self.p4_td_conv_bn = nn.BatchNorm2d(self.W_bifpn)
        self.p4_td_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_td_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p3_out_conv = nn.Conv2d(P3_channels, self.W_bifpn, kernel_size=3, stride=1, bias=True, padding=1)
        self.p3_out_conv_2 = nn.Conv2d(self.W_bifpn, self.W_bifpn, kernel_size=3, stride=1, groups=self.W_bifpn,
                                       bias=True, padding=1)
        self.p3_out_act = nn.ReLU()
        self.p3_out_conv_bn = nn.BatchNorm2d(self.W_bifpn)
        self.p3_out_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p3_out_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # self.p4_out_conv = nn.Conv2d(P4_channels, self.W_bifpn, kernel_size=3, stride=1, bias=True, padding=1)
        self.p4_out_conv = nn.Conv2d(self.W_bifpn, self.W_bifpn, kernel_size=3, stride=1, groups=self.W_bifpn,
                                     bias=True, padding=1)
        self.p4_out_act = nn.ReLU()
        self.p4_out_conv_bn = nn.BatchNorm2d(self.W_bifpn)
        self.p4_out_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_out_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_out_w3 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p3_downsample = nn.MaxPool2d(kernel_size=2)

        # self.p5_out_conv = nn.Conv2d(P5_channels,self.W_bifpn, kernel_size=3, stride=1, bias=True, padding=1)
        self.p5_out_conv = nn.Conv2d(self.W_bifpn, self.W_bifpn, kernel_size=3, stride=1, groups=self.W_bifpn,
                                     bias=True, padding=1)
        self.p5_out_act = nn.ReLU()
        self.p5_out_conv_bn = nn.BatchNorm2d(self.W_bifpn)
        self.p5_out_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p5_out_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p5_out_w3 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_downsample = nn.MaxPool2d(kernel_size=2)

        # self.p6_out_conv = nn.Conv2d(P6_channels, self.W_bifpn, kernel_size=3, stride=1, bias=True, padding=1)
        self.p6_out_conv = nn.Conv2d(self.W_bifpn, self.W_bifpn, kernel_size=3, stride=1, groups=self.W_bifpn,
                                     bias=True, padding=1)
        self.p6_out_act = nn.ReLU()
        self.p6_out_conv_bn = nn.BatchNorm2d(self.W_bifpn)
        self.p6_out_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p6_out_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p5_downsample = nn.MaxPool2d(kernel_size=2)
        # self.p4_downsample= nn.MaxPool2d(kernel_size=2)
        self.classifier = nn.Conv2d(64, classnum, 1)

    def forward(self, inputs):
        epsilon = 0.0001
        P3, P4, P5, P6 = inputs["4"], inputs["8"], inputs["16"], inputs["32"]
        # print ("Input::", P3.shape, P4.shape, P5.shape, P6.shape, P7.shape)
        # P6_td = self.p6_td_conv((self.p6_td_w1 * P6 ) /
        #                         (self.p6_td_w1 + epsilon))

        # P6_td_inp = self.p6_td_conv(P6)
        P6_td = self.p6_td_conv(P6)
        # P6_td = self.p6_td_conv_2(P6_td_inp)
        P6_td = self.p6_td_act(P6_td)
        P6_td = self.p6_td_conv_bn(P6_td)

        P5_td_inp = self.p5_td_conv(P5)
        # print (P5_td_inp.shape, P6_td.shape)

        P5_td = self.p5_td_conv_2((self.p5_td_w1 * P5_td_inp + self.p5_td_w2 * self.p6_upsample(P6_td)) /
                                  (self.p5_td_w1 + self.p5_td_w2 + epsilon))
        P5_td = self.p5_td_act(P5_td)
        P5_td = self.p5_td_conv_bn(P5_td)

        # print (P4.shape, P5_td.shape)
        P4_td_inp = self.p4_td_conv(P4)
        P4_td = self.p4_td_conv_2((self.p4_td_w1 * P4_td_inp + self.p4_td_w2 * self.p5_upsample(P5_td)) /
                                  (self.p4_td_w1 + self.p4_td_w2 + epsilon))
        P4_td = self.p4_td_act(P4_td)
        P4_td = self.p4_td_conv_bn(P4_td)

        P3_td = self.p3_out_conv(P3)
        P3_out = self.p3_out_conv_2((self.p3_out_w1 * P3_td + self.p3_out_w2 * self.p4_upsample(P4_td)) /
                                    (self.p3_out_w1 + self.p3_out_w2 + epsilon))
        P3_out = self.p3_out_act(P3_out)
        P3_out = self.p3_out_conv_bn(P3_out)

        # print (P4_td.shape, P3_out.shape)

        P4_out = self.p4_out_conv(
            (self.p4_out_w1 * P4_td_inp + self.p4_out_w2 * P4_td + self.p4_out_w3 * self.p3_downsample(P3_out))
            / (self.p4_out_w1 + self.p4_out_w2 + self.p4_out_w3 + epsilon))
        P4_out = self.p4_out_act(P4_out)
        P4_out = self.p4_out_conv_bn(P4_out)

        P5_out = self.p5_out_conv(
            (self.p5_out_w1 * P5_td_inp + self.p5_out_w2 * P5_td + self.p5_out_w3 * self.p4_downsample(P4_out))
            / (self.p5_out_w2 + self.p5_out_w3 + epsilon))
        P5_out = self.p5_out_act(P5_out)
        P5_out = self.p5_out_conv_bn(P5_out)

        P6_out = self.p6_out_conv((self.p6_out_w1 * P6_td + self.p6_out_w2 * self.p5_downsample(P5_out))
                                  / (self.p6_out_w1 + self.p6_out_w2 + epsilon))
        P6_out = self.p6_out_act(P6_out)
        P6_out = self.p6_out_conv_bn(P6_out)

        P6_out = F.interpolate(P6_out, size=P3_out.shape[-2:], mode='bilinear', align_corners=False)
        P5_out = F.interpolate(P5_out, size=P3_out.shape[-2:], mode='bilinear', align_corners=False)
        P4_out = F.interpolate(P4_out, size=P3_out.shape[-2:], mode='bilinear', align_corners=False)

        P3_out = self.classifier(P3_out + P4_out + P5_out + P6_out)

        return P3_out


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        # self.bn_atten = nn.BatchNorm2d(out_chan)
        self.bn_atten = nn.BatchNorm2d(out_chan)

        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        # atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out


class STDC_docker(nn.Module):
    def __init__(self):
        super(STDC_docker, self).__init__()
        self.avg_conv = nn.Sequential(nn.Conv2d(256, 128, (1, 1), (1, 1)), nn.ReLU(inplace=True))
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(256, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        # self.conv_avg = ConvBNReLU(256, 128, ks=1, stride=1, padding=0)
        self.ffm = FeatureFusionModule(256, 256)

        self.conv = ConvBNReLU(256, 256, ks=3, stride=1, padding=1)
        self.classfier = nn.Conv2d(256, 19, kernel_size=(1, 1), bias=False)

    def forward(self, x):
        feat4, feat8, feat16, feat32 = x["4"], x["8"], x["16"], x["32"]
        H4, W4 = feat4.size()[2:]
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        avg = self.avg_conv(avg)
        avg_up = F.interpolate(avg, (H32, W32), mode='nearest')

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)
        fout = self.ffm(feat8, feat16_up)
        fout = self.conv(fout)
        fout = self.classfier(fout)
        fout = F.interpolate(fout, (H4, W4), mode='bilinear')
        return fout


class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class down32_Decoder1(nn.Module):  # Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4, channels8, channels16, channels32 = channels["4"], channels["8"], channels["16"], channels["32"]
        self.aspp = ASPP(256, 256)
        self.head32 = ConvBnAct(channels32, 128, 1)
        self.head16 = ConvBnAct(channels16, 128, 1)
        self.head8 = ConvBnAct(channels8, 128, 1)
        self.head4 = ConvBnAct(channels4, 8, 1)
        self.conv16 = ConvBnAct(128, 128, 3, 1, 1)
        self.conv8 = ConvBnAct(128, 64, 3, 1, 1)
        self.conv4 = ConvBnAct(64 + 8, 64, 3, 1, 1)
        self.classifier = nn.Conv2d(64, num_classes, 1)
        self.gloable_pool = nn.AdaptiveAvgPool2d((1, 1));

    def forward(self, x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x32 = self.aspp(x32)
        x32 = self.head32(x32)
        x16 = self.head16(x16)
        x8 = self.head8(x8)
        x4 = self.head4(x4)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)  # 数组上采样操作
        x16 = x16 + x32
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)  # 数组上采样操作
        x8 = x8 + x16
        x8 = self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = torch.cat((x8, x4), dim=1)
        x4 = self.conv4(x4)
        x4 = self.classifier(x4)
        return x4


class Multi_scale_future_Extraction(nn.Module):
    def __init__(self, dilations):
        super(Multi_scale_future_Extraction, self).__init__()
        convs = []
        for d in dilations:
            convs.append(nn.Conv2d(1, 1, (3, 3), (1, 1), padding=(d, d), dilation=(d, d)))
        self.convs = nn.ModuleList(convs)
        self.dil = len(dilations)
        self.bn1 = nn.BatchNorm2d(len(dilations))
        self.act1 = nn.ReLU(inplace=True)
        self.ConvBnAct = nn.Sequential(nn.Conv2d(len(dilations), 1, (1, 1)), nn.BatchNorm2d(1), nn.ReLU(inplace=True))

    def forward(self, x):
        res = []
        for i in range(self.dil):
            res.append(self.convs[i](x))
        res = torch.cat(res, dim=1)
        res = self.act1(self.bn1(res))
        res = self.ConvBnAct(res)
        return res


class e_ASPP(nn.Module):
    def __init__(self, inChannel, outchannel, dilations):
        super(e_ASPP, self).__init__()
        self.Channel_Reduction = nn.Sequential(
            nn.Conv2d(inChannel, inChannel // 4, (1, 1)),
            nn.BatchNorm2d(inChannel // 4),
            nn.ReLU(inplace=True))
        MSFF = []
        for i in range(inChannel // 4):
            MSFF.append(Multi_scale_future_Extraction(dilations))
        self.MSFF = nn.ModuleList(MSFF)
        self.channels = inChannel // 4
        self.MSFF_BN_ACt = nn.Sequential(nn.BatchNorm2d(inChannel // 4), nn.ReLU(inplace=True))
        self.Inter_channel_Feature_Fusion = nn.Sequential(nn.Conv2d(inChannel // 4, outchannel, (1, 1)),
                                                          nn.BatchNorm2d(outchannel),
                                                          nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.Channel_Reduction(x)
        x = torch.tensor_split(x, self.channels, dim=1)
        res = []
        for i in range(self.channels):
            res.append(self.MSFF[i](x[i]))
        res = torch.cat(res, dim=1)
        res = self.MSFF_BN_ACt(res)
        res = self.Inter_channel_Feature_Fusion(res)
        return res


class eASPP_deccoder(nn.Module):  # Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4, channels8, channels32 = channels["4"], channels["8"], channels["32"]
        self.head8 = ConvBnAct(channels8, 128, 1)
        self.head4 = ConvBnAct(channels4, 16, 1)
        self.conv8 = ConvBnAct(128, 64, 3, 1, 1)
        self.conv4 = ConvBnAct(64 + 16, 64, 3, 1, 1)
        self.classifier = nn.Conv2d(64, num_classes, (1, 1))
        self.easpp = e_ASPP(256, 128, [6, 12, 18])

    def forward(self, x):
        x4, x8, x32 = x["4"], x["8"], x["32"]
        x32 = F.interpolate(x32, scale_factor=2, mode='bilinear', align_corners=False)  # 数组上采样操作
        x32 = self.easpp(x32)
        x8 = self.head8(x8)
        x4 = self.head4(x4)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)  # 数组上采样操作
        x8 = x8 + x32
        x8 = self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = torch.cat((x8, x4), dim=1)
        x4 = self.conv4(x4)
        x4 = self.classifier(x4)
        return x4


class eASPP_deccoder2(nn.Module):  # Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4, channels8, channels16, channels32 = channels["4"], channels["8"], channels["16"], channels["32"]
        self.head16 = ConvBnAct(channels16, 128, 1)
        self.head8 = ConvBnAct(channels8, 128, 1)
        self.head4 = ConvBnAct(channels4, 16, 1)
        self.conv16 = ConvBnAct(128, 128, 3, 1, 1)
        self.conv8 = ConvBnAct(128, 64, 3, 1, 1)
        self.conv4 = ConvBnAct(64 + 16, 64, 3, 1, 1)
        self.classifier = nn.Conv2d(64, num_classes, (1, 1))
        self.easpp = e_ASPP(256, 128, [6, 12, 18])

    def forward(self, x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)  # 数组上采样操作
        x32 = self.easpp(x32)
        x16 = self.head16(x16)
        x8 = self.head8(x8)
        x4 = self.head4(x4)
        x16 = self.conv16(x16 + x32)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)  # 数组上采样操作
        x8 = x8 + x16
        x8 = self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = torch.cat((x8, x4), dim=1)
        x4 = self.conv4(x4)
        x4 = self.classifier(x4)
        return x4


class eASPP_deccoder3(nn.Module):  # Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4, channels8, channels16, channels32 = channels["4"], channels["8"], channels["16"], channels["32"]
        self.head16 = ConvBnAct(channels16, 128, 1)
        self.head8 = ConvBnAct(channels8, 128, 1)
        self.head4 = ConvBnAct(channels4, 8, 1)
        self.conv16 = ConvBnAct(128, 128, 3, 1, 1)
        self.conv8 = ConvBnAct(128, 64, 3, 1, 1)
        self.conv4 = ConvBnAct(64 + 8, 64, 3, 1, 1)
        self.classifier = nn.Conv2d(64, num_classes, (1, 1))
        self.easpp = e_ASPP(256, 128, [6, 12, 18])

    def forward(self, x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)  # 数组上采样操作
        x32 = self.easpp(x32)
        x16 = self.head16(x16)
        x8 = self.head8(x8)
        x4 = self.head4(x4)
        x16 = self.conv16(x16 + x32)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)  # 数组上采样操作
        x8 = x8 + x16
        x8 = self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = torch.cat((x8, x4), dim=1)
        x4 = self.conv4(x4)
        x4 = self.classifier(x4)
        return x4


class eASPP_deccoder4(nn.Module):  # Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4, channels8, channels16, channels32 = channels["4"], channels["8"], channels["16"], channels["32"]
        self.head16 = ConvBnAct(channels16, 128, 1)
        self.head8 = ConvBnAct(channels8, 128, 1)
        self.head4 = ConvBnAct(channels4, 16, 1)
        self.conv16 = ConvBnAct(128, 128, 3, 1, 1)
        self.conv8 = ConvBnAct(128, 64, 3, 1, 1)
        self.conv4 = ConvBnAct(64 + 16, 64, 3, 1, 1)
        self.classifier = nn.Conv2d(64, num_classes, (1, 1))
        self.easpp = e_ASPP(256, 128, [4, 8, 12])

    def forward(self, x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)  # 数组上采样操作
        x32 = self.easpp(x32)
        x16 = self.head16(x16)
        x8 = self.head8(x8)
        x4 = self.head4(x4)
        x16 = self.conv16(x16 + x32)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)  # 数组上采样操作
        x8 = x8 + x16
        x8 = self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = torch.cat((x8, x4), dim=1)
        x4 = self.conv4(x4)
        x4 = self.classifier(x4)
        return x4


class eASPP_deccoder5(nn.Module):  # Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4, channels8, channels32 = channels["4"], channels["8"], channels["32"]
        self.head8 = ConvBnAct(channels8, 128, 1)
        self.head4 = ConvBnAct(channels4, 16, 1)
        self.conv8 = ConvBnAct(128, 64, 3, 1, 1)
        self.conv4 = ConvBnAct(64 + 16, 64, 3, 1, 1)
        self.classifier = nn.Conv2d(64, num_classes, (1, 1))
        self.easpp = e_ASPP(256, 128, [2, 4, 8])

    def forward(self, x):
        x4, x8, x32 = x["4"], x["8"], x["32"]
        x32 = F.interpolate(x32, scale_factor=2, mode='bilinear', align_corners=False)  # 数组上采样操作
        x32 = self.easpp(x32)
        x8 = self.head8(x8)
        x4 = self.head4(x4)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)  # 数组上采样操作
        x8 = x8 + x32
        x8 = self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = torch.cat((x8, x4), dim=1)
        x4 = self.conv4(x4)
        x4 = self.classifier(x4)
        return x4


if __name__ == "__main__":
    fpn = BiFPN([48, 128, 256, 256, 19])

    c1 = torch.randn([1, 48, 256, 512])
    c2 = torch.randn([1, 128, 128, 256])
    c3 = torch.randn([1, 256, 64, 128])
    c4 = torch.randn([1, 256, 32, 64])

    feats = {"4": c1, "8": c2, "16": c3, "32": c4}
    #d1 = twelve_Decoder0(19, {"4": 48, "8": 128, "16": 256, "32": 256})
    c1 = torch.randn([1, 48, 256, 512])
    c2 = torch.randn([1, 128, 128, 256])
    c3 = torch.randn([1, 256, 64, 128])
    for i in range(100):
        torch.cuda.synchronize()
        start_time = time.time()
        output = fpn(feats)
        torch.cuda.synchronize()
        end_tiem = time.time()
        time_sum = end_tiem - start_time
        print(time_sum)
        print(output.shape)

        fin = {"4": c1, "8": c2, "16": c3, "32": c4}
        torch.cuda.synchronize()
        start_time = time.time()
        output = d1(fin)
        torch.cuda.synchronize()
        end_tiem = time.time()
        time_sum = end_tiem - start_time
        print(time_sum)
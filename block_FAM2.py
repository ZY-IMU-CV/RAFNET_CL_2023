from torch import nn
import torch
from torch.nn import functional as F
# from network.sfnet_resnet import UperNetAlignHead
from RAFNet.block_operate import AlignedModule, PSPModule


def activation():
    return nn.ReLU(inplace=True)


def norm2d(out_channels):
    return nn.BatchNorm2d(out_channels)


class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, apply_act=True):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = norm2d(out_channels)
        if apply_act:
            self.act = activation()
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, group_width, stride):
        super().__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = norm2d(out_channels)
        self.act1 = activation()
        dilation = dilations[0]
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=(1, 1), groups=out_channels,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm2d(out_channels)
        self.act2 = activation()
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = norm2d(out_channels)
        self.act3 = activation()
        self.avg = nn.AvgPool2d(2, 2, ceil_mode=True)
        # self.short = Shortcut(out_channels,groups)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.stride == 2:
            x = self.avg(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        return x


class RegSegBody(nn.Module):
    def __init__(self):
        super().__init__()
        gw = 16
        self.stage4 = DBlock(32, 48, [1], gw, 2)
        self.stage8 = nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16 = nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256, 256, [2], gw, 1),
            DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256, 256, [2], gw, 1),
            DBlock(256, 256, [4], gw, 1),
            DBlock(256, 256, [8], gw, 1)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4": x4, "8": x8, "16": x16, "32": x32}

    def channels(self):
        return {"4": 48, "8": 128, "16": 256, "32": 256}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3_bn_relu(in_planes, out_planes, stride=1, normal_layer=nn.BatchNorm2d):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        normal_layer(out_planes),
        nn.ReLU(inplace=True),
    )


class UperNetAlignHead(nn.Module):

    def __init__(self, inplane, num_class, norm_layer=nn.BatchNorm2d, fpn_inplanes=[256, 512, 1024, 2048], fpn_dim=256,
                 conv3x3_type="conv", fpn_dsn=False):
        super(UperNetAlignHead, self).__init__()

        self.ppm = PSPModule(inplane, norm_layer=norm_layer, out_features=fpn_dim)
        self.fpn_dsn = fpn_dsn
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    norm_layer(fpn_dim),
                    nn.ReLU(inplace=False)
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        self.fpn_out_align = []
        self.dsn = []
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))

            if conv3x3_type == 'conv':
                self.fpn_out_align.append(
                    AlignedModule(inplane=fpn_dim, outplane=fpn_dim // 2)
                )

            if self.fpn_dsn:
                self.dsn.append(
                    nn.Sequential(
                        nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1),
                        norm_layer(fpn_dim),
                        nn.ReLU(),
                        nn.Dropout2d(0.1),
                        nn.Conv2d(fpn_dim, num_class, kernel_size=1, stride=1, padding=0, bias=True)
                    )
                )

        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.fpn_out_align = nn.ModuleList(self.fpn_out_align)

        if self.fpn_dsn:
            self.dsn = nn.ModuleList(self.dsn)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )
        # self.ppm_replace = ConvBnAct(256,64,1)

    def forward(self, x):
        conv_out = [x["4"], x["8"], x["16"], x["32"]]
        psp_out = self.ppm(conv_out[-1])
        # psp_out = self.ppm_replace(conv_out[-1])
        f = psp_out
        fpn_feature_list = [psp_out]
        out = []
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)
            f = self.fpn_out_align[i]([conv_x, f])
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))
            if self.fpn_dsn:
                out.append(self.dsn[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]

        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=True))

        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        return x


class UperNetAlignHead2(nn.Module):

    def __init__(self, inplane, num_class, norm_layer=nn.BatchNorm2d, fpn_inplanes=[256, 512, 1024, 2048], fpn_dim=256,
                 conv3x3_type="conv", fpn_dsn=False):
        super(UperNetAlignHead2, self).__init__()

        self.fpn_dsn = fpn_dsn
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    norm_layer(fpn_dim),
                    nn.ReLU(inplace=False)
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        self.fpn_out_align = []
        self.dsn = []
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
            if self.fpn_dsn:
                self.dsn.append(
                    nn.Sequential(
                        nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1),
                        norm_layer(fpn_dim),
                        nn.ReLU(),
                        nn.Dropout2d(0.1),
                        nn.Conv2d(fpn_dim, num_class, kernel_size=1, stride=1, padding=0, bias=True)
                    )
                )

        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.fpn_out_align = nn.ModuleList(self.fpn_out_align)

        if self.fpn_dsn:
            self.dsn = nn.ModuleList(self.dsn)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )
        self.ppm_replace = ConvBnAct(256, 64, 1)

    def forward(self, x):
        conv_out = [x["4"], x["8"], x["16"], x["32"]]
        psp_out = self.ppm_replace(conv_out[-1])
        f = psp_out
        fpn_feature_list = [psp_out]
        out = []
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)
            f = f = nn.functional.interpolate(f, size=conv_x.shape[-2:], mode='bilinear', align_corners=True)
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))
            if self.fpn_dsn:
                out.append(self.dsn[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]

        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=True))

        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        return x


class UperNetAlignHead3(nn.Module):

    def __init__(self, inplane, num_class, norm_layer=nn.BatchNorm2d, fpn_inplanes=[256, 512, 1024, 2048], fpn_dim=256,
                 conv3x3_type="conv", fpn_dsn=False):
        super(UperNetAlignHead3, self).__init__()

        self.fpn_dsn = fpn_dsn
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    norm_layer(fpn_dim),
                    nn.ReLU(inplace=False)
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)
        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )
        self.ppm_replace = ConvBnAct(256, 64, 1)

    def forward(self, x):
        conv_out = [x["4"], x["8"], x["16"], x["32"]]
        psp_out = self.ppm_replace(conv_out[-1])
        f = psp_out
        fpn_feature_list = [psp_out]
        out = []
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)
            f = f = nn.functional.interpolate(f, size=conv_x.shape[-2:], mode='bilinear', align_corners=True)
            f = conv_x + f
        output_size = f.size()[2:]
        x = self.conv_last(f)
        return x


class UperNetAlignHead4(nn.Module):

    def __init__(self, inplane, num_class, norm_layer=nn.BatchNorm2d, fpn_inplanes=[256, 512, 1024, 2048], fpn_dim=256,
                 conv3x3_type="conv"):
        super(UperNetAlignHead4, self).__init__()

        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    norm_layer(fpn_dim),
                    nn.ReLU(inplace=False)
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        self.fpn_out_align = []
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.fpn_out_align = nn.ModuleList(self.fpn_out_align)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )
        self.ppm_replace = ConvBnAct(256, 32, 1)

    def forward(self, x):
        conv_out = [x["4"], x["8"], x["16"], x["32"]]
        psp_out = self.ppm_replace(conv_out[-1])
        f = psp_out
        fpn_feature_list = [psp_out]
        out = []
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)
            f = f = nn.functional.interpolate(f, size=conv_x.shape[-2:], mode='bilinear', align_corners=True)
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]

        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=True))
        fusion_out = fusion_list[0] + fusion_list[1] + fusion_list[2] + fusion_list[3]

        x = self.conv_last(fusion_out)

        return x


class UperNetAlignHead5(nn.Module):

    def __init__(self, inplane, num_class, norm_layer=nn.BatchNorm2d, fpn_inplanes=[256, 512, 1024, 2048], fpn_dim=256,
                 conv3x3_type="conv"):
        super(UperNetAlignHead5, self).__init__()

        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    norm_layer(fpn_dim),
                    nn.ReLU(inplace=False)
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        self.fpn_out_align = []
        self.dsn = []
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim // 2, 1),
            ))

        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.fpn_out_align = nn.ModuleList(self.fpn_out_align)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(fpn_dim // 2, fpn_dim // 2, 1),
            nn.Conv2d(fpn_dim // 2, num_class, kernel_size=1)
        )
        self.ppm_replace = ConvBnAct(256, 64, 1)
        self.ppm_2 = ConvBnAct(64, 32, 3, 1, 1)

    def forward(self, x):
        conv_out = [x["4"], x["8"], x["16"], x["32"]]
        psp_out = self.ppm_replace(conv_out[-1])
        f = psp_out
        fpn_feature_list = [psp_out]
        out = []
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)
            f = f = nn.functional.interpolate(f, size=conv_x.shape[-2:], mode='bilinear', align_corners=True)
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        fpn_feature_list[3] = self.ppm_2(fpn_feature_list[3])
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=True))
        fusion_out = fusion_list[0] + fusion_list[1] + fusion_list[2] + fusion_list[3]

        x = self.conv_last(fusion_out)

        return x


class AlignNetCnet(nn.Module):

    def __init__(self, num_classes):
        super(AlignNetCnet, self).__init__()

        self.stem = ConvBnAct(3, 32, 3, 2, 1)
        self.backbone = RegSegBody()

        self.head = UperNetAlignHead(inplane=256, num_class=num_classes, norm_layer=nn.BatchNorm2d,
                                     fpn_inplanes=[128, 256, 256], fpn_dim=64, conv3x3_type="conv", fpn_dsn=False)

    def forward(self, x):
        print(x.shape)
        input_shape = x.shape[-2:]
        x = self.stem(x)
        x = self.backbone(x)
        x = self.head(x)
        x = F.interpolate(x[0], size=input_shape, mode='bilinear', align_corners=False)
        return x


class UperNetAlignHead6(nn.Module):

    def __init__(self, inplane, num_class, norm_layer=nn.BatchNorm2d, fpn_inplanes=[256, 512, 1024, 2048], fpn_dim=256,
                 conv3x3_type="conv"):
        super(UperNetAlignHead6, self).__init__()

        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    norm_layer(fpn_dim),
                    nn.ReLU(inplace=False)
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        self.fpn_out_align = []
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(nn.Sequential(
                ConvBnAct(fpn_dim, fpn_dim, 3, 1, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.fpn_out_align = nn.ModuleList(self.fpn_out_align)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )
        self.ppm_replace = ConvBnAct(256, 128, 1)

    def forward(self, x):
        conv_out = [x["4"], x["8"], x["16"], x["32"]]
        psp_out = self.ppm_replace(conv_out[-1])
        f = psp_out
        fpn_feature_list = [psp_out]
        out = []
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)
            f = f = nn.functional.interpolate(f, size=conv_x.shape[-2:], mode='bilinear', align_corners=True)
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]

        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=True))
        fusion_out = fusion_list[0] + fusion_list[1] + fusion_list[2] + fusion_list[3]

        x = self.conv_last(fusion_out)

        return x


class UperNetAlignHead7(nn.Module):

    def __init__(self, inplane, num_class, norm_layer=nn.BatchNorm2d, fpn_inplanes=[256, 512, 1024, 2048], fpn_dim=256,
                 conv3x3_type="conv"):
        super(UperNetAlignHead7, self).__init__()

        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    norm_layer(fpn_dim),
                    nn.ReLU(inplace=False)
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        self.fpn_out_align = []
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(nn.Sequential(
                ConvBnAct(fpn_dim, fpn_dim, 3, 1, 1, groups=fpn_dim),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.fpn_out_align = nn.ModuleList(self.fpn_out_align)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )
        self.ppm_replace = ConvBnAct(256, 128, 1)

    def forward(self, x):
        conv_out = [x["4"], x["8"], x["16"], x["32"]]
        psp_out = self.ppm_replace(conv_out[-1])
        f = psp_out
        fpn_feature_list = [psp_out]
        out = []
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)
            f = f = nn.functional.interpolate(f, size=conv_x.shape[-2:], mode='bilinear', align_corners=True)
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]

        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=True))
        fusion_out = fusion_list[0] + fusion_list[1] + fusion_list[2] + fusion_list[3]

        x = self.conv_last(fusion_out)

        return x


class UperNetAlignHead8(nn.Module):

    def __init__(self, inplane, num_class, norm_layer=nn.BatchNorm2d, fpn_inplanes=[256, 512, 1024, 2048], fpn_dim=256,
                 conv3x3_type="conv"):
        super(UperNetAlignHead8, self).__init__()

        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    norm_layer(fpn_dim),
                    nn.ReLU(inplace=False)
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        self.fpn_out_align = []
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(nn.Sequential(
                ConvBnAct(fpn_dim, fpn_dim, 3, 1, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.fpn_out_align = nn.ModuleList(self.fpn_out_align)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(fpn_dim, 64, 1),
            nn.Conv2d(64, num_class, kernel_size=1)
        )
        self.ppm_replace = ConvBnAct(256, 128, 1)

    def forward(self, x):
        conv_out = [x["4"], x["8"], x["16"], x["32"]]
        psp_out = self.ppm_replace(conv_out[-1])
        f = psp_out
        fpn_feature_list = [psp_out]
        out = []
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)
            f = f = nn.functional.interpolate(f, size=conv_x.shape[-2:], mode='bilinear', align_corners=True)
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]

        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=True))
        fusion_out = fusion_list[0] + fusion_list[1] + fusion_list[2] + fusion_list[3]

        x = self.conv_last(fusion_out)

        return x


''' 
class decoder_RDDNet_cat(nn.Module):
    def __init__(self):
        super().__init__()
        self.x4 = ConvBnAct(48,8,1)
        self.x8 = ConvBnAct(128,128,1)
        self.x16 = ConvBnAct(256,120,1)
        self.x32 = ConvBnAct(256,128,1)
        self.conv4 = ConvBnAct(128,128,3,1,1,groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,1,groups=128)
        self.convlast = ConvBnAct(128,64,1)
        self.classer = nn.Conv2d(64,19,1)
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.cat((x4,x16),dim=1))
        x8 = self.conv8(torch.add(x8,x32))
        x4 = x4+ F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.classer(self.convlast(x4))
        return x4

class decoder_RDDNet_add(nn.Module):
    def __init__(self):
        super().__init__()
        self.x4 = ConvBnAct(48,128,1)
        self.x8 = ConvBnAct(128,128,1)
        self.x16 = ConvBnAct(256,128,1)
        self.x32 = ConvBnAct(256,128,1)
        self.conv4 = ConvBnAct(128,128,3,1,1,groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,1,groups=128)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        x8 = self.conv8(torch.add(x8,x32))
        x4 = x4+ F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.classer(self.convlast(x4))
        return x4
'''
if __name__ == "__main__":
    net = AlignNetCnet(19)
    fin = torch.randn(8, 3, 1024, 2048)
    net.eval()
    res = net(fin)
    print(res.shape)

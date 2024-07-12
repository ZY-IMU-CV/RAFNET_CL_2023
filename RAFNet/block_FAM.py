import torch
import torch.nn as nn
import torch.nn.functional as F
def activation():
    return nn.ReLU(inplace=True)
def norm2d(out_channels):
    return nn.BatchNorm2d(out_channels)
class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, apply_act=True):
        super(ConvBnAct, self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias)
        self.bn=norm2d(out_channels)
        if apply_act:
            self.act=activation()
        else:
            self.act=None
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x=self.act(x)
        return x
class AlignedModule(nn.Module):

    def __init__(self, inplane, outplane, kernel_size=(3,3),padding = 1):
        super(AlignedModule, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, (1,1), bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, (1,1), bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 2, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        low_feature, h_feature = x
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature= self.down_h(h_feature)
        h_feature = F.upsample(h_feature, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output

class decoder_FAM(nn.Module):
    def __init__(self,num_classes,channels):
        super(decoder_FAM,self).__init__()
        channels4, channels8, channels16, channels32 = channels["4"], channels["8"], channels["16"], channels["32"]
        self.Cat32_16 = AlignedModule(256,128)
        self.Cat16_8 = AlignedModule(128,64)
        self.Cat8_4 = AlignedModule(48,32)
        self.conv16 = ConvBnAct(channels16, 128, 1)
        self.conv8 = ConvBnAct(channels8, 48, 1)
        self.conv4 = ConvBnAct(channels4, 32, 1)
        self.classifier = nn.Conv2d(32, num_classes, 1)
    def forward(self, x):
        x4, x8, x16,x32=x["4"], x["8"],x["16"],x["32"]
        x16 = self.Cat32_16([x32,x16])
        x16 = self.conv16(x16)
        x8 = self.Cat16_8([x16,x8])
        x8 = self.conv8(x8)
        x4 = self.Cat8_4([x8,x4])
        x4 = self.conv4(x4)
        x4 = self.classifier(x4)
        return x4

import time
if __name__ == "__main__":
    x32 = torch.randn(1,256,32,64)
    x16 = torch.randn(1,256,64,128)
    x8 =  torch.randn(1, 128, 128, 256)
    x4 = torch.randn(1, 48, 256, 512)
    fin = {"4":x4,"8":x8,"16":x16,"32":x32}

    net = decoder_FAM(19,{"4":48,"8":128,"16":256,"32":256})
    #net2 = down32_Decoder0(19,{"4":48,"8":128,"16":256,"32":256})
    torch.cuda.synchronize()
    start_time = time.time()

    res = net(fin)
    torch.cuda.synchronize()
    end_tiem = time.time()
    time_sum = end_tiem - start_time
    print(time_sum)

    print(res.shape)
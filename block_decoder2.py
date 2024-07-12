from torch import nn
import torch
from torch.nn import functional as F
from block_operate import AlignedModule
from blocks import *
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
        if self.act is not None:
            x=self.act(x)
        return x
'''
class decoder_RDDNet_add(nn.Module):
    def __init__(self):
        super().__init__()
        self.x4 = ConvBnAct(48,256,1)
        self.x8 = ConvBnAct(128,256,1)
        self.x16 = ConvBnAct(256,256,1)
        self.x32 = ConvBnAct(256,256,1)
        self.conv4 = ConvBnAct(256,256,3,1,1,groups=256)
        self.conv8 = ConvBnAct(256,256,3,1,1,groups=256)
        self.convlast = ConvBnAct(256,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        x8 = self.conv8(torch.add(x32,x8))
        x4 = x4+ F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.classer(self.convlast(x4))
        return x4
'''
class decoder_RDDNet_add(nn.Module):
    def __init__(self):
        super().__init__()
        self.x4 = ConvBnAct(48,128,1)
        self.x8 = ConvBnAct(128,128,1)
        self.x16 = ConvBnAct(256,128,1)
        self.x32 = ConvBnAct(256,128,1)
        self.conv4 = ConvBnAct(128,128,3,1,1,groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        x8 = self.conv8(torch.add(x32,x8))
        x4 = x4+ F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.classer(self.convlast(x4))
        return x4
        
class decoder_RDDNet_add256(nn.Module):
    def __init__(self):
        super().__init__()
        self.x4 = ConvBnAct(48,256,1)
        self.x8 = ConvBnAct(128,256,1)
        self.x16 = ConvBnAct(256,256,1)
        self.x32 = ConvBnAct(256,256,1)
        self.conv4 = ConvBnAct(256,256,3,1,1,groups=128)
        self.conv8 = ConvBnAct(256,256,3,1,1,groups=128)
        self.convlast = ConvBnAct(256,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        x8 = self.conv8(torch.add(x32,x8))
        x4 = x4+ F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.classer(self.convlast(x4))
        return x4
class DDDDBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilations,group_width, stride):
        super().__init__()
        self.stride = stride
        avg_downsample=True
        groups=out_channels//group_width
        self.conv1=nn.Conv2d(in_channels, out_channels,kernel_size=1,bias=True)
        self.bn1=norm2d(out_channels)
        self.act1=activation()
        dilation=dilations[0]
        self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=(1,1),groups=out_channels, padding=dilation,dilation=dilation,bias=True)
        self.bn2=norm2d(out_channels)
        self.act2=activation()
        self.conv3=nn.Conv2d(out_channels, out_channels,kernel_size=1,bias=True)
        self.bn3=norm2d(out_channels)
        self.act3=activation()
        self.avg = nn.AvgPool2d(2,2,ceil_mode=True)
    def forward(self, x):
        x=self.conv1(x)
        x=self.bn1(x)
        x = self.act1(x)
        if self.stride == 2:
            x=self.avg(x)
        x=self.conv2(x)+x
        x=self.bn2(x)
        x=self.act2(x)
        x=self.conv3(x)
        x=self.bn3(x)
        x = self.act3(x)
        return x        
class decoder_RDDNet_add_RCD(nn.Module): 
    def __init__(self):
        super().__init__()
        self.x4 = ConvBnAct(48,128,1)
        self.x8 = ConvBnAct(128,128,1)
        self.x16 = ConvBnAct(256,128,1)
        self.x32 = ConvBnAct(256,128,1)
        self.conv4 = DDDDBlock(128,128,[1],1,1)
        self.conv8 = DDDDBlock(128,128,[1],1,1)
        self.convlast = DDDDBlock(128,32,[1],1,1)
        self.classer = nn.Conv2d(32,19,1)
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        x8 = self.conv8(torch.add(x32,x8))
        x4 = x4+ F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.classer(self.convlast(x4))
        return x4

class decoder_RDDNet_addcat(nn.Module):
    def __init__(self):
        super().__init__()
        self.x4 = ConvBnAct(48,128,1)
        self.x8 = ConvBnAct(128,128,1)
        self.x16 = ConvBnAct(256,128,1)
        self.x32 = ConvBnAct(256,128,1)
        self.conv4 = ConvBnAct(128,128,3,1,1,groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(256,32,3,1,1)
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
        x4 = torch.cat((x4, F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)),dim=1)
        x4 = self.classer(self.convlast(x4))
        return x4
class decoder_RDDNet_catadd(nn.Module):
    def __init__(self):
        super().__init__()
        self.x4 = ConvBnAct(48,128,1)
        self.x8 = ConvBnAct(128,128,1)
        self.x16 = ConvBnAct(256,128,1)
        self.x32 = ConvBnAct(256,128,1)
        self.conv4 = ConvBnAct(256,256,3,1,1,groups=256)
        self.conv8 = ConvBnAct(256,256,3,1,1,groups=256)
        self.convlast = ConvBnAct(256,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.cat((x4,x16),dim=1))
        x8 = self.conv8(torch.cat((x8,x32),dim=1))
        x4 = x4+ F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.classer(self.convlast(x4))
        return x4
        
class decoder_RDDNet_add3x3(nn.Module):
    def __init__(self):
        super().__init__()
        self.x4 = ConvBnAct(48,128,1)
        self.x8 = ConvBnAct(128,128,1)
        self.x16 = ConvBnAct(256,128,1)
        self.x32 = ConvBnAct(256,128,1)
        self.conv4 = ConvBnAct(128,128,3,1,1,groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(128,64,3,1,1)
        self.classer = nn.Conv2d(64,19,1)
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
        
class decoder_RDDNet_add64(nn.Module):
    def __init__(self):
        super().__init__()
        self.x4 = ConvBnAct(48,64,1)
        self.x8 = ConvBnAct(128,64,1)
        self.x16 = ConvBnAct(256,64,1)
        self.x32 = ConvBnAct(256,64,1)
        self.conv4 = ConvBnAct(64,64,3,1,1,groups=64)
        self.conv8 = ConvBnAct(64,64,3,1,1,groups=64)
        self.convlast = ConvBnAct(64,32,3,1,1)
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
        
class decoder_RDDNet_add32(nn.Module):
    def __init__(self):
        super().__init__()
        self.x4 = ConvBnAct(48,32,1)
        self.x8 = ConvBnAct(128,32,1)
        self.x16 = ConvBnAct(256,32,1)
        self.x32 = ConvBnAct(256,32,1)
        self.conv4 = ConvBnAct(32,32,3,1,1,groups=32)
        self.conv8 = ConvBnAct(32,32,3,1,1,groups=32)
        self.convlast = ConvBnAct(32,32,3,1,1)
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
        
class decoder_RDDNet_addcommon(nn.Module):
    def __init__(self):
        super().__init__()
        self.x4 = ConvBnAct(48,128,1)
        self.x8 = ConvBnAct(128,128,1)
        self.x16 = ConvBnAct(256,128,1)
        self.x32 = ConvBnAct(256,128,1)
        self.conv4 = ConvBnAct(128,128,3,1,1,groups=1)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=1)
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
        
class decoder_RDDNet_addcommon1x1(nn.Module):
    def __init__(self):
        super().__init__()
        self.x4 = ConvBnAct(48,128,1)
        self.x8 = ConvBnAct(128,128,1)
        self.x16 = ConvBnAct(256,128,1)
        self.x32 = ConvBnAct(256,128,1)
        self.conv4 = ConvBnAct(128,128,1,groups=1)
        self.conv8 = ConvBnAct(128,128,1,groups=1)
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
        
class decoder_RDDNet_addFAM(nn.Module): 
    def __init__(self):
        super().__init__()
        self.x4 = ConvBnAct(48,128,1)
        self.x8 = ConvBnAct(128,128,1)
        self.x16 = ConvBnAct(256,128,1)
        self.x32 = ConvBnAct(256,128,1)
        self.conv4 = ConvBnAct(128,128,1,groups=1)
        self.conv8 = ConvBnAct(128,128,1,groups=1)
        self.convlast = ConvBnAct(128,64,1)
        self.classer = nn.Conv2d(64,19,1)
        self.FAM1 = AlignedModule(inplane=128, outplane=128)
        self.FAM2 = AlignedModule(inplane=128, outplane=128)
        self.FAM3 = AlignedModule(inplane=128, outplane=128)
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x16 = self.FAM1([x4,x16])
        x32 = self.FAM2([x8,x32])
        x4 = self.conv4(torch.add(x4,x16))
        x8 = self.conv8(torch.add(x8,x32))
        x4 = x4+self.FAM3([x4,x8])
        x4 = self.classer(self.convlast(x4))
        return x4
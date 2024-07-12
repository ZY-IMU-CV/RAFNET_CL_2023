from torch import nn
import torch
from torch.nn import functional as F

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
        
class inputlayer(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(inputlayer, self).__init__()
        self.avg = nn.AvgPool2d(2,2,ceil_mode=True)
        self.conv=nn.Conv2d(in_channels,out_channels,3,1,1)
        self.bn=norm2d(out_channels)
        self.act=activation()
    def forward(self, x):
        x= self.avg(x)
        x = self.conv(x)
        x = self.bn(x)
        x=self.act(x)
        return x
        
class DWConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, apply_act=True):
        super(DWConvBnAct, self).__init__()
        self.avg = nn.AvgPool2d(2,2,ceil_mode=True)
        self.stride=stride
        self.conv3x3=nn.Conv2d(in_channels,in_channels,3,stride=1,padding=padding,groups=in_channels)
        self.conv1x1=nn.Conv2d(in_channels,out_channels,1)
        self.bn=norm2d(out_channels)
        if apply_act:
            self.act=activation()
        else:
            self.act=None
    def forward(self, x):
        if self.stride ==2:
            x= self.avg(x)
        x = self.conv3x3(x)
        x = self.conv1x1(x)
        x = self.bn(x)
        if self.act is not None:
            x=self.act(x)
        return x
class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilations,group_width, stride):
        super().__init__()
        self.stride = stride
        avg_downsample=True
        groups=out_channels//group_width
        self.conv1=nn.Conv2d(in_channels, out_channels,kernel_size=1,bias=False)
        self.bn1=norm2d(out_channels)
        self.act1=activation()
        dilation=dilations[0]
        self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=(1,1),groups=out_channels, padding=dilation,dilation=dilation,bias=False)
        self.bn2=norm2d(out_channels)
        self.act2=activation()
        self.conv3=nn.Conv2d(out_channels, out_channels,kernel_size=1,bias=False)
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

class CDBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, stride):
        super().__init__()
        self.stride = stride
        dilation=dilations[0]

        self.conv1=nn.Conv2d(in_channels, out_channels,kernel_size=1,bias=False)
        self.bn1=norm2d(out_channels)
        self.act1=activation()
        self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=1,groups=out_channels, padding=dilation,dilation=dilation,bias=False)
        self.bn2=norm2d(out_channels)
        self.act2=activation()
        self.avg = nn.AvgPool2d(2,2,ceil_mode=True)
    def forward(self, x):
        x=self.conv1(x)
        x=self.bn1(x)
        x = self.act1(x)
        if self.stride == 2:
            x=self.avg(x) 
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.act2(x)
        return x
'''
class CDBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, stride):
        super().__init__()
        self.stride = stride
        dilation=dilations[0]
        
        #self.conv1=nn.Conv2d(in_channels, out_channels,kernel_size=1,bias=False)
        #self.bn1=norm2d(out_channels)
        #self.act1=activation()
        self.conv2=nn.Conv2d(in_channels, in_channels,kernel_size=3,stride=1,groups=in_channels, padding=dilation,dilation=dilation,bias=False)
        #self.bn2=norm2d(in_channels)
        #self.act2=activation()
        self.conv3=nn.Conv2d(in_channels, out_channels,groups=1,kernel_size=1,bias=False)
        self.bn3=norm2d(out_channels)
        self.act3=activation()
        self.avg = nn.AvgPool2d(2,2,ceil_mode=True)
    def forward(self, x):
        if self.stride == 2:
            x=self.avg(x)
        #x=self.conv1(x)
        #x=self.bn1(x)
        #x = self.act1(x)
        x=self.conv2(x)
        #x=self.bn2(x)
        #x=self.act2(x)
        x=self.conv3(x)
        x=self.bn3(x)
        x = self.act3(x)
        return x
'''
class SEModule(nn.Module):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, Act, FC, Sigmoid."""
    def __init__(self, w_in, w_se):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1=nn.Conv2d(w_in, w_se, 1, bias=True)
        self.act1=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(w_se, w_in, 1, bias=True)
        self.act2=nn.Sigmoid()

    def forward(self, x):
        y=self.avg_pool(x)
        y=self.act1(self.conv1(y))
        y=self.act2(self.conv2(y))
        return x * y

class Shortcut(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, avg_downsample=False):
        super(Shortcut, self).__init__()
        if avg_downsample and stride != 1:
            self.avg=nn.AvgPool2d(2,2,ceil_mode=True)
            self.conv=nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.bn=nn.BatchNorm2d(out_channels)
        else:
            self.avg=None
            self.conv=nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            self.bn=nn.BatchNorm2d(out_channels)
    def forward(self, x):
        if self.avg is not None:
            x=self.avg(x)
        x = self.conv(x)
        x = self.bn(x)
        return x

class DilatedConv(nn.Module):
    def __init__(self,w,dilations,group_width,stride,bias):
        super().__init__()
        num_splits=len(dilations)
        assert(w%num_splits==0)
        temp=w//num_splits
        assert(temp%group_width==0)
        groups=temp//group_width
        convs=[]
        for d in dilations:
            convs.append(nn.Conv2d(temp,temp,3,padding=d,dilation=d,stride=stride,bias=bias,groups=groups))
        self.convs=nn.ModuleList(convs)
        self.num_splits=num_splits
    def forward(self,x):
        x=torch.tensor_split(x,self.num_splits,dim=1)
        res=[]
        for i in range(self.num_splits):
            res.append(self.convs[i](x[i]))
        return torch.cat(res,dim=1)

class ConvBnActConv(nn.Module):
    def __init__(self,w,stride,dilation,groups,bias):
        super().__init__()
        self.conv=ConvBnAct(w,w,3,stride,dilation,dilation,groups)
        self.project=nn.Conv2d(w,w,1,bias=bias)
    def forward(self,x):
        x=self.conv(x)
        x=self.project(x)
        return x


class YBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation,group_width, stride):
        super(YBlock, self).__init__()
        groups = out_channels // group_width
        self.conv1=nn.Conv2d(in_channels, out_channels,kernel_size=1,bias=False)
        self.bn1=norm2d(out_channels)
        self.act1=activation()
        self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=stride,groups=groups, padding=dilation,dilation=dilation,bias=False)
        self.bn2=norm2d(out_channels)
        self.act2=activation()
        self.conv3=nn.Conv2d(out_channels, out_channels,kernel_size=1,bias=False)
        self.bn3=norm2d(out_channels)
        self.act3=activation()
        self.se=SEModule(out_channels,in_channels//4)
        if stride != 1 or in_channels != out_channels:
            self.shortcut=Shortcut(in_channels,out_channels,stride)
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut=self.shortcut(x) if self.shortcut else x
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.act1(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.act2(x)
        x=self.se(x)
        x=self.conv3(x)
        x=self.bn3(x)
        x = self.act3(x + shortcut)
        return x

class DilaBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilations,group_width, stride,attention="se"):
        super().__init__()
        avg_downsample=True
        groups=out_channels//group_width
        self.conv1=nn.Conv2d(in_channels, out_channels,kernel_size=1,bias=False)
        self.bn1=norm2d(out_channels)
        self.act1=activation()
        if len(dilations)==1:
            dilation=dilations[0]
            self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=stride,groups=groups, padding=dilation,dilation=dilation,bias=False)
        else:
            self.conv2=DilatedConv(out_channels,dilations,group_width=group_width,stride=stride,bias=False)
        self.bn2=norm2d(out_channels)
        self.act2=activation()
        self.conv3=nn.Conv2d(out_channels, out_channels,kernel_size=1,bias=False)
        self.bn3=norm2d(out_channels)
        self.act3=activation()
        if attention=="se":
            self.se=SEModule(out_channels,in_channels//4)
        elif attention=="se2":
            self.se=SEModule(out_channels,out_channels//4)
        else:
            self.se=None
        if stride != 1 or in_channels != out_channels:
            self.shortcut=Shortcut(in_channels,out_channels,stride,avg_downsample)
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut=self.shortcut(x) if self.shortcut else x
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.act1(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.act2(x)
        if self.se is not None:
            x=self.se(x)
        x=self.conv3(x)
        x=self.bn3(x)
        x = self.act3(x + shortcut)
        return x

class eleven_Decoder0(nn.Module):#Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16=channels["4"],channels["8"],channels["16"]
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 8, 1)
        self.conv8=ConvBnAct(128,64,3,1,1)
        self.conv4=ConvBnAct(64+8,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x4, x8, x16=x["4"], x["8"],x["16"]
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4

class eleven_Decoder1(nn.Module):#Sum+Sum
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16=channels["4"],channels["8"],channels["16"]
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 64, 1)
        self.conv8=ConvBnAct(128,64,3,1,1)
        self.conv4=ConvBnAct(64,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x4, x8, x16=x["4"], x["8"],x["16"]
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4= x8+x4
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4

class eleven_Decoder2(nn.Module):#Cat+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16=channels["4"],channels["8"],channels["16"]
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 8, 1)
        self.conv8=ConvBnAct(256,64,3,1,1)
        self.conv4=ConvBnAct(64+8,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x4, x8, x16=x["4"], x["8"],x["16"]
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= torch.cat((x8,x16),dim=1)
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4

class LRASPP(nn.Module):
    def __init__(self, num_classes, channels, inter_channels=128):
        super().__init__()
        channels8, channels16 = channels["8"], channels["16"]
        self.cbr = ConvBnAct(channels16, inter_channels, 1)
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels16, inter_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.low_classifier = nn.Conv2d(channels8, num_classes, 1)
        self.high_classifier = nn.Conv2d(inter_channels, num_classes, 1)

    def forward(self, x):
        # intput_shape=x.shape[-2:]
        x8, x16 = x["8"], x["16"]
        x = self.cbr(x16)
        s = self.scale(x16)
        x = x * s
        x = F.interpolate(x, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x = self.low_classifier(x8) + self.high_classifier(x)
        return x
'''
class down32_Decoder0(nn.Module):#Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16,channels32=channels["4"],channels["8"],channels["16"],channels["32"]
        self.head32 = ConvBnAct(channels32, 128, 1)
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 8, 1)
        self.conv16 = DWConvBnAct(128,128,3,1,1)
        self.conv8=DWConvBnAct(128,64,3,1,1)
        self.conv4=DWConvBnAct(64+8,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)
    def forward(self, x):
        x4, x8, x16,x32=x["4"], x["8"],x["16"],x["32"]
        x32=self.head32(x32)
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x16 = x16+x32
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4
''' 
'''
class down32_Decoder0(nn.Module):#Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16,channels32=channels["4"],channels["8"],channels["16"],channels["32"]
        self.head32 = ConvBnAct(channels32, 128, 1)
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 128, 1)
        self.conv16 = ConvBnAct(128,64,3,1,1)
        self.conv8=ConvBnAct(64,64,3,1,1)
        self.conv4=DWConvBnAct(128,32,3,1,1)
        self.classifier=nn.Conv2d(32, num_classes, 1)
    def forward(self, x):
        x4, x8, x16,x32=x["4"], x["8"],x["16"],x["32"]
        x32=self.head32(x32)
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x16 = x16+x32
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4
'''  
class down32_Decoder0(nn.Module):#Sum+Cat 
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16,channels32=channels["4"],channels["8"],channels["16"],channels["32"]
        self.head32 = ConvBnAct(channels32, 128, 1)
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 128, 1)
        self.conv16 = ConvBnAct(128,128,3,1,1,groups=128)
        self.conv8=ConvBnAct(128,128,3,1,1,groups=128)
        self.conv4=ConvBnAct(128,32,3,1,1)
        self.classifier=nn.Conv2d(32, num_classes, 1)
    def forward(self, x):
        x4, x8, x16,x32=x["4"], x["8"],x["16"],x["32"]
        x32=self.head32(x32)
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x16 = x16+x32
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.add(x8,x4)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4

def generate_stage2(ds,block_fun):
    blocks=[]
    for d in ds:
        blocks.append(block_fun(d))
    return blocks

class RegSegBody(nn.Module):
    def __init__(self,ds):
        super().__init__()
        gw=16
        attention="se"
        self.stage4=DBlock(32, 48, [1], gw, 2, attention)
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2, attention),
            DBlock(128, 128, [1], gw, 1, attention),
            DBlock(128, 128, [1], gw, 1, attention)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2, attention),
            *generate_stage2(ds[:-1], lambda d: DBlock(256, 256, d, gw, 1, attention)),
            DBlock(256, 256, ds[-1], gw, 1, attention)
        )
    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        return {"4":x4,"8":x8,"16":x16}
    def channels(self):
        return {"4":48,"8":128,"16":256}
class RegSegBody2(nn.Module):
    def __init__(self,ds):
        super().__init__()
        gw=16
        self.stage4=DBlock(32, 48, [1], gw, 2)
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            *generate_stage2(ds[:-1], lambda d: DBlock(256, 256, d, gw, 1)),
            DBlock(256, 256, ds[-1], gw, 1)
        )
        self.stage32 = DBlock(256, 256, [16], gw, 2)
    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        
class RegSegBody_1(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=DBlock(32, 48, [1], gw, 2)
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1),
            DBlock(256, 256, [4], gw, 1),
            DBlock(256, 256, [8], gw, 1),
            DBlock(256, 256, [8], gw, 1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        return {"4":x4,"8":x8,"16":x16}
    def channels(self):
        return {"4":48,"8":128,"16":256}

class RegSegBody_2(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=DBlock(32, 48, [1], gw, 2)
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[3],gw,1),
            DBlock(256, 256, [6], gw, 1),
            DBlock(256, 256, [6], gw, 1),
            DBlock(256, 256, [9], gw, 1),
            DBlock(256, 256, [9], gw, 1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        return {"4":x4,"8":x8,"16":x16}
    def channels(self):
        return {"4":48,"8":128,"16":256}

class RegSegBody_3(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=DBlock(32, 48, [1], gw, 2)
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [4], gw, 2),
            DBlock(256, 256, [8], gw, 1),
            DBlock(256, 256, [12], gw, 1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}


class RegSegBody_4(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=DBlock(32, 48, [1], gw, 2)
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256, 256, [4], gw, 1),
            DBlock(256, 256, [8], gw, 1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}


class RegSegBody_5(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=DBlock(32, 48, [1], gw, 2)
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[3],gw,1),
            DBlock(256, 256, [6], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [6], gw, 2),
            DBlock(256, 256, [9], gw, 1),
            DBlock(256, 256, [12], gw, 1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}


class RegSegBody_6(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=DBlock(32, 48, [1], gw, 2)
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[3],gw,1),
            DBlock(256, 256, [6], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256, 256, [6], gw, 1),
            DBlock(256, 256, [9], gw, 1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_7(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48, 48, [1], gw, 1)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[1],gw,1),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1),
            DBlock(256, 256, [4], gw, 1),
            DBlock(256, 256, [8], gw, 1),
            DBlock(256, 256, [8], gw, 1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        return {"4":x4,"8":x8,"16":x16}
    def channels(self):
        return {"4":48,"8":128,"16":256}

class RegSegBody_8(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48, 48, [1], gw, 1)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[1],gw,1),
            DBlock(256,256,[3],gw,1),
            DBlock(256,256,[3],gw,1),
            DBlock(256, 256, [6], gw, 1),
            DBlock(256, 256, [6], gw, 1),
            DBlock(256, 256, [9], gw, 1),
            DBlock(256, 256, [9], gw, 1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        return {"4":x4,"8":x8,"16":x16}
    def channels(self):
        return {"4":48,"8":128,"16":256}

class RegSegBody_9(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48, 48, [1], gw, 1)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1),
            DBlock(256, 256, [8], gw, 1)
        )
        self.stage32=nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1),
            DBlock(256, 256, [8], gw, 1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        
class RegSegBody_10(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48, 48, [1], gw, 1)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[3],gw,1),
            DBlock(256, 256, [6], gw, 1),
            DBlock(256, 256, [9], gw, 1)
        )
        self.stage32=nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256,256,[3],gw,1),
            DBlock(256, 256, [6], gw, 1),
            DBlock(256, 256, [9], gw, 1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}


class RegSegBody_11(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48, 48, [1], gw, 1),
            DBlock(48, 48, [1], gw, 1)
            )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[1],gw,1),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[4],gw,1),
            DBlock(256,256,[4],gw,1),
            DBlock(256,256,[4],gw,1),
            DBlock(256,256,[8],gw,1),
            DBlock(256, 256, [8], gw, 1)
        )
    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        return {"4":x4,"8":x8,"16":x16}
    def channels(self):
        return {"4":48,"8":128,"16":256}

class RegSegBody_12(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48, 48, [1], gw, 1),
            DBlock(48, 48, [1], gw, 1)
            )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[1],gw,1),
            DBlock(256,256,[3],gw,1),
            DBlock(256,256,[3],gw,1),
            DBlock(256,256,[3],gw,1),
            DBlock(256,256,[6],gw,1),
            DBlock(256,256,[6],gw,1),
            DBlock(256,256,[6],gw,1),
            DBlock(256,256,[9],gw,1),
            DBlock(256, 256, [9], gw, 1)
        )
    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        return {"4":x4,"8":x8,"16":x16}
    def channels(self):
        return {"4":48,"8":128,"16":256}
        
class RegSegBody_13(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48, 48, [1], gw, 1),
            DBlock(48, 48, [1], gw, 1)
            )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[4],gw,1),
            DBlock(256,256,[4],gw,1),
            DBlock(256, 256, [8], gw, 1)
        )
        self.stage32=nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[4],gw,1),
            DBlock(256,256,[4],gw,1),
            DBlock(256, 256, [8], gw, 1)
        )
    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
    
    
class RegSegBody_14(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48, 48, [1], gw, 1),
            DBlock(48, 48, [1], gw, 1)
            )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[3],gw,1),
            DBlock(256,256,[6],gw,1),
            DBlock(256,256,[6],gw,1),
            DBlock(256, 256, [9], gw, 1)
        )
        self.stage32=nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256,256,[3],gw,1),
            DBlock(256,256,[6],gw,1),
            DBlock(256,256,[6],gw,1),
            DBlock(256, 256, [9], gw, 1)
        )
    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
    
class RegSegBody_test(nn.Module):
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
            DBlock(256, 256, [1], gw, 1)
        )
        self.stage32_10 = DBlock(256, 128, [2], gw, 2)
        self.stage32_11 = DBlock(256, 128, [4], gw, 1)
        self.stage32_12 = DBlock(256, 128, [8], gw, 1)

        self.stage32_20 = DBlock(256, 128, [1], gw, 2)
        self.stage32_21 = DBlock(256, 128, [1], gw, 1)
        self.stage32_22 = DBlock(256, 128, [1], gw, 1)

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32 = torch.cat((self.stage32_20(x16) , self.stage32_10(x16)),dim=1)
        x32 = torch.cat((self.stage32_21(x32) , self.stage32_11(x32)),dim=1)
        x32 = torch.cat((self.stage32_22(x32) , self.stage32_12(x32)),dim=1)
        return {"4": x4, "16": x16, "32":x32}

    def channels(self):
        return {"4": 48, "16": 256,"32":256}
        
class down32_test(nn.Module):#Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels32=channels["4"],channels["8"],channels["32"]
        self.head32 = ConvBnAct(channels32, 128, 1)
        #self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 8, 1)
        #self.conv16 = ConvBnAct(128,64,3,1,1)
        self.conv8=ConvBnAct(128,64,3,1,1)
        self.conv4=ConvBnAct(64+8,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)
        self.gloable_pool = nn.AdaptiveAvgPool2d((1,1));
    def forward(self, x):
        x4, x8,x32=x["4"], x["8"],x["32"]
        x32=self.head32(x32)
        #x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        #x16 = x16+x32
        #x16 = self.conv16(x16)
        #x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x8 + x32
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4

class CDblock(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage4=CDBlock(32, 48, [1], 2)
        self.stage8=nn.Sequential(
            CDBlock(48, 128, [1],2),
            CDBlock(128, 128, [1],1)
        )
        self.stage16=nn.Sequential(
            CDBlock(128, 256, [1],2),
            CDBlock(256,256,[2],1),
            CDBlock(256, 256, [4],1)
        )
        self.stage32 = nn.Sequential(
            CDBlock(256, 256, [1], 2),
            CDBlock(256,256,[2],1),
            CDBlock(256, 256, [4], 1),
            CDBlock(256, 256, [8], 1)
        )
    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_15(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=DBlock(32, 48, [1], gw, 2)
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1),
            DBlock(256, 256, [8], gw, 1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        
class RegSegBody_16(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=DBlock(32, 48, [1], gw, 2)
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256, 256, [4], gw, 1),
            DBlock(256, 256, [8], gw, 1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        
class RegSegBody_17(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256,256,[4],gw,1),
            DBlock(256, 256, [8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        
        
class RegSegBody_17Y(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            YBlock(32, 48, [1], gw, 2),
            YBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            YBlock(48, 128, [1], gw, 2),
            YBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            YBlock(128, 256, [1], gw, 2),
            YBlock(256,256,[2],gw,1),
            YBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            YBlock(256, 256, [1], gw, 2),
            YBlock(256,256,[4],gw,1),
            YBlock(256, 256, [8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        
class RegSegBody_17D(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DilaBlock(32, 48, [1], gw, 2),
            DilaBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DilaBlock(48, 128, [1], gw, 2),
            DilaBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DilaBlock(128, 256, [1], gw, 2),
            DilaBlock(256,256,[1,2],gw,1),
            DilaBlock(256, 256, [1,4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DilaBlock(256, 256, [1], gw, 2),
            DilaBlock(256,256,[1,4],gw,1),
            DilaBlock(256, 256, [1,8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        
class RegSegBody_18(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256,256,[4],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        
class RegSegBody_19(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256,256,[8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
class RegSegBody_20(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48,48,[1],gw,1),
            DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256,256,[4],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        
class RegSegBody_21(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256, 256, [2], gw, 1),
            DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256, 256, [4], gw, 1),
            DBlock(256,256,[8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
 
class RegSegBody_22(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage160=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1)
        )
        self.stage161 = nn.Sequential(
            DBlock(256, 256, [1], gw, 1),
            DBlock(256,256,[4],gw,1),
            DBlock(256, 256, [8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x160=self.stage160(x8)
        x161 = self.stage161(x160)
        return {"4":x4,"8":x8,"16":x160,"32":x161}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        
class RegSegBody_23(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48,48,[1],gw,1)
        )
        self.stage80=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage81=nn.Sequential(
            DBlock(128, 256, [1], gw, 1),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1)
        )
        self.stage82 = nn.Sequential(
            DBlock(256, 256, [1], gw, 1),
            DBlock(256,256,[4],gw,1),
            DBlock(256, 256, [8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x80=self.stage80(x4)
        x81=self.stage81(x80)
        x82 = self.stage82(x81)
        return {"4":x4,"8":x80,"16":x81,"32":x82}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_24(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48,48,[1],gw,1),
            DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256,256,[4],gw,1),
            DBlock(256, 256, [8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}



class RegSegBody_25(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48,48,[1],gw,1),
            DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 1),
            DBlock(256,256,[1],gw,1),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[4],gw,1),
            DBlock(256,256,[4],gw,1),
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 1),
            DBlock(256,256,[1],gw,1),
            DBlock(256, 256, [4],gw,1),
            DBlock(256,256,[4],gw,1),
            DBlock(256,256,[8],gw,1),
            DBlock(256,256,[8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_26(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48,48,[1],gw,1),
            DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[1],gw,1),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[4],gw,1),
            DBlock(256,256,[4],gw,1),
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 1),
            DBlock(256,256,[1],gw,1),
            DBlock(256, 256, [4],gw,1),
            DBlock(256,256,[4],gw,1),
            DBlock(256,256,[8],gw,1),
            DBlock(256,256,[8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
class RegSegBody_27(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48,48,[1],gw,1),
            DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[1],gw,1),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[4],gw,1),
            DBlock(256,256,[4],gw,1),
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256,256,[1],gw,1),
            DBlock(256, 256, [4],gw,1),
            DBlock(256,256,[4],gw,1),
            DBlock(256,256,[8],gw,1),
            DBlock(256,256,[8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        
        
class RegSegBody_28(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48,48,[1],gw,1),
            DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[1],gw,1),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[4],gw,1),
            DBlock(256,256,[4],gw,1),
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256,256,[4],gw,1),
            DBlock(256,256,[8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        
class RegSegBody_29(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48,48,[1],gw,1),
            DBlock(48,48,[1],gw,1),
            DBlock(48,48,[1],gw,1),
            DBlock(48,48,[1],gw,1),
            DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[4],gw,1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256,256,[1],gw,1),
            DBlock(256,256,[4],gw,1),
            DBlock(256,256,[4],gw,1),
            DBlock(256,256,[8],gw,1),
            DBlock(256, 256, [8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        
class RegSegBody_30(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48,48,[1],gw,1),
            DBlock(48,48,[1],gw,1),
            DBlock(48,48,[1],gw,1),
            DBlock(48,48,[1],gw,1),
            DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[4],gw,1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256,256,[4],gw,1),
            DBlock(256, 256, [8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_addallshort(nn.Module):
    def __init__(self):
        super().__init__()
        gw = 16
        self.stage4 = DBlock_noshort(32, 48, [1], gw, 2)
        self.stage8 = nn.Sequential(
            DBlock_addallshort(48, 128, [1], gw, 2),
            DBlock_addallshort(128, 128, [1], gw, 1),
            DBlock_addallshort(128, 128, [1], gw, 1)
        )
        self.stage16 = nn.Sequential(
            DBlock_addallshort(128, 256, [1], gw, 2),
            DBlock_addallshort(256, 256, [2], gw, 1),
            DBlock_addallshort(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DBlock_addallshort(256, 256, [1], gw, 2),
            DBlock_addallshort(256, 256, [4], gw, 1),
            DBlock_addallshort(256, 256, [8], gw, 1)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4": x4, "8": x8, "16": x16, "32": x32}

    def channels(self):
        return {"4": 48, "8": 128, "16": 256, "32": 256}
        

class RegSegBody_152(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=DBlock(32, 48, [1], gw, 2)
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 320, [1], gw, 2),
            DBlock(320,320,[2],gw,1),
            DBlock(320, 320, [4], gw, 1),
            DBlock(320, 320, [8], gw, 1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":320}

class RegSegBody_153(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=DBlock(32, 48, [1], gw, 2)
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1),
            DBlock(256, 256, [8], gw, 1)
        )
        self.short_x4 = ConvBnAct(48,128,1,stride=2)
        self.short_x8 = ConvBnAct(128,256,1,stride=2)
    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8+self.short_x4(x4))
        x32 = self.stage32(x16+self.short_x8(x8))
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}


class down32_Decoder1(nn.Module):#Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16,channels32=channels["4"],channels["8"],channels["16"],channels["32"]
        self.head32 = ConvBnAct(channels32, 128, 1)
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 8, 1)
        self.conv16 = ConvBnAct(128,128,3,1,1)
        self.conv8=ConvBnAct(128,64,3,1,1)
        self.conv4=ConvBnAct(64+8,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)
        self.gloable_pool = nn.AdaptiveAvgPool2d((1,1));
    def forward(self, x):
        x4, x8, x16,x32=x["4"], x["8"],x["16"],x["32"]
        x32=self.head32(x32)
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x16 = x16+x32
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4
        
class RegSegBody_repvgg(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=SBlock(32, 48, [1], gw, 2)
        self.stage8=nn.Sequential(
            SBlock(48, 128, [1], gw, 2),
            SBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            SBlock(128, 256, [1], gw, 2),
            SBlock(256,256,[2],gw,1),
            SBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            SBlock(256, 256, [1], gw, 2),
            SBlock(256,256,[2],gw,1),
            SBlock(256, 256, [4], gw, 1),
            SBlock(256, 256, [8], gw, 1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        
class down32_Decoder64(nn.Module):#Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16,channels32=channels["4"],channels["8"],channels["16"],channels["32"]
        self.head32 = ConvBnAct(channels32, 128, 1)
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 64, 1)
        self.conv16 = ConvBnAct(128,128,3,1,1)
        self.conv8=ConvBnAct(128,64,3,1,1)
        self.conv4=ConvBnAct(64+64,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)
    def forward(self, x):
        x4, x8, x16,x32=x["4"], x["8"],x["16"],x["32"]
        x32=self.head32(x32)
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x16 = x16+x32
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4
        
class down32_Decoder128(nn.Module):#Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16,channels32=channels["4"],channels["8"],channels["16"],channels["32"]
        self.head32 = ConvBnAct(channels32, 128, 1)
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 128, 1)
        self.conv16 = ConvBnAct(128,128,3,1,1)
        self.conv8=ConvBnAct(128,64,3,1,1)
        self.conv4=ConvBnAct(64+128,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)
    def forward(self, x):
        x4, x8, x16,x32=x["4"], x["8"],x["16"],x["32"]
        x32=self.head32(x32)
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x16 = x16+x32
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4
        
class down32_Decoder_cat(nn.Module):#cat+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16,channels32=channels["4"],channels["8"],channels["16"],channels["32"]
        self.head32 = ConvBnAct(channels32, 128, 1)
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 8, 1)
        self.conv16 = ConvBnAct(256,128,3,1,1)
        self.conv8=ConvBnAct(256,64,3,1,1)
        self.conv4=ConvBnAct(64+8,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)
    def forward(self, x):
        x4, x8, x16,x32=x["4"], x["8"],x["16"],x["32"]
        x32=self.head32(x32)
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x16 = torch.cat((x32,x16),dim=1)
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= torch.cat((x16,x8),dim=1)
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4
        
class down32_Decoder_sum(nn.Module):#sum+sum+sum
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16,channels32=channels["4"],channels["8"],channels["16"],channels["32"]
        self.head32 = ConvBnAct(channels32, 128, 1)
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 64, 1)
        self.conv16 = ConvBnAct(128,128,3,1,1)
        self.conv8=ConvBnAct(128,64,3,1,1)
        self.conv4=ConvBnAct(64,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)
    def forward(self, x):
        x4, x8, x16,x32=x["4"], x["8"],x["16"],x["32"]
        x32=self.head32(x32)
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x16 = x32+x16
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x16+x8
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4= x8+x4
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4
        
class down32_Decoderprocess64(nn.Module):#Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16,channels32=channels["4"],channels["8"],channels["16"],channels["32"]
        self.head32 = ConvBnAct(channels32, 64, 1)
        self.head16=ConvBnAct(channels16, 64, 1)
        self.head8=ConvBnAct(channels8, 64, 1)
        self.head4=ConvBnAct(channels4, 8, 1)
        self.conv16 = ConvBnAct(64,64,3,1,1)
        self.conv8=ConvBnAct(64,64,3,1,1)
        self.conv4=ConvBnAct(64+8,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)
    def forward(self, x):
        x4, x8, x16,x32=x["4"], x["8"],x["16"],x["32"]
        x32=self.head32(x32)
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x16 = x16+x32
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4
        
class down32_Decoder1x1(nn.Module):#Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16,channels32=channels["4"],channels["8"],channels["16"],channels["32"]
        self.head32 = ConvBnAct(channels32, 128, 1)
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 8, 1)
        self.conv16 = ConvBnAct(128,128,1)
        self.conv8=ConvBnAct(128,64,1)
        self.conv4=ConvBnAct(64+8,64,3,1,1) 
        self.classifier=nn.Conv2d(64, num_classes, 1)
    def forward(self, x):
        x4, x8, x16,x32=x["4"], x["8"],x["16"],x["32"]
        x32=self.head32(x32)
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x16 = x16+x32
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4
        
class down32_Decoderban(nn.Module):#Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16,channels32=channels["4"],channels["8"],channels["16"],channels["32"]
        self.head32 = ConvBnAct(channels32, 128, 1)
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 24, 1)
        self.conv16 = ConvBnAct(128,128,3,1,1)
        self.conv8=ConvBnAct(128,64,3,1,1)
        self.conv4=ConvBnAct(64+24,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)
    def forward(self, x):
        x4, x8, x16,x32=x["4"], x["8"],x["16"],x["32"]
        x32=self.head32(x32)
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x16 = x16+x32
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4
        
class down32_DecoderFAM(nn.Module):#Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16,channels32=channels["4"],channels["8"],channels["16"],channels["32"]
        self.head32 = ConvBnAct(channels32, 128, 1)
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 8, 1)
        self.conv16 = ConvBnAct(128,128,3,1,1)
        self.conv8=ConvBnAct(128,64,3,1,1)
        self.conv4=ConvBnAct(128+128+64+8,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)
    def forward(self, x):
        x4, x8, x16,x32=x["4"], x["8"],x["16"],x["32"]
        x32=self.head32(x32)
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x16 = x16+x32
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x32 = F.interpolate(x32, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x32,x16,x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4
        
import math
class h_swish(nn.Module):
    def __init__(self):
        super(h_swish, self).__init__()
        self.sigmoid = nn.ReLU(inplace=True)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )



class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks

        # building first laye
        #self.inputelayers = conv_3x3_bn(3, 32, 2)

        # building inverted residual blocks
        self.stage4 = InvertedResidual(32,48,48,3,2,1)
        self.stage8 = nn.Sequential(InvertedResidual(48,48,128,3,2,1),
                                    InvertedResidual(128,128,128,3,1,1))
        self.stage16 = nn.Sequential(InvertedResidual(128,128,128,3,2,1),
                                     InvertedResidual(128,128,128,3,1,1),
                                     InvertedResidual(128,128,128,3,1,1),
                                     InvertedResidual(128,128,128,3,1,1),
                                     InvertedResidual(128, 256, 256, 3, 1, 1),
                                     InvertedResidual(256,256,256,3,1,1))
        self.stage32 = nn.Sequential(InvertedResidual(256,256,256,3,2,1),
                                     InvertedResidual(256,256,256,3,1,1),
                                     InvertedResidual(256,256,256,3,1,1))

        # building last several layers
        #self.classifier = nn.Conv2d()
        self._initialize_weights()

    def forward(self, x):
        #x=self.inputelayers(x)
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32 = self.stage32(x16)
        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.classifier(x)
        return {"4":x4,"8":x8,"16":x16,"32":x32}

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
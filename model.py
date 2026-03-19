from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import torch.utils.model_zoo as model_zoo
from torchvision import models
import timm
from monai.networks.nets import SwinUNETR


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
    
    


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.rate = rate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, n_channels, block, layers, os=16, pretrained=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if os == 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
            blocks = [1, 2, 4]
        elif os == 8:
            strides = [1, 2, 1, 1]
            rates = [1, 1, 2, 2]
            blocks = [1, 2, 1]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], rate=rates[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], rate=rates[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], rate=rates[2])
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], rate=rates[3])

        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks=[1,2,4], stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate=blocks[0]*rate, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, rate=blocks[i]*rate))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

def ResNet101(n_channels=3, os=16, pretrained=False):
    model = ResNet(n_channels, Bottleneck, [3, 4, 23, 3], os, pretrained=pretrained)
    return model


class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabv3_plus(nn.Module):
    def __init__(self, n_channels=3, n_classes=21, os=16, pretrained=False, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(os))
            print("Number of Input Channels: {}".format(n_channels))
        super(DeepLabv3_plus, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        

        # Atrous Conv
        self.resnet_features = ResNet101(n_channels, os, pretrained=pretrained)

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, n_classes, kernel_size=1, stride=1))

    def forward(self, input):
        x, low_level_features = self.resnet_features(input)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.upsample(x, size=(int(math.ceil(input.size()[-2]/4)),
                                int(math.ceil(input.size()[-1]/4))), mode='bilinear', align_corners=True)

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)


        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = [model.resnet_features]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = [model.aspp1, model.aspp2, model.aspp3, model.aspp4, model.conv1, model.conv2, model.last_conv]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k




class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
        
class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class U_Net(nn.Module):
    def __init__(self,n_channels=1, n_classes=1):
        super(U_Net,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=n_channels,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,n_classes,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2U_Net(nn.Module):
    def __init__(self,n_channels=3,n_classes=1,t=2):
        super(R2U_Net,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=n_channels,ch_out=64,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

        self.Conv_1x1 = nn.Conv2d(64,n_classes,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1



class AttU_Net(nn.Module):
    def __init__(self,n_channels=3,n_classes=1):
        super(AttU_Net,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=n_channels,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,n_classes,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # print (x.shape)
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2AttU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,t=2):
        super(R2AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


############################################################################################
#LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation
#Paper-Link:   https://arxiv.org/pdf/1707.03718.pdf
############################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision.models import resnet



__all__ = ["LinkNet"]

class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size, 1, padding, groups=groups, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(out_planes),)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.relu(out+residual)

        return out


class Encoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(Encoder, self).__init__()
        self.block1 = BasicBlock(in_planes, out_planes, kernel_size, stride, padding, groups, bias)
        self.block2 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        return x


class Decoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        # TODO bias=True
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//4, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True))
        self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes//4, in_planes//4, kernel_size, stride, padding, output_padding, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes//4, out_planes, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(out_planes),
                                nn.ReLU(inplace=True))

    def forward(self, x_high_level, x_low_level):
        x = self.conv1(x_high_level)
        x = self.tp_conv(x)

        # solution for padding issues
        # diffY = x_low_level.size()[2] - x_high_level.size()[2]
        # diffX = x_low_level.size()[3] - x_high_level.size()[3]
        # x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = center_crop(x, x_low_level.size()[2], x_low_level.size()[3])

        x = self.conv2(x)

        return x

def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    diffy = (h - max_height) // 2
    diffx = (w -max_width) // 2
    return layer[:,:,diffy:(diffy + max_height),diffx:(diffx + max_width)]


def up_pad(layer, skip_height, skip_width):
    _, _, h, w = layer.size()
    diffy = skip_height - h
    diffx = skip_width -w
    return F.pad(layer,[diffx // 2, diffx - diffx // 2,
                        diffy // 2, diffy - diffy // 2])


def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src


class LinkNetImprove(nn.Module):
    """
    Generate Model Architecture
    """

    def __init__(self,n_channels, n_classes=19):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        

        base = resnet.resnet18(pretrained=True)

        self.in_block = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )

        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)

        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(32, n_classes, 2, 2, 0)
        self.cat1 = nn.Conv2d(256,1,3)
        self.cat2 = nn.Conv2d(128,1,3)
        self.cat3 = nn.Conv2d(64,1,3)
        self.cat4 = nn.Conv2d(5,1,3)
        


    def forward(self, x):
        # x = x.shape
        # Initial block
        x = self.in_block(x)

        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder blocks
        d4 = e3 + self.decoder4(e4, e3)
        d3 = e2 + self.decoder3(d4, e2)
        d2 = e1 + self.decoder2(d3, e1)
        d1 = x + self.decoder1(d2, x)

        # Classifier
        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)

        d4 = self.cat1(d4)
        d4 = F.interpolate(d4, size=(256, 256), mode='bicubic', align_corners=False)
        # print ('d4', d4.shape)
        d3 = self.cat2(d3)
        d3 = F.interpolate(d3, size=(256, 256), mode='bicubic', align_corners=False)
        # print ('d3', d3.shape)
        d2 = self.cat3(d2)
        d2 = F.interpolate(d2, size=(256, 256), mode='bicubic', align_corners=False)
        # print ('d2', d2.shape)
        d1 = self.cat3(d1)
        d1 = F.interpolate(d1, size=(256, 256), mode='bicubic', align_corners=False)
        # print ('d1', d1.shape)
        # print ('y', y.shape)
        y = torch.cat([y,d4,d3,d2,d1], dim=1)
        y = self.cat4(y)
        y = F.interpolate(y, size=(256, 256), mode='bicubic', align_corners=False)

        return y


class LinkNet(nn.Module):
    """
    Generate model architecture
    """

    def __init__(self, classes=19):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.encoder1 = Encoder(64, 64, 3, 1, 1)
        self.encoder2 = Encoder(64, 128, 3, 2, 1)
        self.encoder3 = Encoder(128, 256, 3, 2, 1)
        self.encoder4 = Encoder(256, 512, 3, 2, 1)


        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)


        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(32, classes, 2, 2, 0)

    def forward(self, x):
        # Initial block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder blocks
        d4 = e3 + self.decoder4(e4, e3)
        d3 = e2 + self.decoder3(d4, e2)
        d2 = e1 + self.decoder2(d3, e1)
        d1 = x + self.decoder1(d2, x)

        # Classifier
        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)

      


        return y

# ------------------------------------------------------------

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src


### RSU-7 ###
class RSU7(nn.Module):#UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool5 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = _upsample_like(hx6d,hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-6 ###
class RSU6(nn.Module):#UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)


        hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-5 ###
class RSU5(nn.Module):#UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4 ###
class RSU4(nn.Module):#UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4F ###
class RSU4F(nn.Module):#UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d + hxin


##### U^2-Net ####
class U2NET(nn.Module):

    def __init__(self,in_ch=3,out_ch=1):
        super(U2NET,self).__init__()

        self.stage1 = RSU7(in_ch,32,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64,32,128)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(128,64,256)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(256,128,512)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(512,256,512)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(512,256,512)

        # decoder
        self.stage5d = RSU4F(1024,256,512)
        self.stage4d = RSU4(1024,128,256)
        self.stage3d = RSU5(512,64,128)
        self.stage2d = RSU6(256,32,64)
        self.stage1d = RSU7(128,16,64)

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(512,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(6,out_ch,1)

    def forward(self,x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)

        #-------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)

### U^2-Net small ###
class U2NETP(nn.Module):

    def __init__(self,n_channels=3,n_classes=1):
        super(U2NETP,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.stage1 = RSU7(n_channels,16,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64,16,64)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(64,16,64)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(64,16,64)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(64,16,64)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(64,16,64)

        # decoder
        self.stage5d = RSU4F(128,16,64)
        self.stage4d = RSU4(128,16,64)
        self.stage3d = RSU5(128,16,64)
        self.stage2d = RSU6(128,16,64)
        self.stage1d = RSU7(128,16,64)

        self.side1 = nn.Conv2d(64,n_classes,3,padding=1)
        self.side2 = nn.Conv2d(64,n_classes,3,padding=1)
        self.side3 = nn.Conv2d(64,n_classes,3,padding=1)
        self.side4 = nn.Conv2d(64,n_classes,3,padding=1)
        self.side5 = nn.Conv2d(64,n_classes,3,padding=1)
        self.side6 = nn.Conv2d(64,n_classes,3,padding=1)

        self.outconv = nn.Conv2d(6,n_classes,1)

    def forward(self,x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)

        #decoder
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        return d0
        
#===============FCN============
class Upsample(nn.Module):
    def __init__(self, inplanes, planes):
        super(Upsample, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x, size):
        x = F.upsample_bilinear(x, size=size)
        x = self.conv1(x)
        x = self.bn(x)
        return x


class Fusion(nn.Module):
    def __init__(self, inplanes):
        super(Fusion, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=1)
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        out = self.bn(self.conv(x1)) + x2
        out = self.relu(out)

        return out


class FCN(nn.Module):
    def __init__(self,n_channels, n_classes):
        super(FCN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        

        resnet = models.resnet101(pretrained=True)
        #resnet.load_state_dict(torch.load('resnet101-5d3b4d8f.pth'))
        #resnet.load_state_dict('resnet101-5d3b4d8f.pth')

        self.conv1 = resnet.conv1
        self.bn0 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.upsample1 = Upsample(2048, 1024)
        self.upsample2 = Upsample(1024, 512)
        self.upsample3 = Upsample(512, 64)
        self.upsample4 = Upsample(64, 64)
        self.upsample5 = Upsample(64, 32)

        self.fs1 = Fusion(1024)
        self.fs2 = Fusion(512)
        self.fs3 = Fusion(256)
        self.fs4 = Fusion(64)
        self.fs5 = Fusion(64)

        self.out0 = self._classifier(2048)
        self.out1 = self._classifier(1024)
        self.out2 = self._classifier(512)
        self.out_e = self._classifier(256)
        self.out3 = self._classifier(64)
        self.out4 = self._classifier(64)
        self.out5 = self._classifier(32)

        self.transformer = nn.Conv2d(256, 64, kernel_size=1)

    def _classifier(self, inplanes):
        if inplanes == 32:
            return nn.Sequential(
                nn.Conv2d(inplanes, self.n_classes, 1),
                nn.Conv2d(self.n_classes, self.n_classes,
                          kernel_size=3, padding=1)
            )
        return nn.Sequential(
            nn.Conv2d(inplanes, inplanes//2, 3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes//2),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(inplanes//2, self.n_classes, 1),
        )

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        out32 = self.out0(fm4)

        fsfm1 = self.fs1(fm3, self.upsample1(fm4, fm3.size()[2:]))
        out16 = self.out1(fsfm1)

        fsfm2 = self.fs2(fm2, self.upsample2(fsfm1, fm2.size()[2:]))
        out8 = self.out2(fsfm2)

        fsfm3 = self.fs4(pool_x, self.upsample3(fsfm2, pool_x.size()[2:]))
        # print(fsfm3.size())
        out4 = self.out3(fsfm3)

        fsfm4 = self.fs5(conv_x, self.upsample4(fsfm3, conv_x.size()[2:]))
        out2 = self.out4(fsfm4)

        fsfm5 = self.upsample5(fsfm4, input.size()[2:])
        out = self.out5(fsfm5)

        return out 

class SegNet(nn.Module):
    def __init__(self, n_channels = 3, n_classes= 19):
        super(SegNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        

        batchNorm_momentum = 0.1

        self.conv11 = nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv31d = nn.Conv2d(256,  128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv11d = nn.Conv2d(64, n_classes, kernel_size=3, padding=1)


    def forward(self, x):

        # Stage 1
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x1_size = x12.size()
        x1p, id1 = F.max_pool2d(x12,kernel_size=2, stride=2,return_indices=True)

        # Stage 2
        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        x2_size = x22.size()
        x2p, id2 = F.max_pool2d(x22,kernel_size=2, stride=2,return_indices=True)

        # Stage 3
        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        x33 = F.relu(self.bn33(self.conv33(x32)))
        x3_size = x33.size()
        x3p, id3 = F.max_pool2d(x33,kernel_size=2, stride=2,return_indices=True)

        # Stage 4
        x41 = F.relu(self.bn41(self.conv41(x3p)))
        x42 = F.relu(self.bn42(self.conv42(x41)))
        x43 = F.relu(self.bn43(self.conv43(x42)))
        x4_size = x43.size()
        x4p, id4 = F.max_pool2d(x43,kernel_size=2, stride=2,return_indices=True)

        # Stage 5
        x51 = F.relu(self.bn51(self.conv51(x4p)))
        x52 = F.relu(self.bn52(self.conv52(x51)))
        x53 = F.relu(self.bn53(self.conv53(x52)))
        x5_size = x53.size()
        x5p, id5 = F.max_pool2d(x53,kernel_size=2, stride=2,return_indices=True)


        # Stage 5d
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2, output_size=x5_size)
        x53d = F.relu(self.bn53d(self.conv53d(x5d)))
        x52d = F.relu(self.bn52d(self.conv52d(x53d)))
        x51d = F.relu(self.bn51d(self.conv51d(x52d)))

        # Stage 4d
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2, output_size=x4_size)
        x43d = F.relu(self.bn43d(self.conv43d(x4d)))
        x42d = F.relu(self.bn42d(self.conv42d(x43d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))

        # Stage 3d
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2, output_size=x3_size)
        x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x32d = F.relu(self.bn32d(self.conv32d(x33d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))

        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2, output_size=x2_size)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))

        # Stage 1d
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2, output_size=x1_size)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)

        return x11d

    def load_from_segnet(self, model_path):
        s_dict = self.state_dict()# create a copy of the state dict
        th = torch.load(model_path).state_dict() # load the weigths
        # for name in th:
            # s_dict[corresp_name[name]] = th[name]
        self.load_state_dict(th)

class UNet2(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet2, self).__init__()
        self.n_classes = out_channels

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        def upsample(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
                nn.ReLU(inplace=True)
            )
        
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = conv_block(512, 1024)
        
        self.up4 = upsample(1024, 512)
        self.dec4 = conv_block(1024, 512)
        self.up3 = upsample(512, 256)
        self.dec3 = conv_block(512, 256)
        self.up2 = upsample(256, 128)
        self.dec2 = conv_block(256, 128)
        self.up1 = upsample(128, 64)
        self.dec1 = conv_block(128, 64)
        
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

        self.apply(self._initialize_weights)
        print("Initialized weights with Kaiming Normal")

        
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        b = self.bottleneck(self.pool(e4))
        
        d4 = self.dec4(torch.cat((self.up4(b), e4), dim=1))
        d3 = self.dec3(torch.cat((self.up3(d4), e3), dim=1))
        d2 = self.dec2(torch.cat((self.up2(d3), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
        
        return self.final(d1)
    
    def _initialize_weights(self, m): 
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            # print(f"Initializing {m}")
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            # print(f"Initializing BatchNorm: {m}")
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ReLU) or isinstance(m, nn.Sequential) or isinstance(m, UNet2) or isinstance(m, nn.MaxPool2d):
            pass
        else:
            print(f"Warning: {m} not initialized")


"""
the 3D Unet takes a 3D volume B x C x D x H x W and outputs a segmentation map of B x n_classes x D x H x W
"""
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet3D, self).__init__()
        self.n_classes = out_channels

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        # Upsampling with optional output_padding
        def upsample(in_c, out_c, output_padding=(0,0,0)):
            return nn.ConvTranspose3d(
                in_c, out_c, kernel_size=2, stride=2, output_padding=output_padding
            )

        # Encoder
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = conv_block(512, 1024)

        # Decoder
        # output_padding = (D_pad, H_pad, W_pad)
        self.up4 = upsample(1024, 512)  # fixes depth 3→6
        self.dec4 = conv_block(1024, 512)
        self.up3 = upsample(512, 256)  # no padding needed
        self.dec3 = conv_block(512, 256)
        self.up2 = upsample(256, 128)  # no padding needed
        self.dec2 = conv_block(256, 128)
        self.up1 = upsample(128, 64, output_padding=(1,0,0))  # fixes final depth 48 -> 49
        self.dec1 = conv_block(128, 64)

        # Final output layer
        self.final = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x) # B x 64 x 49 x 256 x 256
        e2 = self.enc2(self.pool(e1)) # B x 128 x 24 x 128 x 128
        e3 = self.enc3(self.pool(e2)) # B x 256 x 12 x 64 x 64
        e4 = self.enc4(self.pool(e3)) # B x 512 x 6 x 32 x 32
        b = self.bottleneck(self.pool(e4)) # B x 1024 x 3 x 16 x 16

        # Decoder
        d4 = self.dec4(torch.cat((self.up4(b), e4), dim=1))
        d3 = self.dec3(torch.cat((self.up3(d4), e3), dim=1))
        d2 = self.dec2(torch.cat((self.up2(d3), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))

        return self.final(d1)
    

class UNet3D_Aniso(nn.Module):
    """
    UNet3D with anisotropic pooling to preserve more depth resolution.
    Pooling strides: (2,2,2), (2,2,2), (1,2,2), (1,2,2)
    Corresponding upsampling uses the same strides reversed.
    """
    def __init__(self, in_channels=1, out_channels=1, base_filters=64):
        super().__init__()
        self.n_classes = out_channels

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True)
            )

        # anisotropic pooling kernels/strides
        self.pool1 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        self.pool2 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        self.pool3 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.pool4 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))

        # Encoder
        f = base_filters
        self.enc1 = conv_block(in_channels, f)
        self.enc2 = conv_block(f, f*2)
        self.enc3 = conv_block(f*2, f*4)
        self.enc4 = conv_block(f*4, f*8)

        # Bottleneck
        self.bottleneck = conv_block(f*8, f*16)

        # Decoder - use ConvTranspose3d with matching strides
        self.up4 = nn.ConvTranspose3d(f*16, f*8, kernel_size=(1,2,2), stride=(1,2,2))
        self.dec4 = conv_block(f*16, f*8)

        self.up3 = nn.ConvTranspose3d(f*8, f*4, kernel_size=(1,2,2), stride=(1,2,2))
        self.dec3 = conv_block(f*8, f*4)

        self.up2 = nn.ConvTranspose3d(f*4, f*2, kernel_size=(2,2,2), stride=(2,2,2))
        self.dec2 = conv_block(f*4, f*2)

        self.up1 = nn.ConvTranspose3d(f*2, f, kernel_size=(2,2,2), stride=(2,2,2), output_padding=(1,0,0))
        self.dec1 = conv_block(f*2, f)

        self.final = nn.Conv3d(f, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)                # B x f x 49 x 256 x 256
        p1 = self.pool1(e1)             # 24 x 128 x 128
        e2 = self.enc2(p1)              # B x 2f x 24 x 128 x 128
        p2 = self.pool2(e2)             # 12 x 64 x 64
        e3 = self.enc3(p2)              # B x 4f x 12 x 64 x 64
        p3 = self.pool3(e3)             # 12 x 32 x 32 (depth preserved)
        e4 = self.enc4(p3)              # B x 8f x 12 x 32 x 32
        p4 = self.pool4(e4)             # 12 x 16 x 16 (depth preserved)
        b = self.bottleneck(p4)         # B x 16f x 12 x 16 x 16

        # Decoder (reverse order of pooling)
        u4 = self.up4(b)                # B x 8f x 12 x 32 x 32
        d4 = self.dec4(torch.cat((u4, e4), dim=1))

        u3 = self.up3(d4)               # B x 4f x 12 x 64 x 64
        d3 = self.dec3(torch.cat((u3, e3), dim=1))

        u2 = self.up2(d3)               # B x 2f x 24 x 128 x 128
        d2 = self.dec2(torch.cat((u2, e2), dim=1))

        u1 = self.up1(d2)               # B x f x 49 x 256 x 256 (may need output_padding)
        # if up1 produces 48 in depth, you may need output_padding=(1,0,0) depending on input dims
        # concatenate with e1
        d1 = self.dec1(torch.cat((u1, e1), dim=1))

        return self.final(d1)

class UNet3D_Aniso2(nn.Module):
    """
    UNet3D with anisotropic pooling to preserve more depth resolution.
    Pooling strides: (2,2,2), (2,2,2), (1,2,2), (1,2,2)
    Corresponding upsampling uses the same strides reversed.
    """
    def __init__(self, in_channels=1, out_channels=1, base_filters=48):
        super().__init__()
        self.n_classes = out_channels

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True)
            )

        # anisotropic pooling kernels/strides
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.pool2 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.pool3 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        self.pool4 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))

        # Encoder
        f = base_filters
        self.enc1 = conv_block(in_channels, f)
        self.enc2 = conv_block(f, f*2)
        self.enc3 = conv_block(f*2, f*4)
        self.enc4 = conv_block(f*4, f*8)

        # Bottleneck
        self.bottleneck = conv_block(f*8, f*16)

        # Decoder - use ConvTranspose3d with matching strides
        self.up4 = nn.ConvTranspose3d(f*16, f*8, kernel_size=(1,2,2), stride=(1,2,2))
        self.dec4 = conv_block(f*16, f*8)

        self.up3 = nn.ConvTranspose3d(f*8, f*4, kernel_size=(2,2,2), stride=(2,2,2), output_padding=(1,0,0))
        self.dec3 = conv_block(f*8, f*4)

        self.up2 = nn.ConvTranspose3d(f*4, f*2, kernel_size=(1,2,2), stride=(1,2,2))
        self.dec2 = conv_block(f*4, f*2)

        self.up1 = nn.ConvTranspose3d(f*2, f, kernel_size=(1,2,2), stride=(1,2,2), output_padding=(0,0,0))
        self.dec1 = conv_block(f*2, f)

        self.final = nn.Conv3d(f, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)         # 
        p1 = self.pool1(e1)           
        e2 = self.enc2(p1)       
        p2 = self.pool2(e2)        
        e3 = self.enc3(p2)      
        p3 = self.pool3(e3)         
        e4 = self.enc4(p3)           
        p4 = self.pool4(e4)             
        b = self.bottleneck(p4)         

        # Decoder (reverse order of pooling)
        u4 = self.up4(b)                # B x 8f x 12 x 32 x 32
        d4 = self.dec4(torch.cat((u4, e4), dim=1))

        u3 = self.up3(d4)               # B x 4f x 12 x 64 x 64
        d3 = self.dec3(torch.cat((u3, e3), dim=1))

        u2 = self.up2(d3)               # B x 2f x 24 x 128 x 128
        d2 = self.dec2(torch.cat((u2, e2), dim=1))

        u1 = self.up1(d2)               # B x f x 49 x 256 x 256 (may need output_padding)
        # if up1 produces 48 in depth, you may need output_padding=(1,0,0) depending on input dims
        # concatenate with e1
        d1 = self.dec1(torch.cat((u1, e1), dim=1))

        return self.final(d1)
    
class SliceAttention(nn.Module):
    def __init__(self, dim, num_heads=4, max_slices=128):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

        # ----- Learned absolute positional embeddings -----
        self.pos_embed = nn.Parameter(torch.randn(1, max_slices, dim))

    def forward(self, x):
        """
        x: B x C x Z x H x W
        """
        B, C, Z, H, W = x.shape

        # 1) flatten spatial dims
        tokens = x.mean(dim=(3,4))      # B x C x Z
        tokens = tokens.permute(0, 2, 1)  # B x Z x C

        # 2) add positional embeddings (crop to Z)
        tokens = tokens + self.pos_embed[:, :Z, :]

        # 3) attention
        attn_out, _ = self.attn(tokens, tokens, tokens)
        out = self.norm(attn_out + tokens)

        # 4) broadcast back to 3D volume
        out = out.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        out = out.expand(B, C, Z, H, W)
        return out
    
class ZAxialAttention(nn.Module):
    """
    Efficient true Z-mixing attention.
    Performs self-attention along Z independently at each downsampled spatial location.

    Steps:
      1. Spatial downsample (adaptive pooling): (H,W) -> (ds,ds)
      2. For each (x,y), do attention along Z: sequence length = Z
      3. Upsample the attended volume back to original spatial resolution
    """

    def __init__(self, dim, num_heads=4, ds=16):
        """
        dim        — feature channels (e.g., 1024 in your bottleneck)
        num_heads  — attention heads
        ds         — spatial downsample Resolution (ds x ds grid)
        """
        super().__init__()
        self.ds = ds
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: (B, C, Z, H, W)
        Output: same shape, but with Z attentively mixed.
        """
        B, C, Z, H, W = x.shape

        # ----------------------------
        # 1. Spatial Downsample
        # ----------------------------
        # pool over spatial dims only → keeps Z unchanged
        # result: B, C, Z, ds, ds
        x_ds = F.adaptive_avg_pool3d(x, (Z, self.ds, self.ds))

        # rearrange for axial attention along Z
        # create (ds*ds) sequences, each of length Z
        x_ds = x_ds.permute(0, 3, 4, 2, 1)   # B, ds, ds, Z, C
        seqs = x_ds.reshape(B * self.ds * self.ds, Z, C)  # (B*ds*ds), Z, C

        # ----------------------------
        # 2. Apply Attention Along Z
        # ----------------------------
        attn_out, _ = self.attn(seqs, seqs, seqs)
        attn_out = self.norm(attn_out + seqs)

        # reshape back to grid
        out = attn_out.reshape(B, self.ds, self.ds, Z, C)
        out = out.permute(0, 4, 3, 1, 2)  # B, C, Z, ds, ds

        # ----------------------------
        # 3. Upsample Back to (H,W)
        # ----------------------------
        out = F.interpolate(
            out, size=(Z, H, W),
            mode="trilinear",
            align_corners=False
        )

        return out

class UNet2D_attention(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_heads=4):
        super().__init__()
        self.n_classes = out_channels

        # ---------- 2D building blocks ----------
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        def upsample(in_c, out_c):
            return nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)

        # ---------- Encoder ----------
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = conv_block(512, 1024)

        # ---------- Slice Attention at bottleneck ----------
        self.slice_attn = ZAxialAttention(dim=1024, num_heads=num_heads)

        # ---------- Decoder ----------
        self.up4 = upsample(1024, 512)
        self.dec4 = conv_block(1024, 512)

        self.up3 = upsample(512, 256)
        self.dec3 = conv_block(512, 256)

        self.up2 = upsample(256, 128)
        self.dec2 = conv_block(256, 128)

        self.up1 = upsample(128, 64)
        self.dec1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, out_channels, 1)

    def _process_slices_2d(self, x, module, pool=None):
        """
        Applies a 2D module slice-wise.
        x: B x C x Z x H x W
        Returns: B x C' x Z x H' x W'
        """
        B, C, Z, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)                # B x Z x C x H x W
        x = x.reshape(B * Z, C, H, W)               # (B*Z) x C x H x W

        x = module(x)                               # run 2D block

        if pool is not None:
            x = pool(x)
        C_out, H_out, W_out = x.shape[1], x.shape[2], x.shape[3]
        x = x.reshape(B, Z, C_out, H_out, W_out)
        x = x.permute(0, 2, 1, 3, 4)                # B x C_out x Z x H_out x W_out
        return x
    
    def _upsample_and_concat(self, x, skip, up_module):
        B, C, Z, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)                # B x Z x C x H x W
        x = x.reshape(B * Z, C, H, W)               # (B*Z) x C x

        x = up_module(x)                             # upsample
        _, C_up, H_up, W_up = x.shape

        x = x.reshape(B, Z, C_up, H_up, W_up)
        x = x.permute(0, 2, 1, 3, 4)                # B x C_up x Z x H_up x W_up
        x = torch.cat([x, skip], dim=1)             # concat with skip connection
        return x

    def forward(self, x):
        """
        x shape: B x C x Z x H x W   (e.g. B x 1 x 49 x 256 x 256)
        """

        # ---------- Encoder ----------
        e1 = self._process_slices_2d(x, self.enc1)
        e2 = self._process_slices_2d(e1, self.enc2, pool=self.pool)
        e3 = self._process_slices_2d(e2, self.enc3, pool=self.pool)
        e4 = self._process_slices_2d(e3, self.enc4, pool=self.pool)

        b = self._process_slices_2d(e4, self.bottleneck, pool=self.pool)
        # ---------- Z-Axis Attention ----------
        b = self.slice_attn(b)

        # ---------- Decoder ----------
        d4 = self._upsample_and_concat(b, e4, self.up4)
        d4 = self._process_slices_2d(d4, self.dec4)

        d3 = self._upsample_and_concat(d4, e3, self.up3)
        d3 = self._process_slices_2d(d3, self.dec3)

        d2 = self._upsample_and_concat(d3, e2, self.up2)
        d2 = self._process_slices_2d(d2, self.dec2)

        d1 = self._upsample_and_concat(d2, e1, self.up1)
        d1 = self._process_slices_2d(d1, self.dec1)

        # ---------- Final per-slice segmentation ----------
        out = self._process_slices_2d(d1, self.final)  # B x out_channels x Z x H x W

        return out

class UNet3DFrawley_backup(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet3DFrawley, self).__init__()
        k = 32
        self.n_classes = out_channels
        self.conv_down_00 = torch.nn.Conv3d(in_channels, k, 3, padding=1)
        self.bn_down_00 = torch.nn.BatchNorm3d(k)
        self.relu_down_00 = torch.nn.ReLU()
        self.conv_down_01 = torch.nn.Conv3d(k, k, 3, padding=1)
        self.bn_down_01 = torch.nn.BatchNorm3d(k)
        self.relu_down_01 = torch.nn.ReLU()
        self.conv_down_02 = torch.nn.Conv3d(k, 2*k, 3, padding=1)
        self.bn_down_02 = torch.nn.BatchNorm3d(2*k)
        self.relu_down_02 = torch.nn.ReLU()
        self.conv_down_03 = torch.nn.Conv3d(2*k, 2*k, 3, padding=1)
        self.bn_down_03 = torch.nn.BatchNorm3d(2*k)
        self.relu_down_03 = torch.nn.ReLU()
        self.max_pool_down_00 = torch.nn.MaxPool3d(2) 

        self.conv_down_10 = torch.nn.Conv3d(2*k, 4*k, 3, padding=1)
        self.bn_down_10 = torch.nn.BatchNorm3d(4*k)
        self.relu_down_10 = torch.nn.ReLU()
        self.conv_down_11 = torch.nn.Conv3d(4*k, 4*k, 3, padding=1)
        self.bn_down_11 = torch.nn.BatchNorm3d(4*k)
        self.relu_down_11 = torch.nn.ReLU()
        self.max_pool_down_10 = torch.nn.MaxPool3d(2) # 12 x 64 x 64

        self.conv_bottom_20 = torch.nn.Conv3d(4*k, 8*k, 3, padding=1)
        self.bn_bottom_20 = torch.nn.BatchNorm3d(8*k)
        self.relu_bottom_20 = torch.nn.ReLU()
        self.conv_bottom_21 = torch.nn.Conv3d(8*k, 8*k, 3, padding=1)
        self.bn_bottom_21 = torch.nn.BatchNorm3d(8*k)
        self.relu_bottom_21 = torch.nn.ReLU()
        self.conv_bottom_22 = torch.nn.Conv3d(8*k, 4*k, 1) 

        self.conv_up_10 = torch.nn.Conv3d(8*k, 4*k, 3, padding=1)
        self.bn_up_10 = torch.nn.BatchNorm3d(4*k)
        self.relu_up_10 = torch.nn.ReLU()
        self.conv_up_11 = torch.nn.Conv3d(4*k, 4*k, 3, padding=1)
        self.bn_up_11 = torch.nn.BatchNorm3d(4*k)
        self.relu_up_11 = torch.nn.ReLU()
        self.conv_up_12 = torch.nn.Conv3d(4*k, 2*k, 1)

        self.conv_up_00 = torch.nn.Conv3d(4*k, 2*k, 3, padding=1)
        self.bn_up_00 = torch.nn.BatchNorm3d(2*k)
        self.relu_up_00 = torch.nn.ReLU()
        self.conv_up_01 = torch.nn.Conv3d(2*k, 2*k, 3, padding=1)
        self.bn_up_01 = torch.nn.BatchNorm3d(2*k)
        self.relu_up_01 = torch.nn.ReLU()
        self.conv_up_02 = torch.nn.ConvTranspose3d(2*k, out_channels, kernel_size=[2, 1, 1], stride=1, padding=0) 

    def forward(self, x):
        # DOWN CONV
        x = self.relu_down_00(self.bn_down_00(self.conv_down_00(x)))
        x = self.relu_down_01(self.bn_down_01(self.conv_down_01(x)))
        x = self.relu_down_02(self.bn_down_02(self.conv_down_02(x)))
        x = self.relu_down_03(self.bn_down_03(self.conv_down_03(x)))
        first_layer_output = x
        x = self.max_pool_down_00(x)

        x = self.relu_down_10(self.bn_down_10(self.conv_down_10(x)))
        x = self.relu_down_11(self.bn_down_11(self.conv_down_11(x)))
        second_layer_output = x
        x = self.max_pool_down_10(x)

        # BOTTOM
        x = self.relu_bottom_20(self.bn_bottom_20(self.conv_bottom_20(x)))
        x = self.relu_bottom_21(self.bn_bottom_21(self.conv_bottom_21(x)))
        x = self.conv_bottom_22(torch.nn.functional.interpolate(x, mode='trilinear', scale_factor=2, align_corners=False))

        # UP CONV
        dx = x.size(-1) - second_layer_output.size(-1)
        dy = x.size(-2) - second_layer_output.size(-2)
        dz = x.size(-3) - second_layer_output.size(-3)
        second_layer_output = torch.nn.functional.pad(second_layer_output, (dx//2, (dx+1)//2, dy//2, (dy+1)//2, dz//2, (dz+1)//2))
        x = torch.cat((x, second_layer_output), dim=1)
        x = self.relu_up_10(self.bn_up_10(self.conv_up_10(x)))
        x = self.relu_up_11(self.bn_up_11(self.conv_up_11(x)))
        x = self.conv_up_12(torch.nn.functional.interpolate(x, mode='trilinear', scale_factor=2, align_corners=False))

        dx = x.size(-1) - first_layer_output.size(-1)
        dy = x.size(-2) - first_layer_output.size(-2)
        dz = x.size(-3) - first_layer_output.size(-3)
        first_layer_output = torch.nn.functional.pad(first_layer_output, (dx//2, (dx+1)//2, dy//2, (dy+1)//2, dz//2, (dz+1)//2))
        x = torch.cat((x, first_layer_output), dim=1)
        x = self.relu_up_00(self.bn_up_00(self.conv_up_00(x)))
        x = self.relu_up_01(self.bn_up_01(self.conv_up_01(x)))
        x = self.conv_up_02(x)

        return x
    
# Frawley's 3D Unet proposal for macular hole segmentation
# https://github.com/gliff-ai/robust-3d-unet-macular-holes/blob/main/src/models/unet_3d_proposal.py
class UNet3DFrawley(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet3DFrawley, self).__init__()
        k = 32
        self.n_classes = out_channels
        self.conv_down_00 = torch.nn.Conv3d(in_channels, k, 3, padding=1)
        self.bn_down_00 = torch.nn.BatchNorm3d(k)
        self.relu_down_00 = torch.nn.ReLU()
        self.conv_down_01 = torch.nn.Conv3d(k, k, 3, padding=1)
        self.bn_down_01 = torch.nn.BatchNorm3d(k)
        self.relu_down_01 = torch.nn.ReLU()
        self.conv_down_02 = torch.nn.Conv3d(k, 2*k, 3, padding=1)
        self.bn_down_02 = torch.nn.BatchNorm3d(2*k)
        self.relu_down_02 = torch.nn.ReLU()
        self.conv_down_03 = torch.nn.Conv3d(2*k, 2*k, 3, padding=1)
        self.bn_down_03 = torch.nn.BatchNorm3d(2*k)
        self.relu_down_03 = torch.nn.ReLU()
        self.max_pool_down_00 = torch.nn.MaxPool3d(2) 

        self.conv_down_10 = torch.nn.Conv3d(2*k, 4*k, 3, padding=1)
        self.bn_down_10 = torch.nn.BatchNorm3d(4*k)
        self.relu_down_10 = torch.nn.ReLU()
        self.conv_down_11 = torch.nn.Conv3d(4*k, 4*k, 3, padding=1)
        self.bn_down_11 = torch.nn.BatchNorm3d(4*k)
        self.relu_down_11 = torch.nn.ReLU()
        self.max_pool_down_10 = torch.nn.MaxPool3d(2) # 12 x 64 x 64

        self.conv_bottom_20 = torch.nn.Conv3d(4*k, 8*k, 3, padding=1)
        self.bn_bottom_20 = torch.nn.BatchNorm3d(8*k)
        self.relu_bottom_20 = torch.nn.ReLU()
        self.conv_bottom_21 = torch.nn.Conv3d(8*k, 8*k, 3, padding=1)
        self.bn_bottom_21 = torch.nn.BatchNorm3d(8*k)
        self.relu_bottom_21 = torch.nn.ReLU()
        self.conv_bottom_22 = torch.nn.Conv3d(8*k, 4*k, 1) 

        self.conv_up_10 = torch.nn.Conv3d(8*k, 4*k, 3, padding=1)
        self.bn_up_10 = torch.nn.BatchNorm3d(4*k)
        self.relu_up_10 = torch.nn.ReLU()
        self.conv_up_11 = torch.nn.Conv3d(4*k, 4*k, 3, padding=1)
        self.bn_up_11 = torch.nn.BatchNorm3d(4*k)
        self.relu_up_11 = torch.nn.ReLU()
        self.conv_up_12 = torch.nn.Conv3d(4*k, 2*k, 1)

        self.conv_up_00 = torch.nn.Conv3d(4*k, 2*k, 3, padding=1)
        self.bn_up_00 = torch.nn.BatchNorm3d(2*k)
        self.relu_up_00 = torch.nn.ReLU()
        self.conv_up_01 = torch.nn.Conv3d(2*k, 2*k, 3, padding=1)
        self.bn_up_01 = torch.nn.BatchNorm3d(2*k)
        self.relu_up_01 = torch.nn.ReLU()
        self.conv_up_02 = torch.nn.ConvTranspose3d(2*k, out_channels, kernel_size=[2, 1, 1], stride=1, padding=0) 

    def forward(self, x):
        # DOWN CONV
        x = self.relu_down_00(self.bn_down_00(self.conv_down_00(x)))
        x = self.relu_down_01(self.bn_down_01(self.conv_down_01(x)))
        x = self.relu_down_02(self.bn_down_02(self.conv_down_02(x)))
        x = self.relu_down_03(self.bn_down_03(self.conv_down_03(x)))
        first_layer_output = x
        x = self.max_pool_down_00(x)

        x = self.relu_down_10(self.bn_down_10(self.conv_down_10(x)))
        x = self.relu_down_11(self.bn_down_11(self.conv_down_11(x)))
        second_layer_output = x
        x = self.max_pool_down_10(x)

        # BOTTOM
        x = self.relu_bottom_20(self.bn_bottom_20(self.conv_bottom_20(x)))
        x = self.relu_bottom_21(self.bn_bottom_21(self.conv_bottom_21(x)))
        x = self.conv_bottom_22(torch.nn.functional.interpolate(x, mode='trilinear', scale_factor=2, align_corners=False))

        # UP CONV
        dx = x.size(-1) - second_layer_output.size(-1)
        dy = x.size(-2) - second_layer_output.size(-2)
        dz = x.size(-3) - second_layer_output.size(-3)
        second_layer_output = torch.nn.functional.pad(second_layer_output, (dx//2, (dx+1)//2, dy//2, (dy+1)//2, dz//2, (dz+1)//2))
        x = torch.cat((x, second_layer_output), dim=1)
        x = self.relu_up_10(self.bn_up_10(self.conv_up_10(x)))
        x = self.relu_up_11(self.bn_up_11(self.conv_up_11(x)))
        x = self.conv_up_12(torch.nn.functional.interpolate(x, mode='trilinear', scale_factor=2, align_corners=False))

        dx = x.size(-1) - first_layer_output.size(-1)
        dy = x.size(-2) - first_layer_output.size(-2)
        dz = x.size(-3) - first_layer_output.size(-3)
        first_layer_output = torch.nn.functional.pad(first_layer_output, (dx//2, (dx+1)//2, dy//2, (dy+1)//2, dz//2, (dz+1)//2))
        x = torch.cat((x, first_layer_output), dim=1)
        x = self.relu_up_00(self.bn_up_00(self.conv_up_00(x)))
        x = self.relu_up_01(self.bn_up_01(self.conv_up_01(x)))
        x = self.conv_up_02(x)

        return x




# Multi-Scale Guided Unet with MGR module. From https://github.com/Jiaxuan-Li/MGU-Net
class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=True)
        self.relu = nn.LeakyReLU(0.2,inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        h = self.relu(h)
        h = self.conv2(h)
        return h

class Basconv(nn.Sequential):
    def __init__(self, in_channels, out_channels, is_batchnorm = False, kernel_size = 3, stride = 1, padding=1):
        super(Basconv, self).__init__()
        if is_batchnorm:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))
        else:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),nn.ReLU(inplace=True))

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')
    
    def forward(self, inputs):
        x = inputs
        x = self.conv(x)
        return x

class GloRe_Unit(nn.Module):

    def __init__(self, num_in, num_mid, stride=(1,1), kernel=1):
        super(GloRe_Unit, self).__init__()

        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)
        kernel_size = (kernel, kernel)
        padding = (1, 1) if kernel == 3 else (0, 0)
        # reduce dimension
        self.conv_state = Basconv(num_in, self.num_s, is_batchnorm = True, kernel_size=kernel_size, padding=padding)  
        # generate projection and inverse projection functions
        self.conv_proj = Basconv(num_in, self.num_n, is_batchnorm = True,kernel_size=kernel_size, padding=padding)   
        self.conv_reproj = Basconv(num_in, self.num_n, is_batchnorm = True,kernel_size=kernel_size, padding=padding)  
        # reasoning by graph convolution
        self.gcn1 = GCN(num_state=self.num_s, num_node=self.num_n)   
        self.gcn2 = GCN(num_state=self.num_s, num_node=self.num_n)  
        # fusion
        self.fc_2 = nn.Conv2d(self.num_s, num_in, kernel_size=kernel_size, padding=padding, stride=(1,1), 
                              groups=1, bias=False)
        self.blocker = nn.BatchNorm2d(num_in) 

    def forward(self, x):
        batch_size = x.size(0)
        # generate projection and inverse projection matrices
        x_state_reshaped = self.conv_state(x).view(batch_size, self.num_s, -1) 
        x_proj_reshaped = self.conv_proj(x).view(batch_size, self.num_n, -1)
        x_rproj_reshaped = self.conv_reproj(x).view(batch_size, self.num_n, -1)
        # project to node space
        x_n_state1 = torch.bmm(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1)) 
        x_n_state2 = x_n_state1 * (1. / x_state_reshaped.size(2))
        # graph convolution
        x_n_rel1 = self.gcn1(x_n_state2)  
        x_n_rel2 = self.gcn2(x_n_rel1)
        # inverse project to original space
        x_state_reshaped = torch.bmm(x_n_rel2, x_rproj_reshaped)
        x_state = x_state_reshaped.view(batch_size, self.num_s, *x.size()[2:])
        # fusion
        out = x + self.blocker(self.fc_2(x_state))

        return out

class  MGR_Module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MGR_Module, self).__init__()

        self.conv0_1 = Basconv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou0 = nn.Sequential(OrderedDict([("GCN%02d" % i, GloRe_Unit(out_channels, out_channels, kernel=1)) for i in range(1)]))

        self.conv1_1 = Basconv(in_channels=in_channels,out_channels=out_channels, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.conv1_2 = Basconv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou1 = nn.Sequential(OrderedDict([("GCN%02d" % i,GloRe_Unit(out_channels, out_channels, kernel=1)) for i in range(1)]))

        self.conv2_1 = Basconv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.conv2_2 = Basconv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou2 = nn.Sequential(OrderedDict([("GCN%02d" % i,GloRe_Unit(out_channels, int(out_channels/2), kernel=1)) for i in range(1)]))

        self.conv3_1 = Basconv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.conv3_2 = Basconv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou3 = nn.Sequential(OrderedDict([("GCN%02d" % i,GloRe_Unit(out_channels, int(out_channels/2), kernel=1)) for i in range(1)]))
        
        self.f1 = Basconv(in_channels=4*out_channels, out_channels=in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)

        self.x0 = self.conv0_1(x)
        self.g0 = self.glou0(self.x0)

        self.x1 = self.conv1_2(self.pool1(self.conv1_1(x)))
        self.g1 = self.glou1(self.x1)
        self.layer1 = F.interpolate(self.g1, size=(h, w), mode='bilinear', align_corners=True)

        self.x2 = self.conv2_2(self.pool2(self.conv2_1(x)))
        self.g2 = self.glou2(self.x2)
        self.layer2 = F.interpolate(self.g2, size=(h, w), mode='bilinear', align_corners=True)

        self.x3 = self.conv3_2(self.pool3(self.conv3_1(x)))
        self.g3= self.glou3(self.x3)
        self.layer3 = F.interpolate(self.g3, size=(h, w), mode='bilinear', align_corners=True)

        out = torch.cat([self.g0, self.layer1, self.layer2, self.layer3], 1)

        return self.f1(out)

class UnetConv(nn.Module):
    def __init__(self, in_channels, out_channels, is_batchnorm, n=2, kernel_size = 3, stride=1, padding=1):
        super(UnetConv, self).__init__()
        self.n = n

        if is_batchnorm:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_channels = out_channels

        else:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_channels = out_channels

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n+1):
            conv = getattr(self, 'conv%d'%i)
            x = conv(x)
        return x

class UnetUp(nn.Module):
    def __init__(self,in_channels, out_channels, is_deconv, n_concat=2):
        super(UnetUp, self).__init__()
        self.conv = UnetConv(in_channels+(n_concat-2)* out_channels, out_channels, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0,*input):
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0,input[i]], 1)
        return self.conv(outputs0)
    

class MGUNet_2(nn.Module):
    def __init__(self, in_channels=1, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):  ##########
        super(MGUNet_2, self).__init__()
        self.n_classes = n_classes
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # encoder
        self.conv1 = UnetConv(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = UnetConv(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = UnetConv(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.mgb =  MGR_Module(filters[2], filters[3])

        self.center = UnetConv(filters[2], filters[3], self.is_batchnorm)

        # decoder
        self.up_concat3 = UnetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UnetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UnetUp(filters[1], filters[0], self.is_deconv)

        # final conv
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)  
        maxpool1 = self.maxpool1(conv1) 
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)  
        conv3 = self.conv3(maxpool2)  
        maxpool3 = self.maxpool3(conv3)  
        feat_sum = self.mgb(maxpool3) 
        center = self.center(feat_sum)  
        up3 = self.up_concat3(center, conv3) 
        up2 = self.up_concat2(up3, conv2) 
        up1 = self.up_concat1(up2, conv1)
        final_1 = self.final_1(up1)

        return final_1


# ---------------------------
# Loss-friendly building blocks
# ---------------------------

class ConvBlock2D(nn.Module):
    """(Conv2d -> GroupNorm -> ReLU) * 2"""
    def __init__(self, in_c: int, out_c: int, groups: int = 8):
        super().__init__()
        g = min(groups, out_c)  # safety for small channel counts
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class ConvBlock3D(nn.Module):
    """(Conv3d -> ReLU) * 2, default full 3x3x3 (mixes slices)."""
    def __init__(self, in_c: int, out_c: int, kz: int = 3, groups: int = 8):
        """
        kz:
          - 3 => kernel (3,3,3) full 3D mixing
          - 1 => kernel (1,3,3) no slice mixing
        """
        super().__init__()
        assert kz in (1, 3)
        k = (kz, 3, 3)
        p = (kz // 2, 1, 1)
        g = min(groups, out_c)

        self.net = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=k, padding=p, bias=False),
            nn.GroupNorm(g, out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=k, padding=p, bias=False),
            nn.GroupNorm(g, out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


def upsample_inplane(in_c: int, out_c: int) -> nn.Module:
    """Upsample H/W only, keep depth D unchanged."""
    return nn.ConvTranspose3d(in_c, out_c, kernel_size=(1, 2, 2), stride=(1, 2, 2), bias=False)


# ---------------------------
# Hybrid 2D-encoder / 3D-decoder U-Net
# ---------------------------

class UNet2DEnc3DDec(nn.Module):
    """
    2D encoder per slice + 3D decoder for volumetric coherence.

    Input : B x C x D x H x W
    Output: B x out_channels x D x H x W (logits)

    Design:
      - Encoder: 2D convs with 2D pooling (downsample H/W only) per slice
      - Decoder: 3D upsampling (H/W only) + 3D conv blocks
      - Slice mixing: controlled by kz in decoder blocks (default: mix at bottleneck, keep high-res aniso)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base: int = 64,
        # anisotropy-friendly defaults:
        # mix slices at coarse scales; avoid excessive mixing at full-res
        kz_bottleneck: int = 3,   # 3 => mix slices at bottleneck
        kz_dec4: int = 3,         # H/8
        kz_dec3: int = 3,         # H/4
        kz_dec2: int = 1,         # H/2
        kz_dec1: int = 1,         # H
    ):
        super().__init__()
        self.n_classes = out_channels

        # 2D encoder (shared across slices)
        self.enc1 = ConvBlock2D(in_channels, base)        # 64
        self.enc2 = ConvBlock2D(base, base * 2)           # 128
        self.enc3 = ConvBlock2D(base * 2, base * 4)       # 256
        self.enc4 = ConvBlock2D(base * 4, base * 8)       # 512
        self.pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck2d = ConvBlock2D(base * 8, base * 16)  # 1024

        # 3D bottleneck mixing (after stacking)
        self.bottleneck3d = ConvBlock3D(base * 16, base * 16, kz=kz_bottleneck)

        # 3D decoder (upsample in-plane only)
        self.up4 = upsample_inplane(base * 16, base * 8)
        self.dec4 = ConvBlock3D(base * 16, base * 8, kz=kz_dec4)

        self.up3 = upsample_inplane(base * 8, base * 4)
        self.dec3 = ConvBlock3D(base * 8, base * 4, kz=kz_dec3)

        self.up2 = upsample_inplane(base * 4, base * 2)
        self.dec2 = ConvBlock3D(base * 4, base * 2, kz=kz_dec2)

        self.up1 = upsample_inplane(base * 2, base)
        self.dec1 = ConvBlock3D(base * 2, base, kz=kz_dec1)

        # self.z_refine = nn.Sequential(
        #     nn.Conv3d(base, base, kernel_size=(3,1,1), padding=(1,0,0), bias=False),
        #     nn.GroupNorm(min(8, base), base),
        #     nn.ReLU(inplace=True),
        # )


        self.final = nn.Conv3d(base, out_channels, kernel_size=1)

    @staticmethod
    def _to_slice_batch(x: torch.Tensor) -> torch.Tensor:
        """
        B x C x D x H x W -> (B*D) x C x H x W
        """
        B, C, D, H, W = x.shape
        return x.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)

    @staticmethod
    def _to_volume(f: torch.Tensor, B: int, D: int) -> torch.Tensor:
        """
        (B*D) x C x H x W -> B x C x D x H x W
        """
        BD, C, H, W = f.shape
        assert BD == B * D, f"Expected BD={B*D}, got {BD}"
        return f.view(B, D, C, H, W).permute(0, 2, 1, 3, 4).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: B x C x D x H x W
        """
        B, C, D, H, W = x.shape

        # --- 2D encoder per slice ---
        xs = self._to_slice_batch(x)  # (B*D, C, H, W)

        e1s = self.enc1(xs)           # (B*D, base, H, W)
        p1 = self.pool2d(e1s)         # (B*D, base, H/2, W/2)

        e2s = self.enc2(p1)           # (B*D, 2*base, H/2, W/2)
        p2 = self.pool2d(e2s)         # (B*D, 2*base, H/4, W/4)

        e3s = self.enc3(p2)           # (B*D, 4*base, H/4, W/4)
        p3 = self.pool2d(e3s)         # (B*D, 4*base, H/8, W/8)

        e4s = self.enc4(p3)           # (B*D, 8*base, H/8, W/8)
        p4 = self.pool2d(e4s)         # (B*D, 8*base, H/16, W/16)

        bs = self.bottleneck2d(p4)    # (B*D, 16*base, H/16, W/16)

        # --- stack to 3D volumes ---
        e1 = self._to_volume(e1s, B, D)  # B x base x D x H x W
        e2 = self._to_volume(e2s, B, D)  # B x 2b  x D x H/2 x W/2
        e3 = self._to_volume(e3s, B, D)  # B x 4b  x D x H/4 x W/4
        e4 = self._to_volume(e4s, B, D)  # B x 8b  x D x H/8 x W/8
        b  = self._to_volume(bs,  B, D)  # B x 16b x D x H/16 x W/16

        # --- 3D bottleneck mixing ---
        b = self.bottleneck3d(b)

        # --- 3D decoder ---
        u4 = self.up4(b)                          # B x 8b x D x H/8  x W/8
        d4 = self.dec4(torch.cat([u4, e4], 1))     # B x 8b x D x H/8  x W/8

        u3 = self.up3(d4)                         # B x 4b x D x H/4  x W/4
        d3 = self.dec3(torch.cat([u3, e3], 1))     # B x 4b x D x H/4  x W/4

        u2 = self.up2(d3)                         # B x 2b x D x H/2  x W/2
        d2 = self.dec2(torch.cat([u2, e2], 1))     # B x 2b x D x H/2  x W/2

        u1 = self.up1(d2)                         # B x b x D x H    x W
        d1 = self.dec1(torch.cat([u1, e1], 1))     # B x b x D x H    x W

        # d1 = self.z_refine(d1)                    # B x b x D x H    x W
        return self.final(d1)                     # B x out_channels x D x H x W
    

# CSAM:
class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, max_pool, return_single=False):
        super().__init__()
        self.max_pool = max_pool
        self.return_single = return_single

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 1, 1),
            nn.InstanceNorm2d(output_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, 1, 1),
            nn.InstanceNorm2d(output_channels),
            nn.LeakyReLU(inplace=True),
        )
        if max_pool:
            self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        b = x
        if self.max_pool:
            x = self.pool(x)
        if self.return_single:
            return x
        return x, b


class DeconvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, intermediate_channels=-1):
        super().__init__()
        input_channels = int(input_channels)
        output_channels = int(output_channels)

        if intermediate_channels < 0:
            intermediate_channels = output_channels * 2
        else:
            intermediate_channels = input_channels

        self.upconv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(input_channels, intermediate_channels // 2, 3, 1, 1),
        )
        self.conv = ConvBlock(intermediate_channels, output_channels, max_pool=False)

    def forward(self, x, b):
        x = self.upconv(x)
        x = torch.cat((x, b), dim=1)
        x, _ = self.conv(x)
        return x


class UNetDecoder(nn.Module):
    """
    Fixed: has a forward() and uses a ModuleList.
    Expects skips in decoder order: high-res -> low-res.
    """
    def __init__(self, num_layers, base_num):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_layers - 1, 0, -1):
            self.blocks.append(
                DeconvBlock(base_num * (2 ** i), base_num * (2 ** (i - 1)))
            )

    def forward(self, x, skips):
        assert len(skips) == len(self.blocks), f"Expected {len(self.blocks)} skips, got {len(skips)}"
        for blk, skip in zip(self.blocks, skips):
            x = blk(x, skip)
        return x
    
# -------------------------
# Vectorized CSAM for volumes: semantic + positional + slice (per B sample)
# Works on x shaped [B, D, C, H, W]
# -------------------------
class SemanticAttention5D(nn.Module):
    """
    Operates on the chanenel dimension C. 
    Which feature channels are more important for this volume?
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )

    def forward(self, x):
        # x: [B,D,C,H,W]
        B, D, C, H, W = x.shape
        avg = x.mean(dim=(1, 3, 4))   # [B,C]
        mx  = x.amax(dim=(1, 3, 4))   # [B,C]
        gate = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(B, 1, C, 1, 1)
        return x * gate


class PositionalAttention5D(nn.Module):
    """
    Operates on the spatial dimensions H/W.
    Which spatial locations are more important for this volume?
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=pad)

    def forward(self, x):
        # x: [B,D,C,H,W]
        B, D, C, H, W = x.shape
        xd = x.reshape(B * D, C, H, W)              # [B*D,C,H,W]
        avg = xd.mean(dim=1, keepdim=True)          # [B*D,1,H,W]
        mx  = xd.amax(dim=1, keepdim=True)          # [B*D,1,H,W]
        att = torch.sigmoid(self.conv(torch.cat([mx, avg], dim=1)))  # [B*D,1,H,W]
        xd = xd * att
        return xd.view(B, D, C, H, W)


class SliceAttention5D(nn.Module):
    """
    Operates on the slice dimension D.
    Which slices are more important for this volume?
    """
    def __init__(self, channels: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),  # score per slice
        )

    def forward(self, x):
        # x: [B,D,C,H,W]
        B, D, C, H, W = x.shape
        desc = x.mean(dim=(3, 4))          # [B,D,C]
        scores = self.mlp(desc)            # [B,D,1]
        w = torch.softmax(scores, dim=1).view(B, D, 1, 1, 1)
        return x * w


class CSAM5D(nn.Module):
    def __init__(self, channels: int, semantic=True, positional=True, slice_att=True):
        super().__init__()
        self.semantic_on = semantic
        self.positional_on = positional
        self.slice_on = slice_att
        if semantic:
            self.semantic = SemanticAttention5D(channels)
        if positional:
            self.positional = PositionalAttention5D(kernel_size=7)
        if slice_att:
            self.slice = SliceAttention5D(channels)

    def forward(self, x):
        if self.semantic_on:
            x = self.semantic(x)
        if self.positional_on:
            x = self.positional(x)
        if self.slice_on:
            x = self.slice(x)
        return x


# -------------------------
# Encoder with CSAM applied at each level (vectorized per-batch)
# Input/Output interface is volume-first to match your training loop:
#   input:  [B, 1, D, H, W]
#   output: [B, 1, D, H, W] (logits)
# -------------------------
class EncoderCSAM5D(nn.Module):
    """
    Slice-wise 2D encoder with volume CSAM inserted on skip features and bottleneck.
    """
    def __init__(self, input_channels, num_layers, base_num,
                 semantic=True, positional=True, slice_att=True):
        super().__init__()
        self.num_layers = num_layers

        self.blocks = nn.ModuleList()
        self.attn = nn.ModuleList()

        for i in range(num_layers):
            in_ch = input_channels if i == 0 else base_num * (2 ** (i - 1))
            out_ch = base_num * (2 ** i)
            use_pool = (i != num_layers - 1)  # no pool at last (bottleneck)
            self.blocks.append(ConvBlock(in_ch, out_ch, max_pool=use_pool))
            self.attn.append(CSAM5D(out_ch, semantic=semantic, positional=positional, slice_att=slice_att))

    def forward(self, x):
        """
        x: [B,1,D,H,W]
        Returns:
          bottleneck_2d: [B*D, Cb, Hb, Wb]
          skips: list of skip tensors for decoder, each [B*D, Ci, Hi, Wi], high-res first
          meta: (B,D) for reshaping
        """
        if x.dim() != 5:
            raise ValueError(f"Expected input [B,1,D,H,W], got {x.shape}")

        B, C0, D, H, W = x.shape

        # Flatten volume into slice-batch: [B*D, C, H, W]
        xs = x.permute(0, 2, 1, 3, 4).reshape(B * D, C0, H, W)

        skips_2d = []
        cur = xs

        for i in range(self.num_layers):
            cur, skip = self.blocks[i](cur)  # skip: [B*D, Ci, Hi, Wi]; cur pooled if not last
            Ci, Hi, Wi = skip.shape[1], skip.shape[2], skip.shape[3]

            # Apply CSAM in 5D space: reshape skip -> [B,D,C,H,W]
            skip5d = skip.view(B, D, Ci, Hi, Wi)
            skip5d = self.attn[i](skip5d)
            skip = skip5d.view(B * D, Ci, Hi, Wi)

            # collect skips for decoder except bottleneck (last)
            if i != self.num_layers - 1:
                skips_2d.append(skip)
            else:
                cur = skip 

        # Decoder expects high-res first; we collected from shallow->deep. Reverse.
        skips_2d = skips_2d[::-1]
        return cur, skips_2d, (B, D)


class CSAM_UNet2p5D(nn.Module):
    """
    End-to-end segmentation model:
      input:  [B,1,D,H,W]
      output: [B,1,D,H,W] logits (for BCEWithLogitsLoss + Dice)
    """
    def __init__(self, in_channels=1, out_channels=1, num_layers=5, base_num=32,
                 semantic=True, positional=True, slice_att=True):
        super().__init__()
        self.n_classes = out_channels
        self.encoder = EncoderCSAM5D(in_channels, num_layers, base_num,
                                     semantic=semantic, positional=positional, slice_att=slice_att)
        self.decoder = UNetDecoder(num_layers, base_num)
        self.head = nn.Conv2d(base_num, out_channels, kernel_size=1)

    def forward(self, x):
        # x: [B,1,D,H,W]
        bottleneck, skips, meta = self.encoder(x)      # bottleneck: [B*D, Cb, hb, wb]
        y2d = self.decoder(bottleneck, skips)          # [B*D, base_num, H, W]
        y2d = self.head(y2d)                           # [B*D, out, H, W]

        B, D = meta
        H, W = y2d.shape[-2], y2d.shape[-1]
        y = y2d.view(B, D, self.n_classes, H, W).permute(0, 2, 1, 3, 4)  # [B,out,D,H,W]
        return y


# 2.5D stacked-slices-as-channels UNet:
class UNet2D(nn.Module):
    """
    Plain 2D U-Net that accepts [N, Cin, H, W] and outputs [N, Cout, H, W]
    Using your blocks.
    """
    def __init__(self, in_channels, out_channels=1, num_layers=5, base_num=32):
        super().__init__()
        self.n_classes = out_channels
        self.num_layers = num_layers

        self.enc = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else base_num * (2 ** (i - 1))
            out_ch = base_num * (2 ** i)
            use_pool = (i != num_layers - 1)
            self.enc.append(ConvBlock(in_ch, out_ch, max_pool=use_pool))

        self.dec = UNetDecoder(num_layers, base_num)
        self.head = nn.Conv2d(base_num, out_channels, kernel_size=1)

    def forward(self, x):
        # x: [N, Cin, H, W]
        skips = []
        cur = x
        for i in range(self.num_layers):
            cur, skip = self.enc[i](cur)
            if i != self.num_layers - 1:
                skips.append(skip)
        skips = skips[::-1]
        cur = self.dec(cur, skips)
        return self.head(cur)


class UNet2p5D_SlidingWindow(nn.Module):
    """
    2.5D stacked-slices-as-channels via sliding window:
      input : [B, 1, D, H, W]
      output: [B, 1, D, H, W]   (logits per slice)

    For each slice d, build k-channel input = neighboring slices around d,
    then run a 2D U-Net to predict the center slice mask.
    """
    def __init__(self, k=7, out_channels=1, num_layers=5, base_num=32, pad_mode="replicate"):
        super().__init__()
        assert k % 2 == 1, "k must be odd (e.g., 5,7,9)"
        self.k = k
        self.r = k // 2
        self.pad_mode = pad_mode
        self.n_classes = out_channels

        # 2D UNet that mixes slices as channels
        self.unet2d = UNet2D(in_channels=k, out_channels=out_channels, num_layers=num_layers, base_num=base_num)

    def forward(self, x):
        """
        x: [B, 1, D, H, W]
        return: [B, 1, D, H, W]
        """
        if x.dim() != 5:
            raise ValueError(f"Expected [B,1,D,H,W], got {x.shape}")
        B, C, D, H, W = x.shape
        if C != 1:
            raise ValueError("Expected single-channel grayscale input (C=1).")

        # Make depth-first: [B, D, H, W]
        v = x[:, 0]  # [B, D, H, W]

        # Pad along depth so every slice has a full k-window
        # pad format for F.pad on 4D is (..., Wpad, Hpad, Dpad) but for [B,D,H,W],
        # treat D as dim=1, so pad=(Wl,Wr,Hl,Hr,Dl,Dr)
        v_pad = F.pad(v, pad=(0, 0, 0, 0, self.r, self.r), mode=self.pad_mode)  # [B, D+2r, H, W]

        # Build sliding windows: for each d in [0..D-1], take v_pad[:, d:d+k]
        # Result: [B, D, k, H, W]
        windows = v_pad.unfold(dimension=1, size=self.k, step=1)  # [B, D, H, W, k] OR [B,D,k,H,W] depending on PyTorch
        # PyTorch unfold puts the new dimension at the end: [B, D, H, W, k]
        # We want [B, D, k, H, W]
        windows = windows.permute(0, 1, 4, 2, 3).contiguous()

        # Flatten slices into batch for 2D UNet: [B*D, k, H, W]
        inp2d = windows.view(B * D, self.k, H, W)

        # Predict center slice for each window: [B*D, 1, H, W]
        out2d = self.unet2d(inp2d)

        # Reshape back to volume: [B, D, 1, H, W] -> [B, 1, D, H, W]
        out = out2d.view(B, D, self.n_classes, H, W).permute(0, 2, 1, 3, 4).contiguous()
        return out



#############
# SWIN UNET
#############

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SwinEncoderUNet2D(nn.Module):
    """
    Pretrained Swin encoder + custom CNN decoder for binary segmentation.
    Input:  [B, C, H, W]
    Output: [B, 1, H, W] logits
    """
    def __init__(
        self,
        n_channels=1,
        n_classes=1,
        backbone="swin_tiny_patch4_window7_224",
        pretrained=True,
    ):
        super().__init__()

        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=n_channels,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size = 256
        )
        self.n_classes = n_classes
        self.n_channels = n_channels

        enc_channels = self.encoder.feature_info.channels()
        # For swin_tiny this is typically [96, 192, 384, 768]

        self.center = nn.Sequential(
            ConvBNReLU(enc_channels[-1], enc_channels[-1]),
            ConvBNReLU(enc_channels[-1], enc_channels[-1]),
        )

        self.dec4 = DecoderBlock(enc_channels[-1], enc_channels[-2], 384)
        self.dec3 = DecoderBlock(384, enc_channels[-3], 192)
        self.dec2 = DecoderBlock(192, enc_channels[-4], 96)
        self.dec1 = DecoderBlock(96, 0, 64)
        self.dec0 = DecoderBlock(64, 0, 32)

        self.seg_head = nn.Conv2d(32, n_classes, kernel_size=1)

    @staticmethod
    def _to_nchw(x):
        # timm Swin features are commonly NHWC
        if x.ndim == 4 and x.shape[1] != x.shape[-1]:
            # likely NHWC -> NCHW
            x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x):
        H, W = x.shape[-2:]

        feats = self.encoder(x)
        f1, f2, f3, f4 = [self._to_nchw(f) for f in feats]

        x = self.center(f4)
        x = self.dec4(x, f3)
        x = self.dec3(x, f2)
        x = self.dec2(x, f1)
        x = self.dec1(x, None)
        x = self.dec0(x, None)

        x = self.seg_head(x)

        if x.shape[-2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

        return x


##########################
# SwinUNETR 3D from MONAI:
##########################

class SwinUNETR3D(nn.Module):
    """
    3D Swin UNETR wrapper:
      input : [B, C, D, H, W]
      output: [B, out_channels, D, H, W]

    Expected OCT input:
      [B, 1, 49, 256, 256]

    Internally, depth is padded from 49 -> 64 so that SwinUNETR's hierarchical
    downsampling works cleanly. The prediction is cropped back to depth 49,
    so externally it behaves like the rest of the models.
    """
        
    def __init__(self, in_channels=1, n_classes=1, img_size=(64,256,256), feature_size=48,
                 use_checkpoint=False, pretrained_path=None, train_decoder_from_scratch=True,
                   freeze_encoder=False):
        super().__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.internal_img_size = img_size
        self.expected_input_size = (49,256,256)
        # https://monai.readthedocs.io/en/latest/networks.html#swinunetr
        self.model = SwinUNETR(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=n_classes,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
        )

        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)

        if freeze_encoder:
            self.freeze_encoder()

        if train_decoder_from_scratch:
            self.reset_decoder()

    def pad_input(self, x):
        """
        Pad [B, C, 49, 256, 256] -> [B, C, 64, 256, 256].
        Returns padded tensor and pad tuple for cropping back.
        """
        _, _, d, h, w = x.shape
        td, th, tw = self.internal_img_size

        if h != th or w != tw:
            raise ValueError(f"Expected HxW={(th, tw)}, got {(h, w)}")
        if d > td:
            raise ValueError(f"Input depth {d} exceeds internal depth {td}")

        pad_d = td - d
        pad_front = pad_d // 2
        pad_back = pad_d - pad_front

        pads = (0, 0, 0, 0, pad_front, pad_back)  # (Wl, Wr, Hl, Hr, Dl, Dr), from last dim to first.
        x = F.pad(x, pads, mode="replicate")
        return x, pads
    
    @staticmethod
    def crop_output(y, pads):
        """
        Crop [B, C, 64, 256, 256] back to [B, C, 49, 256, 256].
        """
        _, _, d, _, _ = y.shape
        pad_front = pads[4] # n means first n - 1 must be removed.
        pad_back = pads[5] 
        d_end = d - pad_back if pad_back > 0 else d
        return y[:, :, pad_front:d_end, :, :]
    
    def forward(self, x):
        x, pads = self.pad_input(x)
        y = self.model(x)
        y = self.crop_output(y, pads)
        return y
    
    def load_pretrained(self, pretrained_path):
        ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        # official MONAI way for SSL SwinUNETR encoder weights
        self.model.load_from(ckpt)
        print(f"Loaded encoder weights from {pretrained_path}")


    def reset_decoder(self):
        """
        Reinitialize decoder / segmentation head while keeping pretrained encoder.
        Good option if you want encoder transfer but task-specific decoder learning.
        """
        decoder_keywords = [
            "decoder",
            "decoder1",
            "decoder2",
            "decoder3",
            "decoder4",
            "decoder5",
            "out",
        ]

        for name, module in self.model.named_modules():
            lname = name.lower()
            if any(k in lname for k in decoder_keywords):
                for m in module.modules():
                    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
                        nn.init.kaiming_normal_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, (nn.InstanceNorm3d, nn.BatchNorm3d, nn.GroupNorm)):
                        if m.weight is not None:
                            nn.init.ones_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

    def freeze_encoder(self):
        """
        Freeze the Swin encoder and leave decoder/head trainable.
        """
        for name, param in self.model.named_parameters():
            if name.startswith("swinViT"):
                param.requires_grad = False



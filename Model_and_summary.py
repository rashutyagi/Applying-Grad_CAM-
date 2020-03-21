'''ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
from torchsummary import summary


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    
class Net(nn.Module):
  
  def __init__(self):
    super(Net, self).__init__()
    
    self.conv_block1= nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3),bias=False, padding=1, padding_mode='same'), 
        nn.BatchNorm2d(32),
        nn.ReLU(),
		nn.Dropout2d(0.1),
        nn.Conv2d(in_channels=32, out_channels=32 , kernel_size=(3,3), bias=False, padding=1, padding_mode='same'),
        nn.BatchNorm2d(32),
        nn.ReLU(),
		nn.Dropout2d(0.1),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), bias=False, padding=1, padding_mode='same'),
        nn.BatchNorm2d(64),
        nn.ReLU(),
		nn.Dropout2d(0.1),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), bias=False, padding=1, padding_mode='same',dilation=2),
        nn.BatchNorm2d(128),
        nn.ReLU(),
		nn.Dropout2d(0.1),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3),bias=False, padding=1, padding_mode='same'),
        nn.BatchNorm2d(256),
        nn.ReLU(),
		nn.Dropout2d(0.1))
    
    self.trans_block1= nn.Sequential(
        nn.MaxPool2d(2,2),   #RF12
        nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1,1), bias=False))
    
    self.conv_block2= nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), bias=False, padding=1, padding_mode='same'),
        nn.BatchNorm2d(128),
        nn.ReLU(),
		nn.Dropout2d(0.1),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), bias=False, padding=1, padding_mode='same'),
        nn.BatchNorm2d(128),
        nn.ReLU(),
		nn.Dropout2d(0.1),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), bias=False, padding=1, groups=128, padding_mode='same'),
        nn.BatchNorm2d(256),
        nn.ReLU(), 
		nn.Dropout2d(0.1))
    
    self.trans_block2= nn.Sequential(
        nn.MaxPool2d(2,2),
        nn.Conv2d(in_channels=256, out_channels=32, kernel_size=(1,1), bias=False))
    
    self.conv_block3= nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), bias=False, padding=1, padding_mode='same'),
        nn.BatchNorm2d(64),
        nn.ReLU(),
		nn.Dropout2d(0.1),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), bias=False, padding=1, padding_mode='same'),
        nn.BatchNorm2d(128),
        nn.ReLU(),
		nn.Dropout2d(0.1),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), bias=False, padding=1,groups=128, padding_mode='same'),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,1), bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(),
		nn.Dropout2d(0.1))
    
    self.trans_block3= nn.Sequential(
        nn.MaxPool2d(2,2),
        nn.Conv2d(in_channels=256, out_channels=16, kernel_size=(1,1), bias=False))

    self.conv_block4=nn.Sequential(
        nn.Conv2d(in_channels=16, out_channels=128, kernel_size=(3,3),bias=False, padding=1, padding_mode='same'),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.AvgPool2d(3),  #RF74
        nn.Conv2d(in_channels=128,out_channels=10, kernel_size=(1,1),bias=False)
    )
    
  def forward(self, x):
    x=self.conv_block1(x)
    x= self.trans_block1(x)
    x=self.conv_block2(x)
    x=self.trans_block2(x)
    x=self.conv_block3(x)
    x=self.trans_block3(x)
    x=self.conv_block4(x)
    x=x.view(-1,10)
    return F.log_softmax(x)


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

def disp_summary(model):
    summary(model, input_size=(3,32,32))

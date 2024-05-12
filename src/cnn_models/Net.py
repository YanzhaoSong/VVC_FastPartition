import torch
from torch import nn
from torch.nn import functional as F


class ConvBN(nn.Module):
  def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=1):
    super(ConvBN, self).__init__()
    self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    self.bn = nn.BatchNorm2d(c_out)

  def forward(self, x):
    return F.relu(self.bn(self.conv(x)))


class Residual(nn.Module):
  def __init__(self, c_in, c_out):
    super(Residual, self).__init__()
    self.pre = ConvBN(c_in, c_out)
    self.conv_bn1 = ConvBN(c_out, c_out)
    self.conv_bn2 = ConvBN(c_out, c_out)

  def forward(self, x):
    x = self.pre(x)
    x = F.max_pool2d(x, 2)
    return self.conv_bn2(self.conv_bn1(x)) + x


class Net(nn.Module):
    def __init__(self, in_channels, num_classes, planes=16, act_layer=nn.ReLU):
        super().__init__()

        self.act_layer = act_layer

        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            self.act_layer(inplace=True),
        )

        self.middle_layer = nn.Sequential(
            Residual(planes, 2*planes),
            ConvBN(2*planes, 4*planes),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(4*planes, num_classes),
        )

    def forward(self, x):
        out = self.pre(x)
        out = self.middle_layer(out)
        
        out = self.classifier(out)

        return out


def MyNet(in_channels, num_classes):
    return Net(in_channels, num_classes, planes=16)


if __name__ == "__main__":
    from torchsummary import summary

    model = MyNet(1, 6)
    summary(model, (1, 32, 32))

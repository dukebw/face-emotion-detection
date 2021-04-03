import torch


class SeparableConv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
    ):
        super(SeparableConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = torch.nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channeld, out_channels):
        super(ResidualBlock, self).__init__()

        self.residual_conv = torch.nn.Conv2d(
            in_channels=in_channeld,
            out_channels=out_channels,
            kernel_size=1,
            stride=2,
            bias=False,
        )
        self.residual_bn = torch.nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)

        self.sepConv1 = SeparableConv2d(
            in_channels=in_channeld,
            out_channels=out_channels,
            kernel_size=3,
            bias=False,
            padding=1,
        )
        self.bn1 = torch.nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
        self.relu = torch.nn.ReLU()

        self.sepConv2 = SeparableConv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            bias=False,
            padding=1,
        )
        self.bn2 = torch.nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
        self.maxp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        res = self.residual_conv(x)
        res = self.residual_bn(res)
        x = self.sepConv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.sepConv2(x)
        x = self.bn2(x)
        x = self.maxp(x)
        return res + x


class Model(torch.nn.Module):
    def __init__(self, num_classes):
        torch.nn.Module.__init__(self)

        conv_channels = 8
        self.conv1 = torch.nn.Conv2d(
            1, conv_channels, kernel_size=3, stride=1, padding=0, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(conv_channels)
        self.relu = torch.nn.ReLU(inplace=True)

        self.conv2 = torch.nn.Conv2d(
            conv_channels, conv_channels, kernel_size=3, stride=1, padding=0, bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(conv_channels)

        residual_channels = [16, 32, 64, 128]
        self.residuals = torch.nn.Sequential(
            ResidualBlock(conv_channels, residual_channels[0]),
            ResidualBlock(residual_channels[0], residual_channels[1]),
            ResidualBlock(residual_channels[1], residual_channels[2]),
            ResidualBlock(residual_channels[2], residual_channels[3]),
        )
        self.conv_out = torch.nn.Conv2d(
            residual_channels[3],
            num_classes,
            kernel_size=3,
            stride=1,
            padding=0,
            bias=True,
        )
        self.pool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.residuals(x)

        x = self.conv_out(x)

        x = self.pool(x)

        return x.squeeze()


class SimpleModel(torch.nn.Module):
    def __init__(self, num_classes):
        torch.nn.Module.__init__(self)

        conv_channels = 8
        self.conv1 = torch.nn.Conv2d(
            1, conv_channels, kernel_size=3, stride=1, padding=0, bias=True
        )
        self.relu = torch.nn.ReLU(inplace=True)

        self.conv_out = torch.nn.Conv2d(
            conv_channels,
            num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(x)

        x = self.conv_out(x)

        return x.mean(dim=(-2, -1))

import torch.nn as nn
import torch


# Basic residual block for ResNet18/34: two 3x3 convolutions
class BasicBlock(nn.Module):
    # expansion=1 if channel count stays the same
    expansion = 1

    # init(): layer definitions
    # downsample=None: solid residual; otherwise: dashed (downsampled) residual — used to change channel count of x
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(out_channel)
        # ReLU activation
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.downsample = downsample

    # forward(): defines forward pass & layer connectivity
    def forward(self, x):
        # Preserve original input for residual connection
        identity = x
        # Downsample if using dashed residual structure
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # -----------------------------------------
        out = self.conv2(out)
        out = self.bn2(out)
        # Add main branch to shortcut
        out += identity
        out = self.relu(out)

        return out


# Bottleneck block for ResNet50/101/152: 1x1 + 3x3 + 1x1 convolutions
class Bottleneck(nn.Module):
    # expansion: channel multiplier within each bottleneck (e.g., 64*4=256)
    # Bottleneck output channels = 4x input
    expansion = 1

    # init(): layer definitions
    # downsample=None: solid residual; otherwise: dashed (downsampled) residual — used to change channel count of x
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)

        self.maxpool1 = nn.MaxPool1d(kernel_size=1, stride=1)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv1d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)

        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        self.bn2 = nn.BatchNorm1d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv1d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.maxpool3 = nn.MaxPool1d(kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm1d(out_channel * self.expansion)
        # ReLU activation
        self.relu = nn.GELU()
        self.downsample = downsample

    # forward(): defines forward pass & layer connectivity
    def forward(self, x):
        # Preserve original input for residual connection
        identity = x
        # Downsample if using dashed residual structure
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.maxpool1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.maxpool2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.maxpool3(out)
        out = self.bn3(out)
        # Add main branch to shortcut
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResNet, self).__init__()
        self.b1 = Bottleneck(in_c, in_c)
        self.b2 = Bottleneck(in_c, in_c)
        self.b3 = Bottleneck(in_c, out_c)

    def forward(self, x):
        out = self.b1(x)
        out = self.b2(out)
        out = self.b3(out)
        return out

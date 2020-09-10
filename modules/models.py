import torch.nn as nn
import torch.nn.functional as F
import os, time
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ================================================================================
# Conditional version of Pix2Pix GAN
# ================================================================================

# G(z)
class cPix2PixGenerator(nn.Module):
    def __init__(self):
        super(cPix2PixGenerator, self).__init__()
        self.conv1_1_input = nn.Conv2d(4, 64, 3, 1, 1)
        self.batch_norm1_1_input = nn.BatchNorm2d(64)

        self.conv1_1_label = nn.Conv2d(4, 64, 3, 1, 1)
        self.batch_norm1_1_label = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(128, 128, 3, 1, 1)
        self.batch_norm2_1 = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.batch_norm3_1 = nn.BatchNorm2d(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.batch_norm4_1 = nn.BatchNorm2d(512)

        self.conv5_1 = nn.Conv2d(512, 256, 3, 1, 1)
        self.batch_norm5_1 = nn.BatchNorm2d(256)

        self.conv6_1 = nn.Conv2d(256, 128, 3, 1, 1)
        self.batch_norm6_1 = nn.BatchNorm2d(128)

        self.conv7_1 = nn.Conv2d(128, 4, 3, 1, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        inp_image = F.relu(self.batch_norm1_1_input(self.conv1_1_input(input)))
        inp_label = F.relu(self.batch_norm1_1_label(self.conv1_1_label(label)))
        x = torch.cat([inp_image, inp_label], 1)
        x = F.relu(self.batch_norm2_1(self.conv2_1(x)))
        x = F.relu(self.batch_norm3_1(self.conv3_1(x)))
        x = F.relu(self.batch_norm4_1(self.conv4_1(x)))
        x = F.relu(self.batch_norm5_1(self.conv5_1(x)))
        x = F.relu(self.batch_norm6_1(self.conv6_1(x)))
        x = self.conv7_1(x)

        return x

class cPix2PixDiscriminator(nn.Module):
    def __init__(self):
        super(cPix2PixDiscriminator, self).__init__()
        self.conv1_1_input = nn.Conv2d(4, 64, 5, 1, 0)
        self.batch_norm1_1_input = nn.BatchNorm2d(64)
        self.maxpool1_1_input = nn.MaxPool2d(2, 2, 0)

        self.conv1_1_label = nn.Conv2d(4, 64, 5, 1, 0)
        self.batch_norm1_1_label = nn.BatchNorm2d(64)
        self.maxpool1_1_label = nn.MaxPool2d(2, 2, 0)

        self.conv2_1 = nn.Conv2d(128, 256, 5, 1, 0)
        self.batch_norm2_1 = nn.BatchNorm2d(256)
        self.maxpool2_1 = nn.MaxPool2d(2, 2, 0)

        self.conv3_1 = nn.Conv2d(256, 128, 5, 1, 0)
        self.batch_norm3_1 = nn.BatchNorm2d(128)
        self.maxpool3_1 = nn.MaxPool2d(2, 2, 0)

        self.conv4_1 = nn.Conv2d(128, 1, 5, 1, 0)

        self.linear1 = nn.Linear(1 * 24 * 24, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 4)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        inp_image = F.leaky_relu(self.maxpool1_1_input(self.batch_norm1_1_input(self.conv1_1_input(input))), 0.2)
        inp_label = F.leaky_relu(self.maxpool1_1_label(self.batch_norm1_1_label(self.conv1_1_label(label))), 0.2)
        x = torch.cat([inp_image, inp_label], 1)
        x = F.leaky_relu(self.maxpool2_1(self.batch_norm2_1(self.conv2_1(x))), 0.2)
        x = F.leaky_relu(self.maxpool3_1(self.batch_norm3_1(self.conv3_1(x))), 0.2)
        x = F.leaky_relu(self.conv4_1(x), 0.2)
        x = F.leaky_relu(self.linear1(x.view(-1, self.num_flat_features(x))), 0.2)
        x = F.leaky_relu(self.linear2(x), 0.2)
        x = F.sigmoid(self.linear3(x))

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# Weight initialization function used by the above cPix2Pix

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

# ================================================================================
# Sample Neural Network to test stuff
# ================================================================================
class SampleNet(nn.Module):

    def __init__(self):
        super().__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(4, 1, 5, padding=2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



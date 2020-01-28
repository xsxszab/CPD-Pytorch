
import numpy as np
import torch
import torch.nn as nn
import torchvision
import PIL


class Branch_vgg16(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(  # VGG16 net conv block1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(  # VGG16 net conv block2
            nn.MaxPool2d(2, stride=2),  # 1/2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(  # VGG16 net conv block3
            nn.MaxPool2d(2, stride=2),  # 1/4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.conv4_1 = nn.Sequential(  # VGG16 net conv block4 (branch 1)
            nn.MaxPool2d(2, stride=2),  # 1/8
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.conv5_1 = nn.Sequential(  # VGG16 net conv block5 (branch 1)
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.conv4_2 = nn.Sequential(  # VGG16 net conv block4 (branch 2)
            nn.MaxPool2d(2, stride=2),  # 1/8
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.conv5_2 = nn.Sequential(  # VGG16 net conv block5 (branch 2)
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self._copy_params()  # copy parameters from pretrained vgg16 net

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x1 = self.conv4_1(x)
        x1 = self.conv5_1(x1)
        x2 = self.conv4_2(x)
        x2 = self.conv5_2(x2)

        return x1, x2

    def _copy_params(self):
        pretrained = torchvision.models.vgg16_bn(pretrained=True).features
        features = list(self.conv1.children())  # containing first 3 blocks
        features.extend(list(self.conv2.children()))
        features.extend(list(self.conv3.children()))
        features = nn.Sequential(*features)
        for layer_1, layer_2 in zip(pretrained, features):
            if (isinstance(layer_1, nn.Conv2d) and
                    isinstance(layer_2, nn.Conv2d)):
                assert layer_1.weight.size() == layer_2.weight.size()
                assert layer_1.bias.size() == layer_2.bias.size()
                layer_2.weight.data = layer_1.weight.data
                layer_2.bias.data = layer_1.bias.data
        pretrained = pretrained[24:]

        self.conv4_1[1].weight.data.copy_(pretrained[0].weight.data)
        self.conv4_1[1].bias.data.copy_(pretrained[0].bias.data)
        self.conv4_1[4].weight.data.copy_(pretrained[3].weight.data)
        self.conv4_1[4].bias.data.copy_(pretrained[3].bias.data)
        self.conv4_1[7].weight.data.copy_(pretrained[6].weight.data)
        self.conv4_1[7].bias.data.copy_(pretrained[6].bias.data)

        self.conv5_1[1].weight.data.copy_(pretrained[10].weight.data)
        self.conv5_1[1].bias.data.copy_(pretrained[10].bias.data)
        self.conv5_1[4].weight.data.copy_(pretrained[13].weight.data)
        self.conv5_1[4].bias.data.copy_(pretrained[13].bias.data)
        self.conv5_1[7].weight.data.copy_(pretrained[16].weight.data)
        self.conv5_1[7].bias.data.copy_(pretrained[16].bias.data)

        self.conv4_2[1].weight.data.copy_(pretrained[0].weight.data)
        self.conv4_2[1].bias.data.copy_(pretrained[0].bias.data)
        self.conv4_2[4].weight.data.copy_(pretrained[3].weight.data)
        self.conv4_2[4].bias.data.copy_(pretrained[3].bias.data)
        self.conv4_2[7].weight.data.copy_(pretrained[6].weight.data)

        self.conv5_2[1].weight.data.copy_(pretrained[10].weight.data)
        self.conv5_2[1].bias.data.copy_(pretrained[10].bias.data)
        self.conv5_2[4].weight.data.copy_(pretrained[13].weight.data)
        self.conv5_2[4].bias.data.copy_(pretrained[13].bias.data)
        self.conv5_2[7].weight.data.copy_(pretrained[16].weight.data)
        self.conv5_2[7].bias.data.copy_(pretrained[16].bias.data)


class RFB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = nn.Conv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = nn.Conv2d(in_channel, out_channel, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = torch.cat((x0, x1, x2, x3), 1)
        x_cat = self.conv_cat(x_cat)

        x = self.relu(x_cat + self.conv_res(x))
        return x


def make_guassian_window(n=16, sigma=3):
    """Generate a gaussian filter with shape n*n, mu=0 and input sigma.

        :param n: filter size.
        :param sigma: sigma of 2D gaussian function.
    """
    nn = int((n - 1) / 2)
    a = np.asarray([[x**2 + y**2 for x in range(-nn, nn + 1)]
                    for y in range(-nn, nn + 1)])
    # np.asarray可以将输入转化为np.array, 这里输入为一个列表推导式
    return np.exp(-a / (2 * sigma**2))


def min_max_norm(feature):
    """Apply min_max normalization.
    """
    max_ = feature.max(3)[0].max(2)[0].unsqueeze(
        2).unsqueeze(3).expand_as(feature)
    min_ = feature.min(3)[0].min(2)[0].unsqueeze(
        2).unsqueeze(3).expand_as(feature)
    feature = feature - min_
    return feature.div(max_ - min_ + 1e-8)


class HAM(nn.Module):
    """Holistic attention module.
    """

    def __init__(self):
        super().__init__()
        gaussian_kernel = np.float32(make_guassian_window(31, 4))
        gaussian_kernel = gaussian_kernel[np.newaxis, np.newaxis, ...]
        self.gaussian_kernel = torch.nn.Parameter(
            torch.from_numpy(gaussian_kernel))

    def forward(self, attention, x):
        soft_attention = nn.functional.conv2d(
            attention, self.gaussian_kernel, padding=15)
        soft_attention = min_max_norm(soft_attention)
        x = torch.mul(x, soft_attention.max(attention))
        return x


class Aggregation(nn.Module):
    """Feature aggration module.
    """
    def __init__(self, channel):
        super().__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=True)
        self.conv_upsample1 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = nn.Conv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = nn.Conv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = nn.Conv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = nn.Conv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, x1, x2, x3):
        # x1: 1/16 x2: 1/8 x3: 1/4
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
            * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class CPD(nn.Module):
    """Main structure of CPD net.
    """

    def __init__(self):
        super().__init__()
        self.branch_vgg = Branch_vgg16()
        self.rfb3_1 = RFB(256, 32)
        self.rfb4_1 = RFB(512, 32)
        self.rfb5_1 = RFB(512, 32)
        self.agg1 = Aggregation(32)

        self.rfb3_2 = RFB(256, 32)
        self.rfb4_2 = RFB(512, 32)
        self.rfb5_2 = RFB(512, 32)
        self.agg2 = Aggregation(32)
        self.ham = HAM()
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.branch_vgg.conv1(x)
        x = self.branch_vgg.conv2(x)
        x = self.branch_vgg.conv3(x)

        x1_1 = x
        x1_2 = self.branch_vgg.conv4_1(x)
        x1_3 = self.branch_vgg.conv5_1(x1_2)
        x1_1 = self.rfb3_1(x1_1)
        x1_2 = self.rfb4_1(x1_2)
        x1_3 = self.rfb5_1(x1_3)
        output1 = self.agg1(x1_3, x1_2, x1_1)

        x2_1 = self.ham(output1.sigmoid(), x)
        x2_2 = self.branch_vgg.conv4_1(x2_1)
        x2_3 = self.branch_vgg.conv5_2(x2_2)
        x2_1 = self.rfb3_2(x2_1)
        x2_2 = self.rfb4_2(x2_2)
        x2_3 = self.rfb5_2(x2_3)
        output2 = self.agg2(x2_3, x2_2, x2_1)
        return self.upsample(output1.sigmoid()), self.upsample(output2.sigmoid())


if __name__ == '__main__':
    test = CPD()
    img = PIL.Image.open('test.jpg', 'r')
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    img = np.transpose(img, (0, 3, 1, 2))
    img = torch.from_numpy(img).float()
    output1, output2 = test.forward(img)

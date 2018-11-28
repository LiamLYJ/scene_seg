import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as nn_F
import numpy as np



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

class vgg16_get(nn.Module):
    def __init__(self, use_bn = True):
        super(vgg16_get, self).__init__()
        self.vgg16_model = models.vgg16_bn() if use_bn else models.vgg16()
        self.features = {}
        self.feature_count = 1

        def extract_feature_hook(module, input, output):
            # self.features['conv_%s'%(self.feature_count)] = output[0]
            self.features['conv_%s'%(self.feature_count)] = output
            self.feature_count += 1
            # print (output.shape)

        layers_for_features = list(self.vgg16_model.features.children())
        # print (self.vgg16_model.features)
        # get select features
        select = [2,9,16,26,36] if use_bn else [1, 6,11,18,25,]
        for index in select:
            layers_for_features[index].register_forward_hook(extract_feature_hook)

    def forward(self, images):
        self.feature_count = 1
        self.features = {}
        self.vgg16_model.features(images)
        return self.features


class encoder_net(nn.Module):
    def __init__(self,):
        super(encoder_net, self).__init__()
        self.vgg16 = vgg16_get()

        self.conv1 = nn.Conv2d(512, 512, kernel_size = 1)
        self.res1 = Bottleneck(512, 512)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size = 1)
        self.res2 = Bottleneck(512, 512)
        self.conv3 = nn.Conv2d(768, 256, kernel_size = 1)
        self.res3 = Bottleneck(256, 256)
        self.conv4 = nn.Conv2d(384, 128, kernel_size = 1)
        self.res4 = Bottleneck(128, 128)
        self.conv5 = nn.Conv2d(192, 128, kernel_size = 1)
        self.res5 = Bottleneck(128, 128)
        self.conv_out = nn.Conv2d(128, 64, kernel_size = 1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.activate = nn.ReLU()

    def forward(self, images):
        vgg_features = self.vgg16(images)
        vgg_conv1 = vgg_features['conv_1']
        vgg_conv2 = vgg_features['conv_2']
        vgg_conv3 = vgg_features['conv_3']
        vgg_conv4 = vgg_features['conv_4']
        vgg_conv5 = vgg_features['conv_5']

        x = self.conv1(vgg_conv5)
        x = self.activate(x)
        x = self.res1(x)
        x = self.upsample(x)
        x = torch.cat((x, vgg_conv4), 1)

        x = self.conv2(x)
        x = self.activate(x)
        x = self.res2(x)
        x = self.upsample(x)
        x = torch.cat((x, vgg_conv3), 1)

        x = self.conv3(x)
        x = self.activate(x)
        x = self.res3(x)
        x = self.upsample(x)
        x = torch.cat((x, vgg_conv2), 1)

        x = self.conv4(x)
        x = self.activate(x)
        x = self.res4(x)
        x = self.upsample(x)
        x = torch.cat((x, vgg_conv1), 1)

        x = self.conv5(x)
        x = self.activate(x)
        x = self.res5(x)

        y = self.conv_out(x)

        return y, vgg_features


class decoder_net(nn.Module):
    def __init__(self,):
        super(decoder_net, self).__init__()

        self.conv1 = nn.Conv2d(512, 128, kernel_size = 1)
        self.res_conv1 = nn.Conv2d(129, 64, kernel_size = 1)
        self.res1 = Bottleneck(64, 64)
        self.conv2 = nn.Conv2d(512, 64, kernel_size = 1)
        self.res_conv2 = nn.Conv2d(129, 64, kernel_size = 1)
        self.res2 = Bottleneck(64, 64)
        self.conv3 = nn.Conv2d(256, 64, kernel_size = 1)
        self.res_conv3 = nn.Conv2d(129, 64, kernel_size = 1)
        self.res3 = Bottleneck(64, 64)
        self.conv4 = nn.Conv2d(128, 64, kernel_size = 1)
        self.res_conv4 = nn.Conv2d(129, 64, kernel_size = 1)
        self.res4 = Bottleneck(64, 64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size = 1)
        self.res_conv5 = nn.Conv2d(129, 64, kernel_size = 1)
        self.res5 = Bottleneck(64, 64)
        self.conv_out = nn.Conv2d(64, 1, kernel_size = 1)

        self.downsample_16 = nn.AdaptiveAvgPool2d((16,16))
        self.downsample_32 = nn.AdaptiveAvgPool2d((32,32))
        self.downsample_64 = nn.AdaptiveAvgPool2d((64,64))
        self.downsample_128 = nn.AdaptiveAvgPool2d((128,128))

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.activate = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, correlations, vgg_features):
        vgg = self.conv1(vgg_features['conv_5'])
        vgg = self.activate(vgg)
        x = torch.cat((vgg, self.downsample_16(correlations)), 1)
        x = self.res_conv1(x)
        x = self.res1(x)
        x = self.upsample(x)

        vgg = self.conv2(vgg_features['conv_4'])
        vgg = self.activate(vgg)
        x = torch.cat((x, vgg, self.downsample_32(correlations)), 1)
        x = self.res_conv2(x)
        x = self.res2(x)
        x = self.upsample(x)

        vgg = self.conv3(vgg_features['conv_3'])
        vgg = self.activate(vgg)
        x = torch.cat((x, vgg, self.downsample_64(correlations)), 1)
        x = self.res_conv3(x)
        x = self.res3(x)
        x = self.upsample(x)

        vgg = self.conv4(vgg_features['conv_2'])
        vgg = self.activate(vgg)
        x = torch.cat((x, vgg, self.downsample_128(correlations)), 1)
        x = self.res_conv4(x)
        x = self.res4(x)
        x = self.upsample(x)

        vgg = self.conv5(vgg_features['conv_1'])
        vgg = self.activate(vgg)
        x = torch.cat((x, vgg, correlations), 1)
        x = self.res_conv5(x)
        x = self.res5(x)

        x = self.conv_out(x)
        y = self.sigmoid(x)

        return y, x


def test_cov():
    filt = torch.randn(2, 10, 5, 5)
    input = torch.randn(3,10,24,24)
    output = F.conv2d(input, filt )
    print (output.shape)

if __name__ == '__main__':
    w = 256
    s_w = 64
    input_img = np.ones(w*w*3).reshape([1,3,w,w])
    input_img = torch.from_numpy(input_img)
    input_img = input_img.type(torch.FloatTensor)

    input_patch = np.ones(s_w*s_w*3).reshape([1,3,s_w,s_w])
    input_patch = torch.from_numpy(input_patch)
    input_patch = input_patch.type(torch.FloatTensor)

    # model_get_feature = vgg16_get()
    # output_feature = model_get_feature(input_img)
    # print (output_feature)

    model_encoder = encoder_net()
    output_encoder_img, vgg_features = model_encoder(input_img)
    output_encoder_patch, _ = model_encoder(input_patch)

    # print (output_encoder_img.shape)
    # print (output_encoder_patch.shape)

    filt_adp = nn.AdaptiveAvgPool2d((5,5))
    filt = filt_adp(output_encoder_patch)
    correlations = nn_F.conv2d(output_encoder_img, filt, stride = 1, padding = 2)

    print (correlations.shape)

    model_decoder = decoder_net()
    final_output, _ = model_decoder(correlations, vgg_features)
    print (final_output.shape)

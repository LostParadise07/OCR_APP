import torch.nn as nn

from auth_app.recognizor.modules.cnn.densenet import DenseNet
from auth_app.recognizor.modules.cnn.hrnet import HRNet
from auth_app.recognizor.modules.cnn.inception_unet import InceptionUNet
from auth_app.recognizor.modules.cnn.rcnn import RCNN
from auth_app.recognizor.modules.cnn.resnet import ResNet
from auth_app.recognizor.modules.cnn.resunet import ResUnet
from auth_app.recognizor.modules.cnn.unet_attn import AttnUNet
from auth_app.recognizor.modules.cnn.unet_plus_plus import NestedUNet
from auth_app.recognizor.modules.cnn.unet import UNet
from auth_app.recognizor.modules.cnn.vgg import VGG

class DenseNet_FeatureExtractor(nn.Module):
    def __init__(self, input_channel=1, output_channel=512):
        super(DenseNet_FeatureExtractor, self).__init__()
        self.ConvNet = DenseNet(input_channel, output_channel)

    def forward(self, input):
        return self.ConvNet(input)

class HRNet_FeatureExtractor(nn.Module):
    def __init__(self, input_channel=1, output_channel=32):
        super(HRNet_FeatureExtractor, self).__init__()
        self.ConvNet = HRNet(input_channel, output_channel)

    def forward(self, input):
        return self.ConvNet(input)

class InceptionUNet_FeatureExtractor(nn.Module):
    def __init__(self, input_channel=1, output_channel=512):
        super(InceptionUNet_FeatureExtractor, self).__init__()
        self.ConvNet = InceptionUNet(input_channel, output_channel)

    def forward(self, input):
        return self.ConvNet(input)

class RCNN_FeatureExtractor(nn.Module):
    def __init__(self, input_channel=1, output_channel=512):
        super(RCNN_FeatureExtractor, self).__init__()
        self.ConvNet = RCNN(input_channel, output_channel)

    def forward(self, input):
        return self.ConvNet(input)

class ResNet_FeatureExtractor(nn.Module):
    def __init__(self, input_channel=1, output_channel=512):
        super(ResNet_FeatureExtractor, self).__init__()
        self.ConvNet = ResNet(input_channel, output_channel)

    def forward(self, input):
        return self.ConvNet(input)
    
class ResUnet_FeatureExtractor(nn.Module):
    def __init__(self, input_channel=1, output_channel=512):
        super(ResUnet_FeatureExtractor, self).__init__()
        self.ConvNet = ResUnet(input_channel, output_channel)

    def forward(self, input):
        return self.ConvNet(input)

class AttnUNet_FeatureExtractor(nn.Module):
    def __init__(self, input_channel=1, output_channel=512):
        super(AttnUNet_FeatureExtractor, self).__init__()
        self.ConvNet = AttnUNet(input_channel, output_channel)

    def forward(self, input):
        return self.ConvNet(input)

class UNet_FeatureExtractor(nn.Module):
    def __init__(self, input_channel=1, output_channel=512):
        super(UNet_FeatureExtractor, self).__init__()
        self.ConvNet = UNet(input_channel, output_channel)

    def forward(self, input):
        return self.ConvNet(input)

class UNetPlusPlus_FeatureExtractor(nn.Module):
    def __init__(self, input_channel=1, output_channel=512):
        super(UNetPlusPlus_FeatureExtractor, self).__init__()
        self.ConvNet = NestedUNet(input_channel, output_channel)

    def forward(self, input):
        return self.ConvNet(input)

class VGG_FeatureExtractor(nn.Module):
    def __init__(self, input_channel=1, output_channel=512):
        super(VGG_FeatureExtractor, self).__init__()
        self.ConvNet = VGG(input_channel, output_channel)

    def forward(self, input):
        return self.ConvNet(input)
    
# x = torch.randn(1, 1, 32, 400)
# model = UNet_FeatureExtractor()
# out = model(x)
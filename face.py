import torch
import torch.nn as nn
import face_alignment
from light_cnn import LightCNN_29Layers_v2
from torch.utils.model_zoo import load_url
import torch.nn.functional as F


class FaceLandmarks(nn.Module):

    def __init__(self):
        super().__init__()
        model = face_alignment.models.FAN(4)
        url = 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar'
        fan_weights = load_url(url, map_location='cpu')
        model.load_state_dict(fan_weights)
        self.net = model

    def forward(self, x):
        x = (x + 1) / 2.
        x = nn.AdaptiveAvgPool2d((256, 256))(x)#resize to 256x256
        _, _, _, hm = self.net(x)
        return hm


class FaceDescriptor(nn.Module):

    def __init__(self, path="LightCNN_29Layers_V2_checkpoint.pth.tar", device="cpu"):
        super().__init__()
        model = LightCNN_29Layers_v2(num_classes=80013)
        checkpoint = torch.load(path, map_location="cpu")
        ck = checkpoint['state_dict']
        ck_ = {}
        for k, v in ck.items():
            ck_[k.replace("module.", "")] = v
        model.load_state_dict(ck_)
        model.to(device)
        self.net = model
        self.latent_size = 256

    def forward(self, x):
        x = (x + 1) / 2.
        x = x.mean(dim=1, keepdim=True) #grayscale
        x = nn.AdaptiveAvgPool2d((128, 128))(x) #resize to 128x128
        return self.net(x)

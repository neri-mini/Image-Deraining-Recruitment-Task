import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

class EasyContrastiveLoss(nn.Module):

    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.model = vgg[:35]
        for p in self.model.parameters():
            p.requires_grad = False
        self.pool = nn.AdaptiveAvgPool2d(1)

    def prepare(self, x):
        x_normed = (x - self.mean) / self.std
        x_feat = self.model(x_normed)
        out = self.pool(x_feat).flatten(1)
        return F.normalize(out, p=2, dim=1)

    def forward(self, generated, rainy, truth):
        generated = self.prepare(generated)
        rainy = self.prepare(rainy)
        truth = self.prepare(truth)
        cos1 = (generated * rainy).sum(dim=1).mean()
        cos2 = (generated * truth).sum(dim=1).mean()
        return cos1 - cos2

class InfoNCEContrastiveLoss(nn.Module):
    def __init__(self, tau=0.07, include_rainy=True, center=False):
        super().__init__()
        self.tau = tau
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.model = vgg[:35]
        for p in self.model.parameters():
            p.requires_grad = False
        self.pool = nn.AdaptiveAvgPool2d(1)
    prepare = EasyContrastiveLoss.prepare

    def forward(self, generated, rainy, truth):
        q = self.prepare(generated)           
        k_truth = self.prepare(truth)         
        k_rainy = self.prepare(rainy)
        # 这里用truth和rainy当字典，一个正样本，2N-1个负样本
        keys = (torch.cat([k_truth, k_rainy], dim=0))        
        logits = q @ keys.t() / self.tau                     
        labels = torch.arange(q.size(0), device=q.device)    
        return F.cross_entropy(logits, labels)



import torch 
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights = VGG19_Weights.IMAGENET1K_V1).features.eval()
        self.register_buffer("mean",torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.criterion = nn.L1Loss()
        self.model = vgg[:35]
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self,generated, ground_truth):
        generated_normed = (generated - self.mean) / self.std
        ground_truth_normed = (ground_truth - self.mean) / self.std
        return self.criterion(self.model(generated_normed),self.model(ground_truth_normed))



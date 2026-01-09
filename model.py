import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg11


class VGG11Embedding(nn.Module):
    def __init__(self, embedding_size, weights=None):
        super(VGG11Embedding, self).__init__()
        vgg = vgg11(weights=weights)
        self.features = vgg.features
        self.linear = nn.Linear(512, embedding_size)
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = F.normalize(x, p=2, dim=1)
        return x

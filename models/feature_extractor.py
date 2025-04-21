import torch.nn as nn
import torchvision.models as models

class SVDDModel(nn.Module):
    """
    Feature extractor for Deep SVDD: ResNet backbone without the final classification head.
    """
    def __init__(self, backbone_name):
        super().__init__()
        base = getattr(models, backbone_name)(pretrained=False)
        # remove final FC layer
        self.features = nn.Sequential(*list(base.children())[:-1])

    def forward(self, x):
        x = self.features(x)              # [B, C, 1, 1]
        return x.view(x.size(0), -1)      # [B, C]
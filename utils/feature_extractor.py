import torch
import torch.nn as nn
import torchvision.models as models


def get_feature_backbone(
    backbone_name: str,
    pretrained: bool = False,
    in_channels: int = 3,
    strip_avgpool: bool = True,
) -> nn.Module:
    """
    Returns a feature extractor:
      1) loads torchvision.backbone_name(pretrained=pretrained),
      2) optionally adapts conv1 to `in_channels`,
      3) strips off avgpool+fc (or just fc if strip_avgpool=False).
    """
    backbone = getattr(models, backbone_name)(pretrained=pretrained)
    # adapt first conv if needed
    if in_channels != 3:
        orig = backbone.conv1
        new = nn.Conv2d(
            in_channels,
            orig.out_channels,
            kernel_size=orig.kernel_size,
            stride=orig.stride,
            padding=orig.padding,
            bias=(orig.bias is not None),
        )
        with torch.no_grad():
            new.weight[:, :3] = orig.weight
            new.weight[:, 3:] = orig.weight.mean(dim=1, keepdim=True)
        backbone.conv1 = new

    # chop off the classification head
    children = list(backbone.children())
    if strip_avgpool:
        # remove avgpool and fc
        feat = nn.Sequential(*children[:-2])
    else:
        # remove only fc
        feat = nn.Sequential(*children[:-1])
    return feat


def get_classifier_backbone(
    backbone_name: str, num_classes: int, pretrained: bool = True
) -> nn.Module:
    """
    Returns a torchvision classifier (e.g. ResNet) with its fc replaced
    to output `num_classes` logits.
    """
    model = getattr(models, backbone_name)(pretrained=pretrained)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model

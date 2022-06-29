from torchvision.models.resnet import *
import torch.nn as nn

class build_ResNet18:
    def __init__(self):
        pass

    def build(self, num_classes):
        ResNet18 = resnet18()

        num_ftrs = ResNet18.fc.in_features
        ResNet18.fc = nn.Linear(num_ftrs, num_classes)
        return ResNet18
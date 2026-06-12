import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class EarSiameseModel(nn.Module):
    def __init__(self, embed_dim=256):
        super(EarSiameseModel, self).__init__()

        self.convnext     = timm.create_model('convnext_atto',   pretrained=False, num_classes=0)
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        self.resnet       = timm.create_model('resnet18',        pretrained=False, num_classes=0)

        # convnext_atto=320, efficientnet_b0=1280, resnet18=512
        combined_dim = 2112

        self.projector = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embed_dim),
        )

    def forward_one(self, x):
        f1 = self.convnext(x)
        f2 = self.efficientnet(x)
        f3 = self.resnet(x)
        out = torch.cat([f1, f2, f3], dim=1)
        return F.normalize(self.projector(out), dim=1)

    def forward(self, x1, x2):
        e1 = self.forward_one(x1)
        e2 = self.forward_one(x2)
        return e1, e2

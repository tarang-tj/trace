"""Shared Siamese pyramid with learned pixel-wise intervention survival pooling."""
import torch
from torch import nn
from torch.nn import functional as F


def block(input_channels, output_channels, stride=1):
    return nn.Sequential(nn.Conv2d(input_channels, output_channels, 3, stride, 1),
                         nn.GroupNorm(4, output_channels), nn.SiLU(),
                         nn.Conv2d(output_channels, output_channels, 3, 1, 1),
                         nn.GroupNorm(4, output_channels), nn.SiLU())


class EvidencePyramid(nn.Module):
    def __init__(self, width=12):
        super().__init__()
        self.encoder = nn.ModuleList([block(3, width), block(width, width * 2, 2),
                                      block(width * 2, width * 4, 2)])
        self.fusion = nn.ModuleList([block(width * 3, width), block(width * 6, width * 2),
                                     block(width * 12, width * 4)])
        self.up1 = block(width * 6, width * 2)
        self.up0 = block(width * 3, width)
        self.evidence = nn.Conv2d(width, 1, 1)
        self.trust = nn.Conv2d(width, 1, 1)

    def forward(self, reference, query):
        a, b, features = reference, query, []
        for encoder, fusion in zip(self.encoder, self.fusion):
            a, b = encoder(a), encoder(b)
            features.append(fusion(torch.cat([a, b, (a - b).abs()], 1)))
        x = self.up1(torch.cat([F.interpolate(features[2], size=features[1].shape[-2:],
                                              mode="bilinear", align_corners=False), features[1]], 1))
        x = self.up0(torch.cat([F.interpolate(x, size=features[0].shape[-2:],
                                              mode="bilinear", align_corners=False), features[0]], 1))
        return self.evidence(x), self.trust(x)


class TRACE(nn.Module):
    def __init__(self, width=12, lattice=True):
        super().__init__()
        self.width, self.lattice = width, lattice
        self.pyramid = EvidencePyramid(width)
        # Learned pooling preserves persistence, contradiction, and branch trust.
        self.survival = nn.Sequential(nn.Conv2d(5, 12, 1), nn.SiLU(), nn.Conv2d(12, 1, 1))
        self.error_head = nn.Sequential(nn.Conv2d(5, 12, 1), nn.SiLU(), nn.Conv2d(12, 1, 1))

    def forward(self, reference, vertices):
        n, k, c, h, w = vertices.shape
        if not self.lattice:
            vertices, k = vertices[:, :1], 1
        a = reference[:, None].expand(-1, k, -1, -1, -1).reshape(n * k, c, h, w)
        evidence, trust = self.pyramid(a, vertices.reshape(n * k, c, h, w))
        evidence, trust = evidence.reshape(n, k, h, w), trust.reshape(n, k, h, w)
        weights = trust.softmax(1)
        weighted = (weights * evidence).sum(1, keepdim=True)
        mean = evidence.mean(1, keepdim=True)
        lower, upper = evidence.amin(1, keepdim=True), evidence.amax(1, keepdim=True)
        variance = evidence.var(1, unbiased=False, keepdim=True)
        statistics = torch.cat([weighted, mean, lower, upper, variance], 1)
        logits = weighted + self.survival(statistics) if self.lattice else weighted
        error = self.error_head(statistics.detach())
        return {"logits": logits, "branches": evidence, "weights": weights, "error_logits": error}


class ActionValueNet(nn.Module):
    """Predict measured re-observation Brier reduction from observable diagnostics."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(12, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 3))

    def forward(self, diagnostics):
        return self.net(diagnostics)


def segmentation_loss(output, target, valid):
    logits = output["logits"]
    prob = logits.sigmoid()
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * valid).sum() / valid.sum()
    intersection = (prob * target * valid).sum((1, 2, 3))
    dice = 1 - ((2 * intersection + 1) / ((prob * valid).sum((1, 2, 3)) +
                                         (target * valid).sum((1, 2, 3)) + 1)).mean()
    branches = output["branches"]
    # Auxiliary segmentation supervision keeps each intervention vertex interpretable.
    aux = (F.binary_cross_entropy_with_logits(branches, target.expand_as(branches), reduction="none") *
           valid).sum() / (valid.sum() * branches.shape[1])
    error_target = ((prob.detach() >= 0.5) != (target >= 0.5)).float()
    error = (F.binary_cross_entropy_with_logits(output["error_logits"], error_target, reduction="none") * valid).sum() / valid.sum()
    return bce + dice + 0.25 * aux + 0.15 * error

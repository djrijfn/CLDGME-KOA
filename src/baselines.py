import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def _get_torchvision_weights(backbone_name, pretrained=True):
    if not bool(pretrained):
        return None
    name = str(backbone_name).lower()
    if name == 'resnet18':
        return models.ResNet18_Weights.IMAGENET1K_V1
    if name == 'resnet50':
        return models.ResNet50_Weights.IMAGENET1K_V2
    if name == 'vgg19':
        return models.VGG19_Weights.IMAGENET1K_V1
    if name == 'vgg19_bn':
        return models.VGG19_BN_Weights.IMAGENET1K_V1
    if name == 'densenet121':
        return models.DenseNet121_Weights.IMAGENET1K_V1
    if name == 'convnext_tiny':
        return models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    if name == 'swin_t':
        return models.Swin_T_Weights.IMAGENET1K_V1
    if name == 'vit_b_16':
        return models.ViT_B_16_Weights.IMAGENET1K_V1
    raise ValueError(f'unsupported backbone: {backbone_name}')


class ImageFeatureExtractor(nn.Module):
    """Torchvision image encoder that returns one global feature vector."""

    def __init__(self, backbone_name='resnet18', pretrained=True):
        super().__init__()
        name = str(backbone_name).lower()
        weights = _get_torchvision_weights(name, pretrained=pretrained)
        self.backbone_name = name

        if name in ('resnet18', 'resnet50'):
            model_fn = models.resnet18 if name == 'resnet18' else models.resnet50
            model = model_fn(weights=weights)
            self.feature_dim = model.fc.in_features
            model.fc = nn.Identity()
            self.encoder = model
            return

        if name in ('vgg19', 'vgg19_bn'):
            model_fn = models.vgg19 if name == 'vgg19' else models.vgg19_bn
            model = model_fn(weights=weights)
            self.feature_dim = model.classifier[-1].in_features
            model.classifier[-1] = nn.Identity()
            self.encoder = model
            return

        if name == 'densenet121':
            model = models.densenet121(weights=weights)
            self.feature_dim = model.classifier.in_features
            model.classifier = nn.Identity()
            self.encoder = model
            return

        if name == 'convnext_tiny':
            model = models.convnext_tiny(weights=weights)
            self.feature_dim = model.classifier[2].in_features
            model.classifier[2] = nn.Identity()
            self.encoder = model
            return

        if name == 'swin_t':
            model = models.swin_t(weights=weights)
            self.feature_dim = model.head.in_features
            model.head = nn.Identity()
            self.encoder = model
            return

        if name == 'vit_b_16':
            model = models.vit_b_16(weights=weights)
            self.feature_dim = model.heads.head.in_features
            model.heads = nn.Identity()
            self.encoder = model
            return

        raise ValueError(f'unsupported backbone: {backbone_name}')

    def forward(self, images):
        return self.encoder(images)


class BackboneClassifier(nn.Module):
    """Single-image CNN classifier baseline."""

    def __init__(self, backbone_name='resnet50', num_classes=5, pretrained=True, dropout=0.0):
        super().__init__()
        self.feature_extractor = ImageFeatureExtractor(backbone_name, pretrained=pretrained)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.classifier = nn.Linear(self.feature_extractor.feature_dim, int(num_classes))

    def forward(self, images, return_features=False):
        features = self.feature_extractor(images)
        logits = self.classifier(self.dropout(features))
        if return_features:
            return logits, features
        return logits


class DeepSiameseCNN(nn.Module):
    """Adapted Siamese CNN baseline for single-knee folders.

    The original Tiulpin et al. method exploits bilateral knee inputs. The
    public folder dataset used in this project does not expose paired knees, so
    this implementation uses a shared encoder on the original image and its
    horizontal flip. The fusion head keeps the Siamese inductive bias while
    preserving the same train/val/test split as the other baselines.
    """

    def __init__(
        self,
        backbone_name='resnet18',
        num_classes=5,
        pretrained=True,
        hidden_dim=512,
        dropout=0.3,
    ):
        super().__init__()
        self.feature_extractor = ImageFeatureExtractor(backbone_name, pretrained=pretrained)
        feat_dim = self.feature_extractor.feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 4, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )

    def forward(self, images):
        left = self.feature_extractor(images)
        right = self.feature_extractor(torch.flip(images, dims=[3]))
        fused = torch.cat([left, right, torch.abs(left - right), left * right], dim=1)
        return self.classifier(fused)


class AdjustableOrdinalLoss(nn.Module):
    """Chen et al. adjustable ordinal loss for softmax KL-grade classifiers.

    For a true grade m and predicted softmax probabilities p_i, the loss is
    sum_i Wbar[i, m] p_i, with zero diagonal and larger off-diagonal penalties
    for larger grade distances. The squared variant follows the paper's
    fine-tuning setup.
    """

    def __init__(self, num_classes=5, distance_power=1.0, offset=1.0, squared=True):
        super().__init__()
        self.num_classes = int(num_classes)
        grades = torch.arange(self.num_classes, dtype=torch.float32)
        distance = torch.abs(grades[:, None] - grades[None, :])
        penalty = distance.pow(float(distance_power)) + float(offset)
        penalty[distance == 0] = 0.0
        self.register_buffer('penalty', penalty)
        self.squared = bool(squared)

    def forward(self, logits, labels):
        probs = torch.softmax(logits, dim=1)
        weights = self.penalty[:, labels].t()
        loss = (weights * probs).sum(dim=1)
        if self.squared:
            loss = loss.pow(2)
        return loss.mean()


class CoralOrdinalClassifier(nn.Module):
    """Backbone + CORAL rank-consistent ordinal regression baseline."""

    def __init__(self, backbone_name='resnet50', num_classes=5, pretrained=True, dropout=0.0):
        super().__init__()
        self.feature_extractor = ImageFeatureExtractor(backbone_name, pretrained=pretrained)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.ordinal_head = nn.Linear(self.feature_extractor.feature_dim, int(num_classes) - 1)

    def forward(self, images):
        features = self.feature_extractor(images)
        return self.ordinal_head(self.dropout(features))


class StrictCoralLayer(nn.Module):
    """Shared-score ordinal head with ordered trainable biases."""

    def __init__(self, in_dim, num_classes=5):
        super().__init__()
        self.num_thresholds = int(num_classes) - 1
        if self.num_thresholds < 1:
            raise ValueError('num_classes must be at least 2')
        self.score = nn.Linear(int(in_dim), 1)
        self.bias_anchor = nn.Parameter(torch.zeros(1))
        self.bias_steps = nn.Parameter(torch.zeros(self.num_thresholds))

    def ordered_bias(self):
        if self.num_thresholds == 1:
            return self.bias_anchor
        deltas = F.softplus(self.bias_steps[1:])
        tail = self.bias_anchor - torch.cumsum(deltas, dim=0)
        return torch.cat([self.bias_anchor, tail], dim=0)

    def forward(self, features):
        score = self.score(features)
        return score + self.ordered_bias().view(1, -1)


class StrictCoralOrdinalClassifier(nn.Module):
    """Rank-consistent CORAL-style baseline with a shared score."""

    def __init__(self, backbone_name='resnet50', num_classes=5, pretrained=True, dropout=0.0):
        super().__init__()
        self.feature_extractor = ImageFeatureExtractor(backbone_name, pretrained=pretrained)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.ordinal_head = StrictCoralLayer(
            in_dim=self.feature_extractor.feature_dim,
            num_classes=num_classes,
        )

    def forward(self, images):
        features = self.feature_extractor(images)
        return self.ordinal_head(self.dropout(features))


class CornOrdinalClassifier(nn.Module):
    """CORN-style conditional ordinal baseline."""

    def __init__(self, backbone_name='resnet50', num_classes=5, pretrained=True, dropout=0.0):
        super().__init__()
        self.feature_extractor = ImageFeatureExtractor(backbone_name, pretrained=pretrained)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.ordinal_head = nn.Linear(self.feature_extractor.feature_dim, int(num_classes) - 1)

    def forward(self, images):
        features = self.feature_extractor(images)
        return self.ordinal_head(self.dropout(features))


class SwinDualAttentionClassifier(nn.Module):
    """Swin-T classifier with lightweight channel-spatial attention.

    This is a controlled reimplementation for comparison with recent
    attention-enhanced KOA networks, not an official reproduction of any one
    released codebase.
    """

    def __init__(self, num_classes=5, pretrained=True, dropout=0.2, reduction=8):
        super().__init__()
        weights = models.Swin_T_Weights.IMAGENET1K_V1 if bool(pretrained) else None
        model = models.swin_t(weights=weights)
        self.features = model.features
        self.norm = model.norm
        self.permute = model.permute
        channels = model.head.in_features
        hidden = max(32, channels // int(reduction))
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            nn.Linear(channels, int(num_classes)),
        )

    def forward(self, images):
        x = self.features(images)
        x = self.norm(x)
        x = self.permute(x)
        x = x * self.channel_gate(x)
        spatial = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        x = x * self.spatial_gate(spatial)
        return self.classifier(self.pool(x))


class BaselineOutputAdapter(nn.Module):
    """Expose baseline logits through the same dict interface as CLDGME-KOA."""

    def __init__(self, model, output_kind='class'):
        super().__init__()
        self.model = model
        self.output_kind = str(output_kind).lower()
        if self.output_kind not in ('class', 'ordinal'):
            raise ValueError(f'unsupported baseline output kind: {output_kind}')

    def forward(self, images, return_dict=False):
        logits = self.model(images)
        out = {
            'text_logits': None,
            'ordinal_logits': logits if self.output_kind == 'ordinal' else None,
            'class_logits': logits if self.output_kind == 'class' else None,
        }
        if return_dict:
            return out
        return logits


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ChannelSpatialAttention(nn.Module):
    """Light CBAM-style attention used in the OsteoHRNet adaptation."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(8, int(channels) // int(reduction))
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x * self.channel(x)
        mean_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        return x * self.spatial(torch.cat([mean_map, max_map], dim=1))


class OsteoHRNetAdapted(nn.Module):
    """Attentive multi-scale high-resolution CNN adapted from OsteoHRNet.

    The original paper describes an attentive multi-scale deep CNN for KOA
    severity prediction. This implementation keeps the high-resolution stream,
    parallel lower-resolution branches, multi-scale fusion, and attention while
    using a compact pure-PyTorch module suitable for this repository.
    """

    def __init__(self, num_classes=5, base_channels=32, dropout=0.2):
        super().__init__()
        c = int(base_channels)
        self.stem = nn.Sequential(
            ConvBnRelu(3, c, kernel_size=3, stride=2),
            ConvBnRelu(c, c, kernel_size=3, stride=1),
        )
        self.high = nn.Sequential(ConvBnRelu(c, c), ConvBnRelu(c, c))
        self.mid = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            ConvBnRelu(c, c * 2),
            ConvBnRelu(c * 2, c * 2),
        )
        self.low = nn.Sequential(
            nn.AvgPool2d(kernel_size=4),
            ConvBnRelu(c, c * 4),
            ConvBnRelu(c * 4, c * 4),
        )
        self.fuse = nn.Sequential(
            ConvBnRelu(c + c * 2 + c * 4, c * 4, kernel_size=1),
            ChannelSpatialAttention(c * 4),
            ConvBnRelu(c * 4, c * 4),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            nn.Linear(c * 4, int(num_classes)),
        )

    def forward(self, images):
        x = self.stem(images)
        high = self.high(x)
        mid = F.interpolate(self.mid(x), size=high.shape[-2:], mode='bilinear', align_corners=False)
        low = F.interpolate(self.low(x), size=high.shape[-2:], mode='bilinear', align_corners=False)
        fused = self.fuse(torch.cat([high, mid, low], dim=1))
        return self.classifier(self.pool(fused))


class OptimizedModelEnsemble(nn.Module):
    """Compact ensemble baseline inspired by Pi et al. Scientific Reports 2023."""

    def __init__(self, backbones=None, num_classes=5, pretrained=True, dropout=0.0):
        super().__init__()
        if backbones is None:
            backbones = ['resnet18', 'densenet121', 'convnext_tiny']
        self.members = nn.ModuleList(
            [
                BackboneClassifier(
                    backbone_name=name,
                    num_classes=num_classes,
                    pretrained=pretrained,
                    dropout=dropout,
                )
                for name in backbones
            ]
        )

    def forward(self, images):
        logits = [member(images) for member in self.members]
        return torch.stack(logits, dim=0).mean(dim=0)


def tensor_strong_augment(images, noise_std=0.02, erase_prob=0.25, erase_scale=0.12):
    """Light tensor-space augmentation for Semixup consistency regularization."""
    out = images
    if torch.rand((), device=images.device).item() < 0.5:
        out = torch.flip(out, dims=[3])
    if float(noise_std) > 0:
        out = out + torch.randn_like(out) * float(noise_std)
    if float(erase_prob) > 0 and torch.rand((), device=images.device).item() < float(erase_prob):
        _, _, h, w = out.shape
        cut_h = max(1, int(h * float(erase_scale)))
        cut_w = max(1, int(w * float(erase_scale)))
        y0 = torch.randint(0, max(1, h - cut_h + 1), (), device=images.device).item()
        x0 = torch.randint(0, max(1, w - cut_w + 1), (), device=images.device).item()
        out = out.clone()
        out[:, :, y0:y0 + cut_h, x0:x0 + cut_w] = 0.0
    return out


def semixup_loss(
    model,
    images,
    labels,
    class_weights=None,
    label_smoothing=0.0,
    alpha=0.75,
    lambda_in=1.0,
    lambda_out=1.0,
    lambda_interp=1.0,
    noise_std=0.02,
):
    """Semixup-style supervised + consistency regularization loss.

    This implementation keeps the three core regularizers: in-manifold
    transformation consistency, out-of-manifold mixup consistency, and
    interpolation consistency. It uses the labeled training batch as the
    consistency pool when no separate unlabeled folder is available.
    """
    logits = model(images)
    supervised = F.cross_entropy(
        logits,
        labels,
        weight=class_weights,
        label_smoothing=float(label_smoothing),
    )

    with torch.no_grad():
        probs = torch.softmax(logits, dim=1)

    aug_images = tensor_strong_augment(images, noise_std=noise_std)
    logits_aug = model(aug_images)
    probs_aug = torch.softmax(logits_aug, dim=1)
    in_consistency = F.mse_loss(probs_aug, probs.detach())

    batch_size = images.shape[0]
    perm = torch.randperm(batch_size, device=images.device)
    lam = torch.distributions.Beta(float(alpha), float(alpha)).sample((batch_size,)).to(images.device)
    lam = torch.maximum(lam, 1.0 - lam).view(batch_size, 1, 1, 1)
    mixed_images = lam * images + (1.0 - lam) * images[perm]
    mixed_images_aug = tensor_strong_augment(mixed_images, noise_std=noise_std)

    logits_mix = model(mixed_images)
    logits_mix_aug = model(mixed_images_aug)
    probs_mix = torch.softmax(logits_mix, dim=1)
    probs_mix_aug = torch.softmax(logits_mix_aug, dim=1)
    out_consistency = F.mse_loss(probs_mix_aug, probs_mix.detach())

    with torch.no_grad():
        lam_vec = lam.view(batch_size, 1)
        interp_target = lam_vec * probs + (1.0 - lam_vec) * probs[perm]
    interpolation = F.mse_loss(probs_mix, interp_target.detach())

    total = (
        supervised
        + float(lambda_in) * in_consistency
        + float(lambda_out) * out_consistency
        + float(lambda_interp) * interpolation
    )
    return {
        'total': total,
        'supervised': supervised,
        'in_consistency': in_consistency,
        'out_consistency': out_consistency,
        'interpolation': interpolation,
    }


class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self, backbone_name='efficientnet_b3', pretrained=True):
        super().__init__()
        name = str(backbone_name).lower()
        if name == 'efficientnet_b3':
            weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if bool(pretrained) else None
            model = models.efficientnet_b3(weights=weights)
            self.feature_dim = model.classifier[1].in_features
            model.classifier = nn.Identity()
            self.encoder = model
        elif name == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if bool(pretrained) else None
            model = models.efficientnet_b0(weights=weights)
            self.feature_dim = model.classifier[1].in_features
            model.classifier = nn.Identity()
            self.encoder = model
        elif name == 'efficientnet_v2_s':
            weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if bool(pretrained) else None
            model = models.efficientnet_v2_s(weights=weights)
            self.feature_dim = model.classifier[1].in_features
            model.classifier = nn.Identity()
            self.encoder = model
        else:
            raise ValueError(f'unsupported efficientnet backbone: {backbone_name}')

    def forward(self, images):
        return self.encoder(images)


class SwinONetsKOA(nn.Module):
    """Swin-O-NETS adaptation for 5-grade KOA KL classification.

    Implements the core design of mtliba KOA-NLCS 2024: Swin-T backbone with
    hierarchical local feature enhancement, skip connection fusion, and a
    multi-prediction head network that merges coarse and fine feature streams.
    """

    def __init__(self, num_classes=5, pretrained=True, dropout=0.2, reduction=8):
        super().__init__()
        weights = models.Swin_T_Weights.IMAGENET1K_V1 if bool(pretrained) else None
        model = models.swin_t(weights=weights)
        num_stages = len(model.features)
        self.coarse_features = model.features[: num_stages // 2]
        self.fine_features = model.features[num_stages // 2 :]
        self.norm = model.norm
        self.permute = model.permute
        embed_dim = int(model.head.in_features)
        coarse_dim = max(192, embed_dim // 4)

        self.coarse_bridge = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LazyLinear(coarse_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.fine_pool = nn.AdaptiveAvgPool2d(1)
        hidden = max(256, embed_dim // 2)
        mid_dim = embed_dim + coarse_dim
        self.fusion = nn.Sequential(
            nn.Linear(mid_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, int(num_classes)),
        )
        self.skip_attn = ChannelSpatialAttention(embed_dim, reduction=int(reduction))

    def forward(self, images):
        x = self.coarse_features(images)
        x_coarse = x
        if hasattr(self, 'permute') and len(x_coarse.shape) == 4:
            pass
        coarse_vec = self.coarse_bridge(x_coarse if len(x_coarse.shape) == 4 else x_coarse.permute(0, 3, 1, 2))
        x = self.fine_features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.skip_attn(x)
        fine_vec = torch.flatten(self.fine_pool(x), 1)
        return self.fusion(torch.cat([fine_vec, coarse_vec], dim=1))


class ResEffiNet(nn.Module):
    """ResEffiNet: Hybrid EfficientNet-B3 + ResNet-18 two-stream fusion.

    Das et al. ICCIT 2025. EfficientNet-B3 extracts fine-grained features
    efficiently; ResNet-18 provides stable residual representations. The
    global feature vectors are concatenated and projected through a small
    MLP classifier for the 5-grade KL task.
    """

    def __init__(self, num_classes=5, pretrained=True, dropout=0.2, hidden_dim=1024):
        super().__init__()
        self.eff = EfficientNetFeatureExtractor('efficientnet_b3', pretrained=pretrained)
        self.res = ImageFeatureExtractor('resnet18', pretrained=pretrained)
        eff_dim = self.eff.feature_dim
        res_dim = self.res.feature_dim
        fusion_dim = eff_dim + res_dim
        hdim = int(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hdim),
            nn.BatchNorm1d(hdim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(hdim, hdim // 2),
            nn.BatchNorm1d(hdim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout) * 0.5),
            nn.Linear(hdim // 2, int(num_classes)),
        )

    def forward(self, images):
        fea_eff = self.eff(images)
        fea_res = self.res(images)
        return self.classifier(torch.cat([fea_eff, fea_res], dim=1))


class KLFuseNet(nn.Module):
    """KL-FuseNet multi-task global-local fusion baseline.

    Zhao et al. 2024 meta-analysis companion architecture: a ConvNeXt-Base
    global stream and a ResNet-50 local stream are fused with a light
    attention gating module and feed a shared ordinal+class multi-task head.
    For 5-grade KOA classification this implementation emits a single set of
    class logits compatible with the existing baseline training pipeline.
    """

    def __init__(self, num_classes=5, pretrained=True, dropout=0.3, fusion_dim=768):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if bool(pretrained) else None
        convnext = models.convnext_tiny(weights=weights)
        convnext_global_dim = convnext.classifier[2].in_features
        convnext.classifier[2] = nn.Identity()
        self.global_stream = convnext

        weights_r50 = models.ResNet50_Weights.IMAGENET1K_V2 if bool(pretrained) else None
        resnet = models.resnet50(weights=weights_r50)
        resnet_local_dim = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.local_stream = resnet

        combined = convnext_global_dim + resnet_local_dim
        self.project = nn.Sequential(
            nn.Linear(combined, int(fusion_dim)),
            nn.LayerNorm(int(fusion_dim)),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(int(fusion_dim), int(fusion_dim)),
            nn.Sigmoid(),
        )
        self.class_head = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(int(fusion_dim), int(num_classes)),
        )
        self._init_weights()

    def _init_weights(self):
        for module in (self.project, self.gate, self.class_head):
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, images):
        g = self.global_stream(images)
        l = self.local_stream(images)
        fused = self.project(torch.cat([g, l], dim=1))
        gated = fused * self.gate(fused)
        return self.class_head(gated)


def build_baseline_model(cfg):
    method = str(cfg.get('method', cfg.get('baseline_method', 'deep_siamese_cnn'))).lower()
    num_classes = int(cfg.get('num_classes', 5))
    pretrained = bool(cfg.get('pretrained_backbone', True))
    dropout = float(cfg.get('dropout', 0.0))

    if method in ('classifier', 'backbone_classifier', 'resnet50', 'convnext_t', 'convnext_tiny_classifier'):
        return BackboneClassifier(
            backbone_name=cfg.get('backbone_type', 'resnet50'),
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )

    if method in ('deep_siamese_cnn', 'siamese', 'deep_siamese'):
        return DeepSiameseCNN(
            backbone_name=cfg.get('backbone_type', 'resnet18'),
            num_classes=num_classes,
            pretrained=pretrained,
            hidden_dim=int(cfg.get('siamese_hidden_dim', 512)),
            dropout=float(cfg.get('dropout', 0.3)),
        )

    if method in ('vgg19_ordinal', 'vgg19_adjustable_ordinal', 'adjustable_ordinal'):
        return BackboneClassifier(
            backbone_name=cfg.get('backbone_type', 'vgg19_bn'),
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )

    if method in ('semixup', 'semi_mixup'):
        return BackboneClassifier(
            backbone_name=cfg.get('backbone_type', 'densenet121'),
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )

    if method in ('ordinal_regression_module', 'coral_ordinal', 'coral'):
        return CoralOrdinalClassifier(
            backbone_name=cfg.get('backbone_type', 'resnet50'),
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )

    if method in ('strict_coral', 'rank_consistent_coral', 'coral_rank_consistent'):
        return StrictCoralOrdinalClassifier(
            backbone_name=cfg.get('backbone_type', 'resnet50'),
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )

    if method in ('corn', 'corn_ordinal', 'conditional_ordinal'):
        return CornOrdinalClassifier(
            backbone_name=cfg.get('backbone_type', 'resnet50'),
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )

    if method in ('osteohrnet', 'osteo_hrnet', 'attentive_multiscale_cnn'):
        return OsteoHRNetAdapted(
            num_classes=num_classes,
            base_channels=int(cfg.get('base_channels', 32)),
            dropout=dropout,
        )

    if method in ('ordinal_vit', 'vit_ordinal', 'vit_coral'):
        return CoralOrdinalClassifier(
            backbone_name='vit_b_16',
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )

    if method in ('swin_dual_attention', 'swin_onets_adapted', 'swin_dual_attention_koa'):
        return SwinDualAttentionClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            reduction=int(cfg.get('attention_reduction', 8)),
        )

    if method in ('optimized_model_ensemble', 'ensemble_network', 'optimized_ensemble'):
        backbones = cfg.get('ensemble_backbones', ['resnet18', 'densenet121', 'convnext_tiny'])
        return OptimizedModelEnsemble(
            backbones=backbones,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )

    if method in ('swin_o_nets', 'swinonets', 'swin_o_nets_koa', 'swin_onets', 'koanlcs'):
        return SwinONetsKOA(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=float(cfg.get('dropout', 0.2)),
            reduction=int(cfg.get('attention_reduction', 8)),
        )

    if method in ('res_effi_net', 'reseffinet', 'res_effi', 'efficientnet_resnet_fusion'):
        return ResEffiNet(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=float(cfg.get('dropout', 0.2)),
            hidden_dim=int(cfg.get('reseffinet_hidden_dim', 1024)),
        )

    if method in ('kl_fusenet', 'klfusenet', 'kl_fuse_net', 'multitask_global_local_fusion'):
        return KLFuseNet(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=float(cfg.get('dropout', 0.3)),
            fusion_dim=int(cfg.get('klfusenet_fusion_dim', 768)),
        )

    raise ValueError(f'unsupported baseline method: {method}')

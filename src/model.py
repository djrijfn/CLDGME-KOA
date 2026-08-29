import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def _flatten_prompt_groups(prompts):
    if len(prompts) == 0:
        raise ValueError('prompts must not be empty')
    if all(isinstance(p, str) for p in prompts):
        return list(prompts), [[i] for i in range(len(prompts))]

    flat = []
    groups = []
    for group in prompts:
        if isinstance(group, str):
            prompt_list = [group]
        else:
            prompt_list = list(group)
        if len(prompt_list) == 0:
            raise ValueError('each prompt class must contain at least one prompt')
        indices = []
        for prompt in prompt_list:
            if not isinstance(prompt, str):
                raise TypeError('prompt entries must be strings')
            indices.append(len(flat))
            flat.append(prompt)
        groups.append(indices)
    return flat, groups


def _mean_prompt_prototypes(prompt_features, groups):
    prototypes = []
    for indices in groups:
        proto = prompt_features[indices].mean(dim=0)
        prototypes.append(proto)
    return F.normalize(torch.stack(prototypes, dim=0), dim=-1)


class HfPromptEncoder(nn.Module):
    """Frozen Sentence-Transformers encoder mapping KL-grade prompts to prototypes."""

    def __init__(self, prompts, embed_dim, model_name):
        super().__init__()
        flat_prompts, groups = _flatten_prompt_groups(prompts)
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError('transformers is required for text_encoder_type=hf') from exc
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        hidden = int(self.model.config.hidden_size)
        self.proj = nn.Linear(hidden, embed_dim)
        tokens = self.tokenizer(flat_prompts, padding=True, truncation=True, return_tensors='pt')
        for key, value in tokens.items():
            self.register_buffer(key, value, persistent=False)
        self.prompt_groups = groups

    def forward(self):
        tokens = {'input_ids': self.input_ids, 'attention_mask': self.attention_mask}
        if hasattr(self, 'token_type_ids'):
            tokens['token_type_ids'] = self.token_type_ids
        outputs = self.model(**tokens)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            feat = outputs.pooler_output
        else:
            feat = outputs.last_hidden_state[:, 0, :]
        feat = self.proj(feat)
        feat = F.normalize(feat, dim=-1)
        return _mean_prompt_prototypes(feat, self.prompt_groups)


def build_text_encoder(
    prompts,
    embed_dim,
    text_encoder_type='hf',
    hf_text_model_name='sentence-transformers/all-MiniLM-L6-v2',
):
    t = str(text_encoder_type).lower()
    if t == 'hf':
        return HfPromptEncoder(prompts, embed_dim, hf_text_model_name)
    raise ValueError(f'unsupported text_encoder_type: {text_encoder_type}')


class DirectionalStripAttention(nn.Module):
    """Directional strip attention (OCAB core) for joint-space and bone-edge cues."""

    def __init__(self, channels):
        super().__init__()
        self.h_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(7, 1),
            padding=(3, 0),
            groups=channels,
            bias=False,
        )
        self.w_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, 7),
            padding=(0, 3),
            groups=channels,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        strip_h = x.mean(dim=3, keepdim=True)
        strip_w = x.mean(dim=2, keepdim=True)
        attn_h = self.sigmoid(self.h_conv(strip_h))
        attn_w = self.sigmoid(self.w_conv(strip_w))
        return attn_h * attn_w


class OrthogonalContextAnchoringBranch(nn.Module):
    """OCAB: orthogonal context anchoring with directional strip attention."""

    def __init__(self, channels):
        super().__init__()
        self.pre = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.attn = DirectionalStripAttention(channels)

    def forward(self, x):
        feat = self.pre(x)
        structure_attn = self.attn(feat)
        enhanced = x * structure_attn
        return enhanced, structure_attn


class HeterogeneousReceptiveFieldAggregationBranch(nn.Module):
    """HRFAB: heterogeneous receptive-field aggregation with 3x3/5x5/7x7 depthwise convs."""

    def __init__(self, channels):
        super().__init__()
        self.dw3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.dw5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False)
        self.dw7 = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        feat3 = self.dw3(x)
        feat5 = self.dw5(x)
        feat7 = self.dw7(x)
        return self.fuse(torch.cat([feat3, feat5, feat7], dim=1))


class DGME(nn.Module):
    """Direction-Guided Microstructure Enhancement Module (OCAB + HRFAB)."""

    def __init__(
        self,
        channels,
        out_dim,
        dropout=0.0,
        use_ocab=True,
        use_hrfab=True,
    ):
        super().__init__()
        self.use_ocab = bool(use_ocab)
        self.use_hrfab = bool(use_hrfab)
        num_enabled = int(self.use_ocab) + int(self.use_hrfab)
        if num_enabled == 0:
            raise ValueError('DGME requires at least one enabled branch')

        self.branch_a = OrthogonalContextAnchoringBranch(channels) if self.use_ocab else None
        self.branch_c = HeterogeneousReceptiveFieldAggregationBranch(channels) if self.use_hrfab else None
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * num_enabled, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.proj = nn.Linear(channels, int(out_dim))

    def forward(self, x, return_map=False):
        branch_feats = []
        structure_attn = None
        if self.use_ocab:
            feat_a, structure_attn = self.branch_a(x)
            branch_feats.append(feat_a)
        if self.use_hrfab:
            feat_c = self.branch_c(x)
            branch_feats.append(feat_c)

        fused = self.fusion(torch.cat(branch_feats, dim=1))
        local_map = fused + x
        local_vec = torch.flatten(self.pool(local_map), 1)
        local_vec = self.dropout(local_vec)
        local_vec = self.proj(local_vec)
        if return_map:
            return local_vec, local_map, structure_attn
        return local_vec


class FeatureFusionHead(nn.Module):
    """Fuse global and local vectors into visual feature z (concat + MLP)."""

    def __init__(self, global_dim, local_dim, embed_dim):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(global_dim + local_dim, global_dim),
            nn.ReLU(inplace=True),
            nn.Linear(global_dim, embed_dim),
        )

    @property
    def uses_text_features(self):
        return False

    def forward(self, global_vec, local_vec):
        x = torch.cat([global_vec, local_vec], dim=-1)
        return self.fusion(x)


class OCLRHead(nn.Module):
    """OCLR head (Ordinal Cumulative Link Regression, manuscript Sec. 3.4).

    Shared latent severity score with anchor + Softplus ordered thresholds.
    """

    def __init__(self, in_dim, num_classes=5):
        super().__init__()
        self.num_thresholds = int(num_classes) - 1
        if self.num_thresholds < 1:
            raise ValueError('num_classes must be >= 2 for ordinal regression')
        self.score = nn.Linear(in_dim, 1)
        self.threshold_anchor = nn.Parameter(torch.tensor(-1.5))
        self.threshold_step_logits = nn.Parameter(torch.full((max(self.num_thresholds - 1, 0),), 0.5413))

    def ordered_thresholds(self):
        if self.num_thresholds == 1:
            return self.threshold_anchor.view(1)
        steps = F.softplus(self.threshold_step_logits)
        offsets = torch.cat(
            [
                self.threshold_anchor.new_zeros(1),
                torch.cumsum(steps, dim=0),
            ],
            dim=0,
        )
        return self.threshold_anchor + offsets

    def forward(self, x):
        eta = self.score(x)
        thresholds = self.ordered_thresholds().view(1, -1)
        return eta - thresholds


class CLDGMEKOA(nn.Module):
    """CLDGME-KOA: ConvNeXt-Tiny backbone + DGME at Stage2 + PS (frozen text)
    + OCLR head, trained with HFC and evaluated with dual-view inference."""

    def __init__(
        self,
        prompts,
        embed_dim=256,
        pretrained_backbone=True,
        text_encoder_type='hf',
        hf_text_model_name='sentence-transformers/all-MiniLM-L6-v2',
        freeze_hf_text_encoder=False,
        num_classes=5,
        backbone_type='convnext_tiny',
        use_ordinal_branch=True,
        use_text_branch=True,
        use_dgme=True,
        dgme_stage='stage2',
        dgme_dropout=0.0,
        dgme_use_ocab=True,
        dgme_use_hrfab=True,
        ordinal_head_type='oclr',
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.use_dgme = bool(use_dgme)
        self.use_ordinal_branch = bool(use_ordinal_branch)
        self.use_text_branch = bool(use_text_branch)
        self.dgme_stage = str(dgme_stage).lower()
        self.backbone_type = str(backbone_type).lower()
        self.dgme_use_ocab = bool(dgme_use_ocab)
        self.dgme_use_hrfab = bool(dgme_use_hrfab)
        self.ordinal_head_type = str(ordinal_head_type).lower()
        if self.ordinal_head_type not in ('oclr', 'cumulative_link', 'cumulative-link', 'clink', 'true_coral', 'rank_consistent'):
            raise ValueError(f'unsupported ordinal_head_type: {ordinal_head_type}')

        self.image_encoder, in_dim = self._build_image_encoder(
            backbone_type=self.backbone_type,
            pretrained_backbone=pretrained_backbone,
        )
        self.image_feature_dim = in_dim
        self.image_proj = nn.Linear(in_dim, embed_dim)

        self.text_encoder = build_text_encoder(prompts, embed_dim, text_encoder_type, hf_text_model_name)
        if str(text_encoder_type).lower() == 'hf' and bool(freeze_hf_text_encoder):
            for p in self.text_encoder.model.parameters():
                p.requires_grad = False
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

        self.local_enhancement_module = None
        self.fusion_head = None
        if self.use_dgme:
            dgme_channels = self._dgme_feature_dim(self.backbone_type, self.dgme_stage)
            self.local_enhancement_module = DGME(
                channels=dgme_channels,
                out_dim=in_dim,
                dropout=float(dgme_dropout),
                use_ocab=self.dgme_use_ocab,
                use_hrfab=self.dgme_use_hrfab,
            )
            self.fusion_head = FeatureFusionHead(
                global_dim=in_dim,
                local_dim=in_dim,
                embed_dim=embed_dim,
            )

        if self.use_ordinal_branch:
            self.ordinal_head = OCLRHead(in_dim=embed_dim, num_classes=self.num_classes)
        else:
            self.ordinal_head = None

    @staticmethod
    def _build_image_encoder(backbone_type, pretrained_backbone=True):
        backbone = str(backbone_type).lower()
        if backbone == 'convnext_tiny':
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained_backbone else None
            model = models.convnext_tiny(weights=weights)
            in_dim = model.classifier[2].in_features
            model.classifier = nn.Identity()
            return model, in_dim
        raise ValueError(f'unsupported backbone_type: {backbone_type}')

    @staticmethod
    def _dgme_feature_dim(backbone_type, dgme_stage):
        backbone = str(backbone_type).lower()
        stage = str(dgme_stage).lower()
        if backbone == 'convnext_tiny':
            dims = {
                'stem': 96,
                'stage1': 96,
                'stage2': 192,
                'stage3': 384,
                'final': 768,
            }
            if stage not in dims:
                raise ValueError(f'unsupported dgme_stage for convnext_tiny: {dgme_stage}')
            return dims[stage]
        raise ValueError(f'unsupported backbone_type for DGME: {backbone_type}')

    def encode_image_with_spatial(self, images):
        """ConvNeXt feature extraction with intermediate spatial map."""
        stem_spatial = self.image_encoder.features[0](images)
        stage1_spatial = self.image_encoder.features[1](stem_spatial)
        x = self.image_encoder.features[2](stage1_spatial)
        stage2_spatial = self.image_encoder.features[3](x)
        x = self.image_encoder.features[4](stage2_spatial)
        local_spatial = self.image_encoder.features[5](x)
        spatial = self.image_encoder.features[6:](local_spatial)
        pooled = self.image_encoder.avgpool(spatial)
        global_vec = torch.flatten(pooled, 1)
        shallow_options = {
            'stem': stem_spatial,
            'stage1': stage1_spatial,
            'stage2': stage2_spatial,
            'stage3': local_spatial,
            'final': spatial,
        }
        shallow_spatial = shallow_options.get(self.dgme_stage, stage1_spatial)
        return global_vec, spatial, local_spatial, shallow_spatial

    def encode_visual_feature(self, images):
        global_vec, spatial_map, local_spatial_map, shallow_spatial_map = self.encode_image_with_spatial(images)
        local_feature_map = None
        structure_attention = None
        if self.use_dgme:
            local_vec, local_feature_map, structure_attention = self.local_enhancement_module(
                shallow_spatial_map,
                return_map=True,
            )
            fused = self.fusion_head(global_vec, local_vec)
        else:
            local_vec = None
            fused = self.image_proj(global_vec)
        fused_norm = F.normalize(fused, dim=-1)
        return {
            'global_feature': global_vec,
            'local_feature': local_vec,
            'fused_feature': fused,
            'fused_feature_norm': fused_norm,
            'spatial_feature': spatial_map,
            'local_spatial_feature': local_spatial_map,
            'shallow_spatial_feature': shallow_spatial_map,
            'local_feature_map': local_feature_map,
            'structure_attention': structure_attention,
        }

    def forward(self, images, return_dict=False):
        txt = self.text_encoder() if self.use_text_branch else None
        visual = self.encode_visual_feature(images)
        out = dict(visual)

        if self.use_text_branch:
            scale = self.logit_scale.exp()
            out['text_logits'] = scale * out['fused_feature_norm'] @ txt.t()
        else:
            out['text_logits'] = None

        if self.use_ordinal_branch:
            ordinal_logits = self.ordinal_head(out['fused_feature'])
            ordinal_probs = torch.sigmoid(ordinal_logits)
            ordinal_pred = (ordinal_probs > 0.5).sum(dim=1).to(dtype=torch.long)
            out['ordinal_logits'] = ordinal_logits
            out['ordinal_probs'] = ordinal_probs
            out['ordinal_pred'] = ordinal_pred
            out['ordinal_thresholds'] = self.ordinal_head.ordered_thresholds()
            out['ordinal_score'] = self.ordinal_head.score(out['fused_feature'])
        else:
            out['ordinal_logits'] = None
            out['ordinal_probs'] = None
            out['ordinal_pred'] = None
            out['ordinal_thresholds'] = None
            out['ordinal_score'] = None

        if return_dict:
            return out
        if out['text_logits'] is not None:
            return out['text_logits']
        if out['ordinal_logits'] is not None:
            return out['ordinal_logits']
        return out['fused_feature_norm']


# Backward-compatible aliases (pre-manuscript-renaming names)
ClipKoaMinimal = CLDGMEKOA
CumulativeLinkOrdinalHead = OCLRHead

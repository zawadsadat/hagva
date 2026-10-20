"""Inference-only adapter for the unmodified official MixVPR inference modules.

Training-only Lightning/metric-learning packages are unnecessary for inference.
Backbone and aggregator source are imported directly from the pinned upstream
checkout; every checkpoint tensor must load strictly. No shared descriptor cache.
"""
import importlib.util
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision import transforms


def source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MixVPRModel(torch.nn.Module):
    def __init__(self, upstream):
        super().__init__()
        resnet = source_module('_mixvpr_resnet', upstream/'models/backbones/resnet.py')
        mixer = source_module('_mixvpr_aggregator', upstream/'models/aggregators/mixvpr.py')
        # The full checkpoint supplies ALL weights, so no ImageNet bootstrap is needed.
        self.backbone = resnet.ResNet(model_name='resnet50', pretrained=False, layers_to_crop=[4])
        self.aggregator = mixer.MixVPR(in_channels=1024, in_h=20, in_w=20,
                                     out_channels=1024, mix_depth=4, mlp_ratio=1, out_rows=4)

    def forward(self, x):
        return self.aggregator(self.backbone(x))


class MixVPRFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.upstream = Path(os.environ['HAGVA_MIXVPR_UPSTREAM']).resolve()
        self.checkpoint = Path(os.environ['HAGVA_MIXVPR_CHECKPOINT']).resolve()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dim = 4096
        self.model = MixVPRModel(self.upstream)
        state = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((320, 320), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def compute_features(self, imgs):
        result = np.empty((len(imgs), self.dim), dtype=np.float32)
        with torch.no_grad():
            for i, image in enumerate(imgs):
                image = Image.fromarray(image).convert('RGB')
                x = self.transform(image).unsqueeze(0).to(self.device)
                result[i] = self.model(x).cpu().numpy()[0]
        return result

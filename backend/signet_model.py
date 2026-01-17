"""SigNet signature verification model (PyTorch inference).

This project ships SigNet weights in `models/signet.pkl` (a pickled dict with
`params` following the original Lasagne/SigNet ordering). The previous
TensorFlow implementation built the architecture but did not load weights into
the graph, which makes the output effectively random.

This module provides a deterministic, dependency-light PyTorch implementation
that correctly loads the shipped weights and produces 2048-D embeddings.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_tensor(array: np.ndarray, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array)).to(dtype)


class _FixedBatchNorm2d(nn.Module):
    def __init__(self, beta: np.ndarray, gamma: np.ndarray, mean: np.ndarray, inv_std: np.ndarray):
        super().__init__()
        self.register_buffer("beta", _as_tensor(beta).view(1, -1, 1, 1))
        self.register_buffer("scale", (_as_tensor(gamma) * _as_tensor(inv_std)).view(1, -1, 1, 1))
        self.register_buffer("mean", _as_tensor(mean).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) * self.scale + self.beta


class _FixedBatchNorm1d(nn.Module):
    def __init__(self, beta: np.ndarray, gamma: np.ndarray, mean: np.ndarray, inv_std: np.ndarray):
        super().__init__()
        self.register_buffer("beta", _as_tensor(beta).view(1, -1))
        self.register_buffer("scale", (_as_tensor(gamma) * _as_tensor(inv_std)).view(1, -1))
        self.register_buffer("mean", _as_tensor(mean).view(1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) * self.scale + self.beta


@dataclass(frozen=True)
class _SigNetParams:
    input_size: Tuple[int, int]
    params: List[np.ndarray]


def _load_signet_pkl(model_path: str | Path) -> _SigNetParams:
    model_path = Path(model_path)
    with model_path.open("rb") as f:
        obj = pickle.load(f, encoding="latin1")
    input_size = tuple(obj["input_size"])
    params = obj["params"]
    if not isinstance(params, list) or len(params) != 35:
        raise ValueError(f"Unexpected SigNet params length: {len(params) if isinstance(params, list) else type(params)}")
    return _SigNetParams(input_size=input_size, params=params)


class _SigNetTorch(nn.Module):
    """SigNet forward pass with fixed BN statistics/affine params."""

    def __init__(self, params: List[np.ndarray]):
        super().__init__()

        # conv1
        self.conv1 = nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=0, bias=False)
        self.bn1 = _FixedBatchNorm2d(params[1], params[2], params[3], params[4])

        # conv2
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, bias=False)  # SAME
        self.bn2 = _FixedBatchNorm2d(params[6], params[7], params[8], params[9])

        # conv3
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = _FixedBatchNorm2d(params[11], params[12], params[13], params[14])

        # conv4
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = _FixedBatchNorm2d(params[16], params[17], params[18], params[19])

        # conv5
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = _FixedBatchNorm2d(params[21], params[22], params[23], params[24])

        # fc1/fc2
        # Note: weights in pkl are shaped as (in_features, out_features) (Lasagne/TF style).
        self.fc1 = nn.Linear(3840, 2048, bias=False)
        self.bn_fc1 = _FixedBatchNorm1d(params[26], params[27], params[28], params[29])

        self.fc2 = nn.Linear(2048, 2048, bias=False)
        self.bn_fc2 = _FixedBatchNorm1d(params[31], params[32], params[33], params[34])

        self._load_weights(params)
        self.eval()

    def _load_weights(self, params: List[np.ndarray]) -> None:
        with torch.no_grad():
            self.conv1.weight.copy_(_as_tensor(params[0]))
            self.conv2.weight.copy_(_as_tensor(params[5]))
            self.conv3.weight.copy_(_as_tensor(params[10]))
            self.conv4.weight.copy_(_as_tensor(params[15]))
            self.conv5.weight.copy_(_as_tensor(params[20]))

            # transpose (in,out) -> (out,in)
            self.fc1.weight.copy_(_as_tensor(params[25]).t())
            self.fc2.weight.copy_(_as_tensor(params[30]).t())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,1,150,220)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x = torch.flatten(x, 1)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        return x


class SigNetModel:
    """SigNet embedding extractor compatible with the project's existing API."""

    def __init__(self, model_path: str = "models/signet.pkl"):
        print("📦 正在加载SigNet模型 (PyTorch)...")
        weights = _load_signet_pkl(model_path)
        self.input_size = weights.input_size  # (150, 220)
        self._net = _SigNetTorch(weights.params)
        print(f"✅ SigNet模型加载成功! 输入尺寸: {self.input_size}")

    def get_feature_vector(self, image: np.ndarray) -> np.ndarray:
        """提取签名特征向量 (2048维)."""
        if image.ndim != 2:
            raise ValueError(f"Expected grayscale image HxW, got shape={image.shape}")

        # Match typical SigNet preprocessing: uint8 [0,255] -> float32 [0,1]
        x = image.astype(np.float32) / 255.0
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        with torch.no_grad():
            feat = self._net(x).squeeze(0).cpu().numpy()
        return feat

    def compute_similarity(self, template_img: np.ndarray, query_img: np.ndarray) -> float:
        """计算两个签名的欧氏距离（越小越相似）"""
        feat1 = self.get_feature_vector(template_img)
        feat2 = self.get_feature_vector(query_img)
        return float(np.linalg.norm(feat1 - feat2))

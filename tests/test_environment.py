"""Phase 0 smoke tests: environment is sane, package imports, torch sees MPS."""

from __future__ import annotations

import sys


def test_python_version() -> None:
    """We target Python 3.13."""
    assert sys.version_info >= (3, 13), f"Need Python 3.13+, got {sys.version_info}"


def test_package_imports() -> None:
    """mlfactory package can be imported."""
    import mlfactory

    assert mlfactory.__version__ == "0.0.1"


def test_torch_available() -> None:
    """PyTorch is installed."""
    import torch

    assert torch.__version__ >= "2.0"


def test_mps_available() -> None:
    """On M-series Mac, MPS backend must be available."""
    import torch

    assert torch.backends.mps.is_built(), "MPS not built into this torch wheel"
    assert torch.backends.mps.is_available(), "MPS not available on this machine"


def test_mps_tensor_roundtrip() -> None:
    """Prove we can actually move a tensor to MPS and do math on it."""
    import torch

    device = torch.device("mps")
    x = torch.randn(32, 32, device=device)
    y = x @ x.T
    assert y.shape == (32, 32)
    assert y.device.type == "mps"

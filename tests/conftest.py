import pytest
import torch


@pytest.fixture
def cpu():
    return torch.device("cpu")


@pytest.fixture
def gpu():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    pytest.skip("no CUDA/MPS device")

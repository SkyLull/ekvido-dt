import pytest
import torch

from ekvido.chart import make_chart


def test_chart_layout():
    c = make_chart(900, 600)
    assert c.shape == (3, 600, 900) and c.dtype == torch.float32
    ladder = c[0, int(0.15 * 600)]
    peaks = sorted({round(v, 3) for v in ladder.tolist() if v > 0.5})
    assert peaks == [2.0**k for k in range(9)]
    row3 = c[:, int(0.65 * 600)]
    assert row3.max().item() == 256.0  # clipped discs = image max, like a real clip
    assert c.max().item() == 256.0
    assert (row3[0] > row3[2]).any() and (
        row3[2] > row3[0]
    ).any()  # orange and blue fringes
    assert c.min().item() == pytest.approx(0.02)

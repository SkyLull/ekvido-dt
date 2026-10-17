import torch

BACKGROUND = 0.02


def _disc(img, cy, cx, r_out, rgb, r_in=0.0):
    _, h, w = img.shape
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    mask = (d2 <= r_out * r_out) & (d2 >= r_in * r_in)
    img[:, mask] = torch.tensor(rgb, dtype=img.dtype).view(3, 1)


def make_chart(width: int = 1800, height: int = 1200) -> torch.Tensor:
    img = torch.full((3, height, width), BACKGROUND)
    y1, y2, y3, y4 = (int(f * height) for f in (0.15, 0.40, 0.65, 0.88))
    for k in range(9):  # +1 EV ladder
        v = 2.0**k
        _disc(img, y1, int(width * (k + 1) / 10), 2, (v, v, v))
    for i, rgb in enumerate([(8.0, 4.0, 1.2), (1.5, 4.0, 10.0), (5.0, 5.0, 5.0)]):
        _disc(img, y2, int(width * (i + 1) / 4), 4, rgb)
    for i, fringe in enumerate([(3.0, 1.5, 0.5), (0.6, 1.5, 4.0), None]):
        cx = int(width * (i + 1) / 4)
        if fringe:
            _disc(img, y3, cx, 18, fringe, r_in=12)
        _disc(
            img, y3, cx, 12, (256.0, 256.0, 256.0)
        )  # clipped = brightest, as in a real file
    img[:, y4, int(0.1 * width) : int(0.4 * width)] = 6.0
    img[:, y4 - int(0.08 * height) : y4, int(0.6 * width)] = 6.0
    return img

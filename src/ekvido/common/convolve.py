import torch


def fft_size(n: int) -> int:
    m = max(1, n)
    while True:
        k = m
        for p in (2, 3, 5):
            while k % p == 0:
                k //= p
        if k == 1:
            return m
        m += 1


class Convolver:
    """Linear convolution of one fixed (C,H,W) source with many centred kernels.

    The FFT grid is sized so any kernel with half-size ≤ `pad` never wraps around.
    """

    def __init__(self, src: torch.Tensor, pad: int, keep_spectra: bool = True):
        self.src = src
        self.pad = pad
        _, self.h, self.w = src.shape
        self.fh = fft_size(max(self.h + pad, 2 * pad + 1))
        self.fw = fft_size(max(self.w + pad, 2 * pad + 1))
        self.keep_spectra = keep_spectra
        self._spectra: torch.Tensor | None = None

    def _spectrum(self, c: int) -> torch.Tensor:
        return torch.fft.rfft2(self.src[c], s=(self.fh, self.fw))

    def _source(self, c: int) -> torch.Tensor:
        if not self.keep_spectra:
            return self._spectrum(c)
        if self._spectra is None:
            self._spectra = torch.stack(
                [self._spectrum(i) for i in range(self.src.shape[0])]
            )
        return self._spectra[c]

    def apply(self, kernel: torch.Tensor) -> torch.Tensor:
        r = (kernel.shape[-1] - 1) // 2
        if r > self.pad:
            raise ValueError(f"kernel half-size {r} exceeds convolver pad {self.pad}")
        kp = torch.zeros(self.fh, self.fw, dtype=self.src.dtype, device=self.src.device)
        kp[: 2 * r + 1, : 2 * r + 1] = kernel.to(self.src.device, self.src.dtype)
        kf = torch.fft.rfft2(torch.roll(kp, shifts=(-r, -r), dims=(0, 1)))
        # Copy each crop out immediately so only one padded grid is alive at a time.
        out = torch.empty_like(self.src)
        for c in range(self.src.shape[0]):
            spec = self._source(c) * kf
            out[c] = torch.fft.irfft2(spec, s=(self.fh, self.fw))[: self.h, : self.w]
        return out


def fft_convolve(src: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    return Convolver(src, pad=(kernel.shape[-1] - 1) // 2).apply(kernel)

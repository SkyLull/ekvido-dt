import os

# Let MPS fall back to CPU for any op it lacks, instead of raising.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "0.1.0"

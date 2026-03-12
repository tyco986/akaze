"""AKAZE - GPU-accelerated (CUDA) and CPU (OpenCV) A-KAZE feature detection and matching"""

try:
    from .libakaze_pybindings import AKAZE, AKAZEOptions, Matcher
except ImportError:
    AKAZE = AKAZEOptions = Matcher = None

from .akaze_aligner import AkazeAligner

__all__ = ["AKAZE", "AKAZEOptions", "Matcher", "AkazeAligner"]

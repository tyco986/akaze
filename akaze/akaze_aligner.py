"""
AKAZE aligner: A-KAZE features + BFMatcher + findHomography(RANSAC).
Supports GPU (CUDA) and CPU (OpenCV) backends via ``device`` parameter.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, Union

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from .libakaze_pybindings import AKAZE, AKAZEOptions, Matcher
    HAS_CUDA_BACKEND = True
except ImportError:
    HAS_CUDA_BACKEND = False

# NNDR threshold for match filtering (same as C++ akaze_match)
DRATIO = 0.80


def _to_gray_float32(x) -> np.ndarray:
    """(B,H,W) or (B,C,H,W) -> (B,H,W) float32 [0,1]. Accepts torch or numpy."""
    if HAS_TORCH and isinstance(x, torch.Tensor):
        arr = x.cpu().numpy()
    else:
        arr = np.asarray(x)
    if arr.ndim == 4:
        if arr.shape[1] == 3:
            arr = (arr[:, 0] * 0.299 + arr[:, 1] * 0.587 + arr[:, 2] * 0.114)
        else:
            arr = arr[:, 0]
    elif arr.ndim == 3 and arr.shape[0] in (1, 3):
        if arr.shape[0] == 3:
            arr = np.dot(arr.transpose(1, 2, 0), [0.299, 0.587, 0.114])
        else:
            arr = arr[0]
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    if arr.size == 0:
        return arr.astype(np.float32)
    if arr.dtype in (np.float32, np.float64):
        if arr.max() > 1.0:
            arr = (arr / 255.0).clip(0, 1)
    else:
        arr = (arr.astype(np.float32) / 255.0).clip(0, 1)
    return arr.astype(np.float32)


class AkazeAligner:
    """
    AKAZE feature-based homography alignment.

    * ``device=None`` or ``device=0,1,...`` — GPU backend (CUDA, requires libakaze_pybindings)
    * ``device="cpu"`` — CPU backend (OpenCV ``cv2.AKAZE``)
    """

    def __init__(
        self,
        omax: int = 4,
        nsublevels: int = 4,
        dthreshold: float = 0.001,
        ransac_thresh: float = 2.5,
        ransac_max_iters: int = 2000,
        ransac_confidence: float = 0.995,
        nndr: float = DRATIO,
        device: Union[str, int] = 0,
        ransac_seed: Optional[int] = 42,
    ):
        self.omax = omax
        self.nsublevels = nsublevels
        self.dthreshold = dthreshold
        self.ransac_thresh = ransac_thresh
        self.ransac_max_iters = ransac_max_iters
        self.ransac_confidence = ransac_confidence
        self.nndr = nndr
        self.ransac_seed = ransac_seed
        self._use_cpu = isinstance(device, str) and device.lower() == "cpu"

        if self._use_cpu:
            self.device = "cpu"
            self._cv_akaze = cv2.AKAZE_create(
                descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
                descriptor_size=0,
                descriptor_channels=3,
                threshold=dthreshold,
                nOctaves=omax,
                nOctaveLayers=nsublevels,
                diffusivity=cv2.KAZE_DIFF_PM_G2,
            )
            self._cv_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            if not HAS_CUDA_BACKEND:
                raise RuntimeError(
                    "CUDA backend (libakaze_pybindings) not available. "
                    "Use device='cpu' for the OpenCV backend."
                )
            self.device = device
            self._matcher = Matcher()
            self._akaze_cache = {}  # (w, h) -> AKAZE instance

    # ------------------------------------------------------------------
    # GPU (CUDA) backend
    # ------------------------------------------------------------------
    def _get_or_create_akaze(self, w: int, h: int):
        key = (w, h)
        if key not in self._akaze_cache:
            opts = AKAZEOptions()
            opts.setWidth(w)
            opts.setHeight(h)
            opts.omax = self.omax
            opts.nsublevels = self.nsublevels
            opts.dthreshold = self.dthreshold
            self._akaze_cache[key] = AKAZE(opts)
        return self._akaze_cache[key]

    def _compute_features_gpu(self, img: np.ndarray):
        h, w = img.shape
        akaze = self._get_or_create_akaze(w, h)
        akaze.Create_Nonlinear_Scale_Space(img)
        return akaze.Compute_Descriptors()

    def _match_and_homography_gpu(
        self, desc_t, kpts_t, desc_i, kpts_i
    ) -> Tuple[np.ndarray, np.ndarray]:
        _I = np.eye(3, dtype=np.float32)
        _Z = np.zeros(2, dtype=np.float32)

        if desc_t is None or desc_i is None or kpts_t is None or kpts_i is None:
            return _I, _Z
        if kpts_t.shape[0] < 4 or kpts_i.shape[0] < 4:
            return _I, _Z

        pts_t = kpts_t[:, :2].astype(np.float32)
        pts_i = kpts_i[:, :2].astype(np.float32)

        dmatches = self._matcher.BFMatch(desc_t, desc_i)
        if dmatches is None or dmatches.size == 0:
            return _I, _Z

        dmatches = np.asarray(dmatches)
        if dmatches.ndim == 1:
            dmatches = dmatches.reshape(-1, 8)
        dist0 = dmatches[:, 3]
        dist1 = dmatches[:, 7]
        mask = (dist1 > 1e-10) & (dist0 < self.nndr * dist1)
        valid = np.where(mask)[0]
        if len(valid) < 4:
            return _I, _Z

        qidx = dmatches[valid, 0].astype(np.int32)
        tidx = dmatches[valid, 1].astype(np.int32)
        pts0 = pts_t[qidx]
        pts1 = pts_i[tidx]

        H, status = self._find_homography_ransac(pts0, pts1)
        if H is None:
            return _I, _Z
        H32 = H.astype(np.float32)
        return H32, H32[:2, 2].copy()

    # ------------------------------------------------------------------
    # CPU (OpenCV) backend
    # ------------------------------------------------------------------
    def _compute_features_cpu(self, img: np.ndarray):
        img_u8 = (img * 255).clip(0, 255).astype(np.uint8)
        kpts, desc = self._cv_akaze.detectAndCompute(img_u8, None)
        if kpts is None or desc is None or len(kpts) == 0:
            return None, None
        pts = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts], dtype=np.float32)
        return desc, pts

    def _match_and_homography_cpu(
        self, desc_t, pts_t, desc_i, pts_i
    ) -> Tuple[np.ndarray, np.ndarray]:
        _I = np.eye(3, dtype=np.float32)
        _Z = np.zeros(2, dtype=np.float32)

        if desc_t is None or desc_i is None or pts_t is None or pts_i is None:
            return _I, _Z
        if len(pts_t) < 4 or len(pts_i) < 4:
            return _I, _Z

        matches = self._cv_matcher.knnMatch(desc_t, desc_i, k=2)
        good = [
            m for m, n in matches
            if n.distance > 1e-10 and m.distance < self.nndr * n.distance
        ]
        if len(good) < 4:
            return _I, _Z

        pts0 = pts_t[[m.queryIdx for m in good]]
        pts1 = pts_i[[m.trainIdx for m in good]]

        H, status = self._find_homography_ransac(pts0, pts1)
        if H is None:
            return _I, _Z
        H32 = H.astype(np.float32)
        return H32, H32[:2, 2].copy()

    # ------------------------------------------------------------------
    # Shared
    # ------------------------------------------------------------------
    def _find_homography_ransac(
        self,
        pts0: np.ndarray,
        pts1: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.ransac_seed is not None:
            cv2.setRNGSeed(self.ransac_seed)
        H, status = cv2.findHomography(
            pts0, pts1,
            cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thresh,
            maxIters=self.ransac_max_iters,
            confidence=self.ransac_confidence,
        )
        return H, status

    def find_transform(self, template, input_image) -> Dict:
        """
        template, input_image: (B,H,W) or (B,C,H,W), torch or numpy.
        Returns: {"warp_matrix": (B,3,3), "motion": (B,2)}
        """
        t = _to_gray_float32(template)
        i = _to_gray_float32(input_image)
        if t.ndim == 2:
            t = t[np.newaxis]
            i = i[np.newaxis]

        B = t.shape[0]
        warps = []
        motions = []

        if self._use_cpu:
            for b in range(B):
                desc_t, pts_t = self._compute_features_cpu(t[b])
                desc_i, pts_i = self._compute_features_cpu(i[b])
                H, mot = self._match_and_homography_cpu(desc_t, pts_t, desc_i, pts_i)
                warps.append(H)
                motions.append(mot)
        else:
            for b in range(B):
                desc_t, kpts_t = self._compute_features_gpu(t[b])
                desc_i, kpts_i = self._compute_features_gpu(i[b])
                H, mot = self._match_and_homography_gpu(desc_t, kpts_t, desc_i, kpts_i)
                warps.append(H)
                motions.append(mot)

        warp = np.stack(warps, axis=0)
        motion = np.stack(motions, axis=0)

        if HAS_TORCH:
            return {
                "warp_matrix": torch.from_numpy(warp).float(),
                "motion": torch.from_numpy(motion).float(),
            }
        return {"warp_matrix": warp, "motion": motion}


__all__ = ["AkazeAligner"]

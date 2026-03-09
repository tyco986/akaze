"""
AKAZE CUDA aligner: A-KAZE features + BFMatcher + findHomography(RANSAC).
Feature-based homography alignment using GPU-accelerated A-KAZE.
"""

import numpy as np
from typing import Dict, Optional, Tuple

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .libakaze_pybindings import AKAZE, AKAZEOptions, Matcher

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
    if arr.dtype in (np.float32, np.float64):
        if arr.max() > 1.0:
            arr = (arr / 255.0).clip(0, 1)
    else:
        arr = (arr.astype(np.float32) / 255.0).clip(0, 1)
    return arr.astype(np.float32)


class AkazeAligner:
    """
    AKAZE CUDA: GPU-accelerated A-KAZE features + BFMatcher + findHomography(RANSAC).
    Feature-based homography alignment. Uses libakaze_pybindings (AKAZE, Matcher).
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
        device: Optional[int] = None,
        ransac_seed: Optional[int] = None,
    ):
        self.omax = omax
        self.nsublevels = nsublevels
        self.dthreshold = dthreshold
        self.ransac_thresh = ransac_thresh
        self.ransac_max_iters = ransac_max_iters
        self.ransac_confidence = ransac_confidence
        self.nndr = nndr
        self.device = device
        self.ransac_seed = ransac_seed
        self._matcher = Matcher()

    def _find_transform_one(
        self, template: np.ndarray, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single pair. Returns (H 3x3, motion 2,)."""
        h, w = template.shape
        options_t = AKAZEOptions()
        options_t.setWidth(w)
        options_t.setHeight(h)
        evolution_t = AKAZE(options_t)

        h2, w2 = image.shape
        options_i = AKAZEOptions()
        options_i.setWidth(w2)
        options_i.setHeight(h2)
        evolution_i = AKAZE(options_i)

        evolution_t.Create_Nonlinear_Scale_Space(template)
        desc_t, kpts_t = evolution_t.Compute_Descriptors()
        evolution_i.Create_Nonlinear_Scale_Space(image)
        desc_i, kpts_i = evolution_i.Compute_Descriptors()

        if desc_t is None or desc_i is None or kpts_t is None or kpts_i is None:
            return np.eye(3, dtype=np.float32), np.zeros(2, dtype=np.float32)
        n_t, n_i = kpts_t.shape[0], kpts_i.shape[0]
        if n_t < 4 or n_i < 4:
            return np.eye(3, dtype=np.float32), np.zeros(2, dtype=np.float32)

        # kpts: (N, 7) - columns [x, y, size, angle, response, octave, class_id]
        pts_t = kpts_t[:, :2].astype(np.float32)
        pts_i = kpts_i[:, :2].astype(np.float32)

        # BFMatch template -> image
        dmatches = self._matcher.BFMatch(desc_t, desc_i)
        if dmatches is None or dmatches.size == 0:
            return np.eye(3, dtype=np.float32), np.zeros(2, dtype=np.float32)

        dmatches = np.asarray(dmatches)
        if dmatches.ndim == 1:
            dmatches = dmatches.reshape(-1, 8)
        # Each row: [queryIdx, trainIdx, 0, dist0, queryIdx1, trainIdx1, 0, dist1]
        dist0 = dmatches[:, 3]
        dist1 = dmatches[:, 7]
        mask = (dist1 > 1e-10) & (dist0 < self.nndr * dist1)
        valid = np.where(mask)[0]
        if len(valid) < 4:
            return np.eye(3, dtype=np.float32), np.zeros(2, dtype=np.float32)

        qidx = dmatches[valid, 0].astype(np.int32)
        tidx = dmatches[valid, 1].astype(np.int32)
        pts0 = pts_t[qidx]
        pts1 = pts_i[tidx]

        H, status = self._find_homography_ransac(pts0, pts1)
        if H is None:
            return np.eye(3, dtype=np.float32), np.zeros(2, dtype=np.float32)
        return H.astype(np.float32), H[:2, 2]

    def _find_homography_ransac(
        self,
        pts0: np.ndarray,
        pts1: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """RANSAC homography. pts0=template, pts1=image."""
        import cv2
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
        for b in range(B):
            H, mot = self._find_transform_one(t[b], i[b])
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

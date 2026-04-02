"""
Embryo analysis pipeline.

VolumeBuilder   — interpolates 7 sparse slices to a dense 3-D volume
Segmentor       — Otsu threshold + hole-fill to produce a binary mask
EllipsoidFitter — PCA/SVD ellipsoid fit to the mask voxels
HatchingDetector— counts voxels outside the fitted ellipsoid
EmbroyoPipeline — convenience wrapper running all four steps
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.interpolate import RegularGridInterpolator


# ══════════════════════════════════════════════════════════════════════════════
class VolumeBuilder:
    """
    Interpolates a sparse (7-slice) stack into a dense 3-D volume.

    Parameters
    ----------
    target_slices : number of z-planes in the output volume
    voxel_um      : physical voxel edge length in micrometres
    """

    def __init__(self, target_slices: int = 56, voxel_um: float = 4.0):
        self.target_slices = target_slices
        self.voxel_um = voxel_um

    def build(self, slices: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        slices : (n_slices, H, W) float array

        Returns
        -------
        (target_slices, H, W) float32 array
        """
        s, h, w = slices.shape
        z_orig = np.linspace(0, 1, s)
        z_new  = np.linspace(0, 1, self.target_slices)
        y_ax   = np.linspace(0, 1, h)
        x_ax   = np.linspace(0, 1, w)

        fn = RegularGridInterpolator(
            (z_orig, y_ax, x_ax), slices,
            method="linear", bounds_error=False, fill_value=0,
        )

        zz, yy, xx = np.meshgrid(z_new, y_ax, x_ax, indexing="ij")
        pts = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)
        return fn(pts).reshape(self.target_slices, h, w).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
class Segmentor:
    """
    Segments the embryo/zona from background using an Otsu threshold
    followed by per-slice hole-filling.
    """

    def segment(self, volume: np.ndarray) -> np.ndarray:
        """
        Returns a boolean mask of the same shape as *volume*.
        """
        flat = volume.ravel()
        flat = flat[flat > 0]
        thresh = self._otsu(flat)
        mask = volume >= thresh
        filled = np.stack([binary_fill_holes(mask[i]) for i in range(mask.shape[0])])
        return filled.astype(bool)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _otsu(values: np.ndarray) -> float:
        """Pure-NumPy Otsu threshold."""
        counts, bins = np.histogram(values, bins=256)
        total = counts.sum()
        sum_total = np.dot(counts, bins[:-1])
        s_b_max, best_t = 0.0, bins[1]
        w0 = sum0 = 0.0

        for cnt, b in zip(counts, bins[:-1]):
            w0 += cnt
            w1 = total - w0
            if w0 == 0 or w1 == 0:
                continue
            sum0 += cnt * b
            m0 = sum0 / w0
            m1 = (sum_total - sum0) / w1
            s_b = w0 * w1 * (m0 - m1) ** 2
            if s_b > s_b_max:
                s_b_max, best_t = s_b, b

        return float(best_t)


# ══════════════════════════════════════════════════════════════════════════════
class EllipsoidFitter:
    """
    Fits a 3-D ellipsoid to a binary mask using PCA on the voxel coordinates.

    The three principal axes of the point cloud become the ellipsoid semi-axes.
    A scale factor of 2.0 on sqrt(eigenvalue) encloses ~95 % of foreground voxels.

    Parameters
    ----------
    voxel_um : physical voxel edge length in micrometres
    """

    def __init__(self, voxel_um: float = 4.0):
        self.voxel_um = voxel_um

    def fit(self, mask: np.ndarray) -> dict | None:
        """
        Parameters
        ----------
        mask : boolean array (z, y, x)

        Returns
        -------
        dict with keys:
            center_vox    — centroid in voxel coords  (z, y, x)
            semi_axes_vox — semi-axis lengths in voxels, largest first
            semi_axes_um  — semi-axis lengths in µm
            axes_vectors  — (3, 3) eigenvectors (columns = principal directions)
            volume_um3    — ellipsoid volume in µm³
            diameter_um   — mean diameter in µm
        or None if too few foreground voxels.
        """
        coords = np.argwhere(mask).astype(float)
        if len(coords) < 10:
            return None

        center = coords.mean(axis=0)
        centered = coords - center
        cov = np.cov(centered.T)                     # (3, 3)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)   # ascending order

        # Reverse so index 0 = largest axis
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues  = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # For a uniform ellipsoid, sqrt(eigenvalue) ≈ semi-axis / sqrt(5).
        # Scale 2.5 ≈ sqrt(5) * 1.12 gives a circumscribing ellipsoid with
        # a small margin so that genuine protrusions (hatching) stand out clearly.
        semi_axes_vox = np.sqrt(np.abs(eigenvalues)) * 2.5
        semi_axes_um  = semi_axes_vox * self.voxel_um

        a, b, c = semi_axes_um
        volume_um3  = (4 / 3) * np.pi * a * b * c
        diameter_um = 2.0 * semi_axes_um.mean()

        return {
            "center_vox":    center,
            "semi_axes_vox": semi_axes_vox,
            "semi_axes_um":  semi_axes_um,
            "axes_vectors":  eigenvectors,
            "volume_um3":    volume_um3,
            "diameter_um":   diameter_um,
        }


# ══════════════════════════════════════════════════════════════════════════════
class HatchingDetector:
    """
    Detects blastocyst hatching by counting foreground voxels that lie
    significantly outside the fitted ellipsoid surface.

    Parameters
    ----------
    threshold_factor      : voxels with normalised distance > this value
                            are counted as protrusions (1.0 = on surface)
    min_protrusion_voxels : minimum protrusion count to trigger hatching flag
    """

    def __init__(self, threshold_factor: float = 1.20, min_protrusion_voxels: int = 10):
        self.threshold_factor = threshold_factor
        self.min_protrusion_voxels = min_protrusion_voxels

    def detect(self, mask: np.ndarray, ellipsoid: dict | None) -> dict:
        """
        Returns dict with keys:
            is_hatching         — bool
            protrusion_voxels   — int count of voxels outside ellipsoid
            protrusion_fraction — fraction of all foreground voxels
        """
        empty = {"is_hatching": False, "protrusion_voxels": 0, "protrusion_fraction": 0.0}

        if ellipsoid is None:
            return empty

        coords = np.argwhere(mask).astype(float)
        if len(coords) == 0:
            return empty

        centered = coords - ellipsoid["center_vox"]
        axes     = ellipsoid["semi_axes_vox"]
        dist_sq  = np.sum((centered / axes) ** 2, axis=1)

        outside  = dist_sq > self.threshold_factor ** 2
        n_prot   = int(outside.sum())
        frac     = n_prot / max(len(coords), 1)

        return {
            "is_hatching":         n_prot >= self.min_protrusion_voxels,
            "protrusion_voxels":   n_prot,
            "protrusion_fraction": round(frac, 4),
        }


# ══════════════════════════════════════════════════════════════════════════════
class EmbryoPipeline:
    """
    Convenience wrapper: raw 7 slices → size estimate + hatching result.

    Parameters
    ----------
    voxel_um            : physical voxel edge length in µm
    target_slices       : z-planes after interpolation
    threshold_factor    : passed to HatchingDetector
    min_protrusion_voxels: passed to HatchingDetector
    """

    def __init__(
        self,
        voxel_um: float = 4.0,
        target_slices: int = 56,
        threshold_factor: float = 1.20,
        min_protrusion_voxels: int = 10,
    ):
        self.builder  = VolumeBuilder(target_slices=target_slices, voxel_um=voxel_um)
        self.seg      = Segmentor()
        self.fitter   = EllipsoidFitter(voxel_um=voxel_um)
        self.detector = HatchingDetector(
            threshold_factor=threshold_factor,
            min_protrusion_voxels=min_protrusion_voxels,
        )

    def run(self, slices: np.ndarray) -> dict:
        """
        Parameters
        ----------
        slices : (7, H, W) float array, values in [0, 1]

        Returns
        -------
        dict with keys: interp_volume, mask, ellipsoid, hatching
        """
        interp    = self.builder.build(slices)
        mask      = self.seg.segment(interp)
        ellipsoid = self.fitter.fit(mask)
        hatching  = self.detector.detect(mask, ellipsoid)

        return {
            "interp_volume": interp,
            "mask":          mask,
            "ellipsoid":     ellipsoid,
            "hatching":      hatching,
        }

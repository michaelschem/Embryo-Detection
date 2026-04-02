"""
Synthetic blastocyst slice generator.

Each generated case is a (num_slices, H, W) float32 array where:
  1.0  = embryo / trophectoderm
  0.5  = zona pellucida shell
  0.0  = background
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import binary_fill_holes


@dataclass
class CaseConfig:
    name: str
    semi_axes: tuple = (0.55, 0.45, 0.70)       # normalised semi-axes in [-1,1] grid
    offcenter_shift: tuple = (0.0, 0.0, 0.0)     # (dx, dy, dz) fraction of grid
    irregularity: float = 0.0                    # boundary noise magnitude
    hatching: bool = False
    zona_shell: bool = True
    seed: Optional[int] = None


class BlastocystGenerator:
    """
    Generates synthetic 7-slice blastocyst scan data as NumPy arrays.

    Usage
    -----
    gen = BlastocystGenerator()
    slices, meta = gen.generate(config=CaseConfig("my_case", hatching=True))
    batch = gen.generate_batch()          # uses DEFAULT_CASES
    meta_path = gen.save_batch(batch, "data/")
    """

    DEFAULT_CASES: list[CaseConfig] = [
        CaseConfig("centred_smooth",
                   offcenter_shift=(0.00,  0.00,  0.00), irregularity=0.00, hatching=False, seed=10),
        CaseConfig("mild_offcentre",
                   offcenter_shift=(0.12, -0.08,  0.04), irregularity=0.05, hatching=False, seed=11),
        CaseConfig("strong_offcentre",
                   offcenter_shift=(0.25,  0.18, -0.08), irregularity=0.10, hatching=False, seed=12),
        CaseConfig("noisy_centred",
                   offcenter_shift=(0.00,  0.00,  0.00), irregularity=0.18, hatching=False, seed=13),
        CaseConfig("noisy_offcentre",
                   offcenter_shift=(0.15, -0.12,  0.06), irregularity=0.15, hatching=False, seed=14),
        CaseConfig("hatching_centred",
                   offcenter_shift=(0.00,  0.00,  0.00), irregularity=0.00, hatching=True,  seed=20),
        CaseConfig("hatching_mild_off",
                   offcenter_shift=(0.10, -0.07,  0.03), irregularity=0.06, hatching=True,  seed=21),
        CaseConfig("hatching_strong_off",
                   offcenter_shift=(0.22,  0.15, -0.07), irregularity=0.12, hatching=True,  seed=22),
        CaseConfig("hatching_noisy",
                   offcenter_shift=(0.00,  0.00,  0.00), irregularity=0.20, hatching=True,  seed=23),
        CaseConfig("hatching_noisy_offcentre",
                   offcenter_shift=(0.18, -0.14,  0.05), irregularity=0.16, hatching=True,  seed=24),
    ]

    # ------------------------------------------------------------------ #

    def generate(
        self,
        height: int = 64,
        width: int = 64,
        num_slices: int = 7,
        config: Optional[CaseConfig] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Generate one synthetic case.

        Returns
        -------
        slices : (num_slices, height, width) float32
        meta   : dict with all configuration info
        """
        if config is None:
            config = CaseConfig("default")

        if config.seed is not None:
            np.random.seed(config.seed)

        a, b, c = config.semi_axes
        dx, dy, dz = config.offcenter_shift

        # Coordinate grids in [-1, 1]
        x = np.linspace(-1, 1, height)[:, None, None]
        y = np.linspace(-1, 1, width)[None, :, None]
        z = np.linspace(-1, 1, num_slices)[None, None, :]

        xc, yc, zc = x - dx, y - dy, z - dz
        r2 = (xc / a) ** 2 + (yc / b) ** 2 + (zc / c) ** 2

        embryo = r2 <= 1.0

        if config.irregularity > 0:
            noise = np.random.normal(0, config.irregularity, embryo.shape)
            boundary = np.abs(r2 - 1.0) < 0.25
            embryo = np.logical_xor(embryo, boundary & (noise > 0.15))
            embryo = binary_fill_holes(embryo)

        vol = embryo.astype(float)

        if config.zona_shell:
            zona_scale = 1.12
            r2_zona = (
                (xc / (a * zona_scale)) ** 2
                + (yc / (b * zona_scale)) ** 2
                + (zc / (c * zona_scale)) ** 2
            )
            zona = (r2_zona <= 1.0) & ~embryo
            vol[zona] = 0.5

        if config.hatching:
            # Trophectoderm herniation: a 3-D bulge on one side of the zona.
            # Elongated slightly along z so it spans ~3 consecutive slices,
            # which matches the physical size of a real hatching protrusion
            # relative to the ~150 µm blastocyst diameter.
            px, py = 0.72 + dx, 0.0 + dy
            r2_bump = (xc - px) ** 2 + (yc - py) ** 2 + (zc / 1.8) ** 2
            vol[r2_bump < 0.10] = 1.0

        # Reorder axes: (z, y, x)  →  (slice, height, width)
        slices = np.moveaxis(vol, 2, 0).astype(np.float32)

        meta = {
            "name":            config.name,
            "semi_axes":       list(config.semi_axes),
            "offcenter_shift": list(config.offcenter_shift),
            "irregularity":    config.irregularity,
            "hatching":        config.hatching,
            "zona_shell":      config.zona_shell,
            "seed":            config.seed,
            "shape":           list(slices.shape),
        }
        return slices, meta

    # ------------------------------------------------------------------ #

    def generate_batch(
        self,
        cases: Optional[list[CaseConfig]] = None,
        **kwargs,
    ) -> list[tuple[np.ndarray, dict]]:
        """Generate all cases. Returns list of (slices, meta) tuples."""
        if cases is None:
            cases = self.DEFAULT_CASES
        return [self.generate(config=c, **kwargs) for c in cases]

    # ------------------------------------------------------------------ #

    def save_batch(self, batch: list[tuple[np.ndarray, dict]], output_dir: str) -> str:
        """
        Save each case as a .npy file and write a metadata.json index.

        Returns
        -------
        Path to the metadata.json file.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        all_meta = []
        for i, (slices, meta) in enumerate(batch):
            fname = f"case_{i:03d}_{meta['name']}.npy"
            np.save(out / fname, slices)
            all_meta.append({**meta, "case_id": i, "file": fname})

        meta_path = out / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(all_meta, f, indent=2)

        return str(meta_path)

    # ------------------------------------------------------------------ #

    @staticmethod
    def load_batch(data_dir: str) -> list[tuple[np.ndarray, dict]]:
        """
        Load a previously saved batch from data_dir.

        Returns list of (slices, meta) tuples in the original order.
        """
        data_dir = Path(data_dir)
        with open(data_dir / "metadata.json") as f:
            all_meta = json.load(f)

        batch = []
        for meta in sorted(all_meta, key=lambda m: m["case_id"]):
            slices = np.load(data_dir / meta["file"])
            batch.append((slices, meta))
        return batch

"""
Visualisation helpers for embryo slice data, segmentation masks, and
ellipsoid fitting results.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection


class Visualiser:
    """
    Rendering helpers for embryo analysis notebooks.

    All methods display figures inline (plt.show()) and return None so they
    are safe to call anywhere in a notebook cell.

    Parameters
    ----------
    cmap : matplotlib colormap for intensity images
    dpi  : figure dots-per-inch
    """

    def __init__(self, cmap: str = "gray", dpi: int = 110):
        self.cmap = cmap
        self.dpi  = dpi

    # ------------------------------------------------------------------ #
    # Raw slice grid

    def plot_slices(self, slices: np.ndarray, title: str = "") -> None:
        """Display each z-slice in a single row."""
        n = slices.shape[0]
        fig, axs = plt.subplots(1, n, figsize=(2 * n, 2.4), dpi=self.dpi)
        fig.suptitle(title, fontsize=10, y=1.02)
        for i, ax in enumerate(axs):
            ax.imshow(slices[i], cmap=self.cmap, vmin=0, vmax=1)
            ax.set_title(f"z={i}", fontsize=8)
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    # 3-D scatter

    def plot_3d(self, mask: np.ndarray, title: str = "") -> None:
        """Sparse 3-D scatter of foreground voxels (downsampled for speed)."""
        coords = np.argwhere(mask)
        step   = max(1, len(coords) // 2000)
        c      = coords[::step]

        fig = plt.figure(figsize=(5, 4), dpi=self.dpi)
        ax  = fig.add_subplot(111, projection="3d")
        ax.scatter(c[:, 2], c[:, 1], c[:, 0], s=1, alpha=0.3, c=c[:, 0], cmap="plasma")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title(title, fontsize=9)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    # Full results panel

    def plot_results_panel(
        self,
        raw_slices: np.ndarray,
        interp_volume: np.ndarray,
        mask: np.ndarray,
        ellipsoid: dict | None,
        hatching: dict,
        meta: dict,
    ) -> None:
        """
        Two-row summary panel:
          Row 0 — raw 7 slices
          Row 1 — corresponding interpolated slices with segmentation
                  contour (green) and ellipsoid cross-section (orange)
          Right column — metrics text box

        Parameters
        ----------
        raw_slices    : (7, H, W)
        interp_volume : (target_slices, H, W)
        mask          : (target_slices, H, W) bool
        ellipsoid     : dict from EllipsoidFitter.fit(), or None
        hatching      : dict from HatchingDetector.detect()
        meta          : case metadata dict
        """
        n_raw   = raw_slices.shape[0]
        n_interp = interp_volume.shape[0]

        # Choose interpolated z-indices that correspond to the original 7 slices
        show_z = np.linspace(0, n_interp - 1, n_raw, dtype=int)

        fig = plt.figure(figsize=(16, 5.5), dpi=self.dpi)
        gs  = GridSpec(2, n_raw + 1, figure=fig, hspace=0.08, wspace=0.05,
                       width_ratios=[1] * n_raw + [1.6])

        # ── Row 0: raw slices ──────────────────────────────────────────
        for i in range(n_raw):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(raw_slices[i], cmap=self.cmap, vmin=0, vmax=1)
            ax.set_title(f"raw z={i}", fontsize=7)
            ax.axis("off")

        # ── Row 1: interpolated slices + overlays ─────────────────────
        cz = cy = cx = az = ay = ax_len = None
        if ellipsoid is not None:
            cz, cy, cx     = ellipsoid["center_vox"]
            az, ay, ax_len = ellipsoid["semi_axes_vox"]

        for j, zi in enumerate(show_z):
            ax = fig.add_subplot(gs[1, j])
            ax.imshow(interp_volume[zi], cmap=self.cmap, vmin=0, vmax=1)
            ax.contour(mask[zi].astype(float), levels=[0.5], colors="lime", linewidths=0.8)

            if ellipsoid is not None:
                dz        = (zi - cz) / max(az, 1e-6)
                remaining = 1.0 - dz ** 2
                if remaining > 0:
                    ell = mpatches.Ellipse(
                        (cx, cy),
                        width=2 * ax_len * np.sqrt(remaining),
                        height=2 * ay * np.sqrt(remaining),
                        edgecolor="orange", facecolor="none", linewidth=1.0,
                    )
                    ax.add_patch(ell)

            ax.set_title(f"interp z={zi}", fontsize=7)
            ax.axis("off")

        # ── Legend on last overlay axis ───────────────────────────────
        ax.legend(
            handles=[
                mpatches.Patch(color="lime",   label="Segmentation"),
                mpatches.Patch(color="orange", label="Ellipsoid fit"),
            ],
            loc="lower right", fontsize=6,
        )

        # ── Right column: metrics text ─────────────────────────────────
        ax_t = fig.add_subplot(gs[:, -1])
        ax_t.axis("off")

        if ellipsoid is not None:
            a, b, c = ellipsoid["semi_axes_um"]
            lines = [
                f"Case: {meta.get('name', '')}",
                "",
                "Semi-axes (µm)",
                f"  a = {a:.1f}",
                f"  b = {b:.1f}",
                f"  c = {c:.1f}",
                "",
                f"Diameter : {ellipsoid['diameter_um']:.1f} µm",
                f"Volume   : {ellipsoid['volume_um3']:.0f} µm³",
                "",
                "─" * 20,
                f"Hatching : {'YES' if hatching['is_hatching'] else 'no'}",
                f"Protrusions: {hatching['protrusion_voxels']}",
                f"  ({100 * hatching['protrusion_fraction']:.1f} %)",
                "",
                "Ground truth:",
                f"  hatching = {meta.get('hatching', '?')}",
            ]
        else:
            lines = ["Ellipsoid fit failed\n(too few foreground\nvoxels)"]

        ax_t.text(
            0.08, 0.95, "\n".join(lines),
            transform=ax_t.transAxes,
            va="top", ha="left", fontsize=8, fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#f5f5f5", alpha=0.9),
        )

        # ── Figure title ──────────────────────────────────────────────
        status = "HATCHING" if hatching["is_hatching"] else "not hatching"
        color  = "#c62828" if hatching["is_hatching"] else "#2e7d32"
        fig.suptitle(
            f"{meta.get('name', 'Case')}  —  {status}",
            fontsize=11, color=color, fontweight="bold", y=1.01,
        )
        plt.show()

    # ------------------------------------------------------------------ #
    # 3-D scatter grid — all cases in one figure

    def plot_3d_grid(
        self,
        batch: list,
        title: str = "3-D shapes — all cases",
        elev: int = 25,
        azim: int = -55,
    ) -> None:
        """
        Overview grid: top row = no-hatching, bottom row = hatching.
        Each subplot is a 3-D scatter of the raw 7 slices.
        Zona pellucida voxels are light blue; embryo body is deeper blue
        (or red for hatching cases) so the protrusion stands out.

        Parameters
        ----------
        batch : list of (slices, meta) tuples from BlastocystGenerator
        """
        no_hatch = [(s, m) for s, m in batch if not m["hatching"]]
        hatching  = [(s, m) for s, m in batch if     m["hatching"]]
        n_col = max(len(no_hatch), len(hatching))

        fig = plt.figure(figsize=(n_col * 2.6, 5.8), dpi=self.dpi)

        for row, group in enumerate([no_hatch, hatching]):
            for col, (slices, meta) in enumerate(group):
                ax = fig.add_subplot(2, n_col, row * n_col + col + 1, projection="3d")

                zona_coords   = np.argwhere((slices > 0.25) & (slices < 0.75))
                embryo_coords = np.argwhere(slices > 0.75)

                def _scatter(coords, color, alpha, s):
                    if len(coords) == 0:
                        return
                    step = max(1, len(coords) // 600)
                    c = coords[::step]
                    # map (slice, row, col) → (x=col, y=row, z=slice)
                    ax.scatter(c[:, 2], c[:, 1], c[:, 0],
                               s=s, alpha=alpha, color=color, linewidths=0)

                _scatter(zona_coords,   "#b0bec5", 0.12, 2)
                body_col = "#ef5350" if meta["hatching"] else "#1e88e5"
                _scatter(embryo_coords, body_col,  0.55, 3)

                ax.set_title(meta["name"].replace("_", "\n"), fontsize=6, pad=2)
                for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                    axis.set_ticklabels([])
                    axis.label.set_text("")
                ax.view_init(elev=elev, azim=azim)
                ax.set_box_aspect([1, 1, 0.55])

        fig.text(0.01, 0.75, "No hatching", va="center", rotation=90,
                 fontsize=8, color="#1e88e5", fontweight="bold")
        fig.text(0.01, 0.27, "Hatching", va="center", rotation=90,
                 fontsize=8, color="#ef5350", fontweight="bold")
        fig.suptitle(title, fontsize=10, y=1.01)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    # 3-D voxel rendering — detailed side-by-side comparison

    def plot_voxel_comparison(
        self,
        case_a: tuple,
        case_b: tuple,
        title: str = "Normal vs Hatching — 3-D voxel view",
        elev: int = 22,
        azim: int = -50,
    ) -> None:
        """
        Side-by-side ax.voxels() rendering of two cases.
        Zona pellucida = translucent grey; embryo = blue (normal) or red (hatching).

        Parameters
        ----------
        case_a, case_b : (slices, meta) tuples
        """
        from scipy.ndimage import zoom

        fig = plt.figure(figsize=(10, 4.5), dpi=self.dpi)

        for col, (slices, meta) in enumerate([case_a, case_b], 1):
            # Downsample xy for voxels() performance: (7, ~22, ~22)
            ds = zoom(slices, (1.0, 0.34, 0.34), order=0)

            embryo = ds > 0.75
            zona   = (ds > 0.25) & ~embryo

            body_rgba = [0.12, 0.47, 0.91, 0.90] if not meta["hatching"] \
                   else [0.90, 0.20, 0.16, 0.90]

            colors = np.zeros(ds.shape + (4,), dtype=float)
            colors[zona]   = [0.78, 0.82, 0.84, 0.25]
            colors[embryo] = body_rgba

            filled = embryo | zona

            ax = fig.add_subplot(1, 2, col, projection="3d")
            # ax.voxels expects (x, y, z) order
            ax.voxels(
                np.moveaxis(filled, 0, 2),
                facecolors=np.moveaxis(colors, 0, 2),
            )

            label  = "HATCHING" if meta["hatching"] else "Normal"
            lcolor = "#c62828"  if meta["hatching"] else "#1565c0"
            ax.set_title(f"{meta['name']}\n{label}", fontsize=9, color=lcolor,
                         fontweight="bold")
            ax.set_xlabel("x", fontsize=7)
            ax.set_ylabel("y", fontsize=7)
            ax.set_zlabel("z (slice)", fontsize=7)
            ax.tick_params(labelsize=5)
            ax.view_init(elev=elev, azim=azim)

        fig.suptitle(title, fontsize=11)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    # Batch summary bar chart

    def plot_batch_summary(self, records: list[dict]) -> None:
        """
        Bar chart of estimated diameters for all cases, coloured by
        hatching status.

        Parameters
        ----------
        records : list of dicts, each with keys:
                    name, diameter_um, is_hatching
        """
        names  = [r["name"] for r in records]
        diams  = [r["diameter_um"] if r["diameter_um"] is not None else 0 for r in records]
        colors = ["#c62828" if r["is_hatching"] else "#1565c0" for r in records]

        fig, ax = plt.subplots(figsize=(max(8, len(records) * 1.1), 4), dpi=self.dpi)
        bars = ax.bar(range(len(names)), diams, color=colors, edgecolor="white", linewidth=0.6)

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Estimated diameter (µm)", fontsize=9)
        ax.set_title("Predicted embryo size by case", fontsize=10)

        ax.legend(
            handles=[
                mpatches.Patch(color="#c62828", label="Hatching detected"),
                mpatches.Patch(color="#1565c0", label="Not hatching"),
            ],
            fontsize=8,
        )

        for bar, val in zip(bars, diams):
            if val:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=7)

        plt.tight_layout()
        plt.show()

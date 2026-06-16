"""Figure quality-control helpers used by the QCagent and inline by the figure
scripts.

Two complementary checks:
  * ``layout_issues(fig)`` -- inspects a live matplotlib Figure via renderer
    bounding boxes and flags (a) overlaps that involve the suptitle or a legend
    against the panels (the failure mode we hit before: legend over the maps),
    and (b) excessive empty space (large all-background margins/bands).
  * ``raster_issues(png)`` -- post-hoc check on a saved PNG: the fraction of
    background pixels and the tallest fully-empty horizontal band, to catch
    "too much white space" without needing the original Figure object.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def _bbox(artist, renderer):
    try:
        bb = artist.get_window_extent(renderer)
    except Exception:
        return None
    if bb is None or bb.width <= 0 or bb.height <= 0:
        return None
    return bb


def layout_issues(fig, tol_px: float = 3.0, whitespace_warn: float = 0.80) -> list[str]:
    """Return a list of human-readable layout problems for a drawn Figure.

    Overlap reporting is restricted to pairs where at least one member is the
    suptitle or a legend (panel-vs-panel tight-bbox touches are expected on a
    dense grid and are ignored). ``tol_px`` is the minimum overlap in BOTH
    dimensions before it is reported.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    from matplotlib.transforms import Bbox

    # specials carry an optional host-axes index so a legend is never flagged
    # against the axes it lives in. panels use the TIGHT bbox so a panel TITLE
    # poking up into the legend/suptitle is caught (the data-area extent misses
    # titles, which was how an earlier legend-over-title overlap slipped through).
    panels, specials = [], []
    if getattr(fig, "_suptitle", None) is not None:
        bb = _bbox(fig._suptitle, r)
        if bb is not None:
            specials.append(("suptitle", bb, None))
    for leg in list(fig.legends):
        bb = _bbox(leg, r)
        if bb is not None:
            specials.append(("legend", bb, None))
    for i, ax in enumerate(fig.axes):
        if not ax.get_visible():
            continue
        lg = ax.get_legend()
        if lg is not None:
            bb = _bbox(lg, r)
            if bb is not None:
                specials.append((f"axes[{i}].legend", bb, i))
        try:
            bb = ax.get_tightbbox(r)
        except Exception:
            bb = _bbox(ax, r)
        if bb is not None and bb.width > 0 and bb.height > 0:
            panels.append((f"axes[{i}]", bb, i))

    issues = []
    for sname, sbb, shost in specials:
        for pname, pbb, phost in panels:
            if shost is not None and shost == phost:
                continue                       # a legend over its own host axes is fine
            inter = Bbox.intersection(sbb, pbb)
            if inter is not None and inter.width > tol_px and inter.height > tol_px:
                issues.append(f"OVERLAP {sname} over {pname} "
                              f"({inter.width:.0f}x{inter.height:.0f} px)")
    # special-vs-special (e.g. legend over suptitle)
    for a in range(len(specials)):
        for b in range(a + 1, len(specials)):
            na, ba, _ = specials[a]
            nb, bb, _ = specials[b]
            inter = Bbox.intersection(ba, bb)
            if inter is not None and inter.width > tol_px and inter.height > tol_px:
                issues.append(f"OVERLAP {na} <-> {nb} "
                              f"({inter.width:.0f}x{inter.height:.0f} px)")

    issues += _whitespace_from_render(fig, whitespace_warn)
    return issues


def _render_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return buf[..., :3]


def _whitespace_metrics(rgb: np.ndarray, bg_thresh: int = 250):
    """Background fraction and tallest all-background horizontal band fraction."""
    bg_row = (rgb >= bg_thresh).all(axis=2)          # H x W bool, True = background
    frac = float(bg_row.mean())
    empty_rows = bg_row.all(axis=1)                  # fully blank scanlines
    # tallest run of blank rows that is NOT the outer top/bottom margin
    longest = cur = 0
    for v in empty_rows[1:-1]:
        cur = cur + 1 if v else 0
        longest = max(longest, cur)
    band_frac = longest / rgb.shape[0]
    return frac, band_frac


def _whitespace_from_render(fig, whitespace_warn: float) -> list[str]:
    rgb = _render_rgb(fig)
    frac, band = _whitespace_metrics(rgb)
    out = []
    if frac > whitespace_warn:
        out.append(f"WHITESPACE background fraction {frac:.0%} > {whitespace_warn:.0%}")
    if band > 0.12:
        out.append(f"WHITESPACE empty horizontal band {band:.0%} of figure height "
                   "(panels too far apart / gap)")
    return out


def raster_issues(png_path, whitespace_warn: float = 0.80) -> list[str]:
    """Post-hoc whitespace check on a saved PNG (no Figure object needed)."""
    try:
        from matplotlib import image as mpimg
    except Exception as exc:  # pragma: no cover
        return [f"could not load {png_path}: {exc}"]
    img = mpimg.imread(str(png_path))
    rgb = (img[..., :3] * 255).astype(np.uint8) if img.dtype != np.uint8 else img[..., :3]
    frac, band = _whitespace_metrics(rgb)
    out = []
    if frac > whitespace_warn:
        out.append(f"{png_path}: background {frac:.0%} > {whitespace_warn:.0%} (too much space)")
    if band > 0.12:
        out.append(f"{png_path}: empty band {band:.0%} of height (panels too far apart)")
    return out


def report(fig=None, png_path=None, name: str = "figure") -> list[str]:
    """Run whichever checks are available and log them; return the issue list."""
    issues = []
    if fig is not None:
        issues += layout_issues(fig)
    if png_path is not None:
        issues += raster_issues(png_path)
    if issues:
        log.warning("QC: %s has %d layout issue(s):", name, len(issues))
        for it in issues:
            log.warning("QC:   - %s", it)
    else:
        log.info("QC: %s layout OK", name)
    return issues

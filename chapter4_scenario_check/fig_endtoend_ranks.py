"""Figure for the dissertation, chapter 4, item 4.7.7: rank probabilities of the five scenario variants under S3.

Written on 27 October 2026, after the analysis. Not part of the frozen plan (PROTOCOL.md, Section 7) and not listed in
protocol/freeze.json. It calls no model and only reads results/main/endtoend_results.json (SHA-256 checked below), so
the frozen record stays as it is. Output: figures/ch4_endtoend_ranks.png in this experiment folder (the integrator copies
it to draft/figures/).

Style follows scripts/figures_ch4.py of the dissertation: printed width 16 cm, Times New Roman, 11 pt text and 10 pt
cell values (both above the 9 pt minimum, checked by check()), greyscale heat map, decimal comma, half-up rounding, and
three decimals where two-decimal rounding would reach a threshold (0,8; 0,9; 0,95) from below (0,949, not 0,95).

Usage (from the experiment folder): python -B scripts/fig_endtoend_ranks.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

sys.dont_write_bytecode = True

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.text import Text  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results" / "main" / "endtoend_results.json"
RESULTS_SHA256 = "187444255ee1de9cdfc6adb6a9d379418ee1e8e7af0133e97b285806f42fa4cc"
OUT = HERE / "figures"

CM = 1 / 2.54
W = 16.0
FS, FS_CELL, MIN_PT = 11.0, 10.0, 9.0
plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": FS,
                     "axes.linewidth": 0.6, "savefig.dpi": 300})
INK, MID, LIGHT = "0.0", "0.35", "0.75"
GREYS = LinearSegmentedColormap.from_list("acceptability", ["1.0", "0.18"])
THRESHOLDS = (0.8, 0.9, 0.95)

# Row order = order of the pooled group S1 in both modes (RESULT.md, Section 4). Short names as on Fig. 2.4
# (ROUTES in scripts/figures_ch3.py of the dissertation).
ORDER = ["DS", "MP", "EE", "PC", "RS"]
NAMES = {"DS": "Сеанс диспетчера (СД)",
         "MP": "Облікові дані підрядника (ПД)",
         "EE": "Корпоративний вузол (КВ)",
         "PC": "Облікові дані персоналу (ОП)",
         "RS": "Віддалений сервіс (ВС)"}
PANELS = [("ratings", "Режим оцінок"), ("relative", "Режим парних порівнянь")]


def half_up(x: float, nd: int = 2) -> Decimal:
    return Decimal(str(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)


def comma(x: float, nd: int = 2) -> str:
    return f"{half_up(x, nd):.{nd}f}".replace(".", ",")


def cell_text(v: float) -> str:
    if half_up(v) >= 1:
        return "1"
    if any(v < t <= float(half_up(v)) for t in THRESHOLDS):
        return comma(v, 3)
    return comma(v)


def load() -> dict:
    raw = RESULTS.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != RESULTS_SHA256:
        raise SystemExit(f"{RESULTS} has SHA-256 {digest}, expected {RESULTS_SHA256}")
    return json.loads(raw.decode("utf-8"))


def heat(ax, b: np.ndarray, labels: list[str], title: str) -> None:
    n_alt, n_rank = b.shape
    ax.imshow(b, cmap=GREYS, vmin=0, vmax=1, aspect="auto")
    for i in range(n_alt):
        for j in range(n_rank):
            v = float(b[i, j])
            if v >= 0.005:
                ax.text(j, i, cell_text(v), ha="center", va="center", fontsize=FS_CELL,
                        color="white" if v >= 0.5 else INK, gid="cell")
    ax.set_xticks(range(n_rank)); ax.set_xticklabels([str(r + 1) for r in range(n_rank)], fontsize=FS)
    ax.set_yticks(range(n_alt)); ax.set_yticklabels(labels, fontsize=FS)
    ax.set_xticks(np.arange(-0.5, n_rank), minor=True); ax.set_yticks(np.arange(-0.5, n_alt), minor=True)
    ax.grid(which="minor", color=LIGHT, lw=0.5)
    ax.tick_params(which="both", length=0)
    for s_ in ax.spines.values():
        s_.set_color(MID); s_.set_linewidth(0.6)
    ax.set_title(title, fontsize=FS, pad=4)
    ax.set_xlabel("Ранг", fontsize=FS, labelpad=2)


def check(fig, name: str) -> None:
    """Every text at least 9 pt (11 pt, cell values 10 pt), inside the figure, and no two texts overlap."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    boxes = []
    for t in texts:
        size = t.get_fontsize()
        assert size >= MIN_PT - 1e-6, (name, t.get_text(), size)
        assert size >= (FS_CELL if t.get_gid() == "cell" else FS) - 1e-6, (name, t.get_text(), size)
        bb = t.get_window_extent(r)
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > fig.bbox.width + 1 or bb.y1 > fig.bbox.height + 1:
            raise AssertionError(f"{name}: text outside the figure: {t.get_text()!r}")
        boxes.append((bb, t.get_text()))
    clashes = [(a[1], b[1]) for i, a in enumerate(boxes) for b in boxes[i + 1:] if a[0].overlaps(b[0])]
    assert not clashes, (name, clashes)
    print(f"{name}: {len(texts)} texts, smallest {min(t.get_fontsize() for t in texts):.0f} pt, no overlaps")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    R = load()
    variants = R["variants"]                      # PC, DS, RS, MP, EE: rows of b
    idx = [variants.index(v) for v in ORDER]
    labels = [NAMES[v] for v in ORDER]
    H = 5.6
    fig, axes = plt.subplots(1, 2, figsize=(W * CM, H * CM), sharey=True)
    fig.subplots_adjust(left=0.325, right=0.995, top=0.89, bottom=0.17, wspace=0.05)
    for ax, (mode, title) in zip(axes, PANELS):
        b = np.array(R["smaa"]["S3"][mode]["b"])[idx, :]
        assert np.allclose(b.sum(axis=1), 1.0, atol=1e-9) and np.allclose(b.sum(axis=0), 1.0, atol=1e-9)
        heat(ax, b, labels, title)
        for v, row in zip(ORDER, b):
            print(mode, v, " ".join(f"{x:.4f}" for x in row))
    axes[1].tick_params(labelleft=False)
    check(fig, "рис. 4.6")
    OUT.mkdir(exist_ok=True)
    path = OUT / "ch4_endtoend_ranks.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"written {path} ({W:.0f} cm wide, SHA-256 {hashlib.sha256(path.read_bytes()).hexdigest()})")


if __name__ == "__main__":
    main()

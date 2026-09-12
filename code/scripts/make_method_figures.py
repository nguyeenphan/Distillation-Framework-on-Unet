"""
make_method_figures.py — Figures for the Methodology chapter
────────────────────────────────────────────────────────────
Each figure defends one design decision with evidence, rather than
illustrating the method in general:

  fig-two-axes           the directional asymmetry (the contribution itself)
  fig-donor-degeneracy   why the content donor must be a different leaf
  fig-label-validity     how the ceiling on t was established
  fig-rank-saturation    why k is not a hyperparameter worth sweeping

    python scripts/make_method_figures.py --out images
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import LeafDiseaseDataset
from augmentation.stycona import StyConaAugmentor

INK, MUTED, ACCENT, LINE = "#1a1c18", "#5a5d54", "#2f7d4f", "#c9cdc0"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": LINE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def outline(img, mask, color=(220, 40, 40), w=2):
    """Draw the mask boundary rather than a filled overlay — the point of these
    panels is whether the lesion edge still sits on the label."""
    out = img.copy()
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cs, -1, color, w)
    return out


def show(ax, img, title, sub=None):
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(LINE)
    ax.set_title(title, fontsize=9, color=INK, pad=4)
    if sub:
        ax.set_xlabel(sub, fontsize=7.5, color=MUTED, labelpad=3)


def pick_leaf(ds, rank=0):
    """A real photo whose lesion is large enough to judge boundary drift.
    Must be a base photo: a CAST-restyled source would make the figure
    unreadable, since the reader could not tell source from augmentation."""
    base = [i for i, s in enumerate(ds.samples)
            if ds.records[s]["style_idx"] == "base"]
    # Target ~25% lesion area: a full-frame mask hides the boundary, which is
    # exactly what these panels are meant to let the reader check.
    areas = sorted((abs(ds[i]["mask"].numpy().mean() - 0.25), i) for i in base)
    return ds[areas[rank][1]]


def mix_ranks(xi, xj, t, ranks, blend_sigma, alpha):
    """StyCona at an arbitrary rank set — the published algorithm mixes t
    RANDOMLY selected content maps, which StyConaAugmentor (top-k) cannot express."""
    out = np.empty(xi.shape, dtype=np.float32)
    for c in range(xi.shape[2]):
        Ui, si, Vi = np.linalg.svd(xi[:, :, c].astype(np.float32), full_matrices=False)
        Uj, sj, Vj = np.linalg.svd(xj[:, :, c].astype(np.float32), full_matrices=False)
        s = alpha * si + (1 - alpha) * sj if blend_sigma else si
        U, V = Ui.copy(), Vi.copy()
        U[:, ranks] = (1 - t) * Ui[:, ranks] + t * Uj[:, ranks]
        V[ranks, :] = (1 - t) * Vi[ranks, :] + t * Vj[ranks, :]
        out[:, :, c] = U * s[None, :] @ V
    return np.clip(out, 0, 255)


def local_detail(d):
    """Energy of the change above a 9x9 box blur: local structure, not global tone."""
    g = d.mean(2)
    return float(np.abs(g - cv2.blur(g, (9, 9))).mean())


def aug(sc, t, k):
    return StyConaAugmentor(tuple(sc["style_alpha_range"]), True, t, k,
                            sc["per_channel_svd"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/spectral.yaml")
    p.add_argument("--out", default="images")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dcfg, sc = cfg["data"], cfg["stycona"]
    cm = sc["content_mix"]
    k_cfg = cm["top_k_ranks"]
    t_lo, t_hi = (cm["t"] if isinstance(cm["t"], list) else [cm["t"], cm["t"]])

    ds = LeafDiseaseDataset(dcfg["train_dir"], dcfg["image_size"], None,
                            False, True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    rec = pick_leaf(ds)
    x = (rec["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    y = rec["mask"].numpy()
    cid = rec["content_id"]
    style_aux = ds._sample_auxiliary(cid, -1)                # same leaf, CAST
    content_aux = ds._sample_other_leaf(cid)                 # other leaf, real

    # ── 1. the directional asymmetry ──────────────────────────────────
    a = aug(sc, (t_lo, t_hi), k_cfg)
    x_s = a(x, style_aux, mode="style").astype(np.uint8)
    x_c = a(x, content_aux, mode="content").astype(np.uint8)

    fig, axs = plt.subplots(1, 5, figsize=(11.5, 2.9))
    show(axs[0], outline(x, y), "$x$  (source)", "supervised by $y$")
    show(axs[1], style_aux, "style donor $x_j$", "CAST variant, same leaf")
    show(axs[2], outline(x_s, y), r"$x_{\mathrm{style}}$  —  blend $\sigma$ only",
         "structure preserved  →  INVARIANCE")
    show(axs[3], content_aux, "content donor $x_j$", "real photo, different leaf")
    show(axs[4], outline(x_c, y), r"$x_{\mathrm{content}}$  —  mix $u,v$ only",
         "structure moved  →  supervised by $y$ only")
    for i in (2, 4):
        axs[i].set_title(axs[i].get_title(), color=ACCENT, fontsize=9)
    fig.savefig(out / "fig-two-axes.png"); plt.close(fig)

    # ── 2. donor degeneracy ───────────────────────────────────────────
    same_leaf = ds._sample_auxiliary(cid, -1)
    x_c_same = a(x, same_leaf, mode="content").astype(np.uint8)
    d_same = np.abs(x_c_same.astype(float) - x).mean()
    d_diff = np.abs(x_c.astype(float) - x).mean()

    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9))
    show(axs[0], x, "$x$  (source)")
    show(axs[1], x_c_same, "donor = same leaf (CAST)",
         f"$|\\Delta|$ = {d_same:.1f}   —  content axis inert")
    show(axs[2], x_c, "donor = different leaf",
         f"$|\\Delta|$ = {d_diff:.1f}   —  content axis active")
    axs[2].set_title(axs[2].get_title(), color=ACCENT)
    fig.savefig(out / "fig-donor-degeneracy.png"); plt.close(fig)

    # ── 3. label validity ceiling ─────────────────────────────────────
    ts = [0.15, 0.3, 0.5, 0.7, 0.9]
    fig, axs = plt.subplots(1, len(ts) + 1, figsize=(2.0 * (len(ts) + 1), 2.9))
    show(axs[0], outline(x, y), "$x$ + label $y$")
    for i, t in enumerate(ts, 1):
        xt = aug(sc, t, k_cfg)(x, content_aux, mode="content").astype(np.uint8)
        d = np.abs(xt.astype(float) - x).mean()
        ok = t <= t_hi
        show(axs[i], outline(xt, y), f"$t$ = {t}", f"$|\\Delta|$ = {d:.1f}")
        axs[i].set_title(axs[i].get_title(), color=ACCENT if ok else "#b03030")
        if not ok:
            axs[i].set_xlabel(f"$|\\Delta|$ = {d:.1f}  —  rejected",
                              fontsize=7.5, color="#b03030")
    fig.savefig(out / "fig-label-validity.png"); plt.close(fig)

    # ── 4. rank saturation ────────────────────────────────────────────
    ks = [1, 2, 3, 4, 8, 16, 32, 64, 128]
    idxs = np.linspace(0, len(ds) - 1, 12).astype(int)
    samples = []
    for i in idxs:
        r = ds[int(i)]
        samples.append(((r["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                        ds._sample_other_leaf(r["content_id"])))
    deltas = [float(np.mean([np.abs(aug(sc, 0.4, k)(xi, xj, mode="content") - xi).mean()
                             for xi, xj in samples])) for k in ks]

    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    ax.plot(ks, deltas, color=ACCENT, lw=2, marker="o", ms=5,
            markerfacecolor="white", markeredgewidth=1.8, clip_on=False, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks); ax.set_xticklabels(ks)
    ax.set_xlabel("number of perturbed content maps  $k$")
    ax.set_ylabel(r"mean $|\Delta|$  (0–255)")
    ax.set_ylim(0, max(deltas) * 1.18)
    ax.grid(axis="y", color=LINE, lw=0.6, alpha=0.7); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.axvline(k_cfg, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.annotate(f"$k$ = {k_cfg} used\n(plateau)", xy=(k_cfg, deltas[ks.index(k_cfg)]),
                xytext=(6, -26), textcoords="offset points",
                fontsize=8, color=MUTED, ha="left")
    ax.annotate(f"{deltas[0]:.0f}", xy=(ks[0], deltas[0]), xytext=(0, 8),
                textcoords="offset points", fontsize=8, color=INK, ha="center")
    ax.annotate(f"{deltas[-1]:.0f}", xy=(ks[-1], deltas[-1]), xytext=(0, 8),
                textcoords="offset points", fontsize=8, color=INK, ha="center")
    fig.savefig(out / "fig-rank-saturation.png"); plt.close(fig)

    # ── 5. published StyCona vs the directional formulation ───────────
    R = min(x.shape[:2])
    paper_ranks = rng.choice(R, 16, replace=False)
    x_paper = mix_ranks(x, content_aux, rng.uniform(0, 1), paper_ranks,
                        True, rng.uniform(0, 1)).astype(np.uint8)

    fig, axs = plt.subplots(1, 4, figsize=(9.3, 3.0))
    show(axs[0], outline(x, y), "$x$  (source)", "shared label $y$")
    show(axs[1], outline(x_paper, y), "StyCona (published)",
         r"both axes at once  →  $\mathcal{L}_{\mathrm{seg}}$ (ERM)")
    show(axs[2], outline(x_s, y), r"ours:  $x_{\mathrm{style}}$",
         r"$\sigma$ only  →  invariance, stop-grad")
    show(axs[3], outline(x_c, y), r"ours:  $x_{\mathrm{content}}$",
         r"$u,v$ only  →  supervised by $y$")
    for i in (2, 3):
        axs[i].set_title(axs[i].get_title(), color=ACCENT)
    fig.savefig(out / "fig-vs-paper.png"); plt.close(fig)

    # ── 6. rank selector: random (paper) vs top-k (ours) ──────────────
    d_rand, h_rand, d_top, h_top = [], [], [], []
    for xi, xj in samples:
        dr = np.abs(mix_ranks(xi, xj, 0.4, rng.choice(R, 16, replace=False),
                              False, 1.0) - xi)
        dt = np.abs(aug(sc, 0.4, 16)(xi, xj, mode="content") - xi)
        d_rand.append(dr.mean()); h_rand.append(local_detail(dr))
        d_top.append(dt.mean()); h_top.append(local_detail(dt))

    x_rand = mix_ranks(x, content_aux, 0.4, rng.choice(R, 16, replace=False),
                       False, 1.0).astype(np.uint8)
    x_top = aug(sc, 0.4, 16)(x, content_aux, mode="content").astype(np.uint8)

    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.0))
    show(axs[0], x, "$x$  (source)")
    show(axs[1], x_rand, "random 16 ranks  (published)",
         f"$|\\Delta|$ = {np.mean(d_rand):.1f},  local = {np.mean(h_rand):.2f}")
    show(axs[2], x_top, "top-16 ranks  (ours)",
         f"$|\\Delta|$ = {np.mean(d_top):.1f},  local = {np.mean(h_top):.2f}")
    axs[2].set_title(axs[2].get_title(), color=ACCENT)
    fig.savefig(out / "fig-rank-selector.png"); plt.close(fig)

    print(f"\nrank selector (12 leaves): random-16 |D|={np.mean(d_rand):.2f} "
          f"local={np.mean(h_rand):.2f}  |  top-16 |D|={np.mean(d_top):.2f} "
          f"local={np.mean(h_top):.2f}")

    print(f"\nk sweep: " + "  ".join(f"k={k}:{d:.1f}" for k, d in zip(ks, deltas)))
    for f in sorted(out.glob("fig-*.png")):
        print(f"  {f}  ({f.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()

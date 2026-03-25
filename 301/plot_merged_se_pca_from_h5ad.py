#!/usr/bin/env python3
"""
从已合并、已跑过 SE 的 h5ad 仅重画 PCA 散点图（不调用 state emb）。
默认输出 .svg，散点层栅格化（SVG 内嵌位图），图例与坐标轴仍为矢量。

示例：
  python plot_merged_se_pca_from_h5ad.py \\
    --input "/data2/zengjh/301/T cells/T_cells_merged_SE_output/Tcells_merged_after_se.h5ad"

若 h5ad 中已有 PCA 坐标（如 obsm['X_state_pca_merged']），可用 --use-pca-key 直接画图，与当时矩阵一致。

只把「按 merge_group 着色」的总图拆成多张单色图（与总图同色、同坐标轴），不覆盖原总图：
  python plot_merged_se_pca_from_h5ad.py --only-merge-group-splits \\
    --input \".../Tcells_merged_after_se.h5ad\" \\
    --output-dir \".../split_merge_group\"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from run_se_pca import _pc_axis_label  # noqa: E402

MERGE_GROUP_COLORS: dict[str, str] = {
    "Ctrl": "#1f77b4",
    "RC": "#ff7f0e",
    "RP": "#2ca02c",
}


def _figure_prefix_from_stem(stem: str) -> str:
    """Tcells_merged_after_se / Tcells_merged_state_pca -> Tcells_merged"""
    for suf in ("_after_se", "_state_pca"):
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def _square_xy_limits(
    xlim: tuple[float, float], ylim: tuple[float, float]
) -> tuple[tuple[float, float], tuple[float, float]]:
    cx = 0.5 * (xlim[0] + xlim[1])
    cy = 0.5 * (ylim[0] + ylim[1])
    half = 0.5 * max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    if half <= 0:
        half = 1.0
    return (cx - half, cx + half), (cy - half, cy + half)


def _rasterize_scatter(ax: plt.Axes, fmt: str) -> None:
    if fmt in ("svg", "pdf", "eps"):
        for coll in ax.collections:
            coll.set_rasterized(True)


def _save_fig(fig: plt.Figure, out: Path, *, dpi: int = 150) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = out.suffix.lower().removeprefix(".") or "png"
    fig.savefig(out, format=fmt, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_pca_panel(
    z: np.ndarray,
    hue: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    out: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    legend_title: str,
    *,
    palette: dict[str, str] | None = None,
    hue_order: list[str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    h = hue.astype(str)
    kw: dict = {"x": z[:, 0], "y": z[:, 1], "hue": h, "s": 3, "linewidth": 0, "ax": ax, "legend": "full"}
    if palette is not None:
        in_data = set(h.unique())
        order_src = hue_order if hue_order is not None else sorted(in_data)
        pal = {c: palette[c] for c in order_src if c in in_data and c in palette}
        kw["palette"] = pal
    if hue_order is not None:
        kw["hue_order"] = hue_order
    sns.scatterplot(**kw)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1), title=legend_title, frameon=False, fontsize=8)
    fig.tight_layout()
    _rasterize_scatter(ax, out.suffix.lower().removeprefix(".") or "png")
    _save_fig(fig, out)


def _plot_pca_single_merge_group(
    z: np.ndarray,
    merge_group: pd.Series,
    group: str,
    color: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    mg = merge_group.astype(str)
    mask = mg == group
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(z[mask, 0], z[mask, 1], c=color, s=3, linewidth=0, label=group)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), title="merge_group", frameon=False, fontsize=8)
    fig.tight_layout()
    _rasterize_scatter(ax, out.suffix.lower().removeprefix(".") or "png")
    _save_fig(fig, out)


def _merge_group_order(series: pd.Series) -> list[str]:
    if isinstance(series.dtype, pd.CategoricalDtype):
        return [str(c) for c in series.cat.categories]
    # Ctrl, RC, RP 优先
    preferred = ["Ctrl", "RC", "RP"]
    u = sorted(series.astype(str).unique())
    ordered = [p for p in preferred if p in u]
    ordered.extend(x for x in u if x not in ordered)
    return ordered


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="从已有合并 h5ad 重画 merged SE PCA 图（SVG 栅格化散点）")
    p.add_argument("--input", required=True, type=Path, help="如 Tcells_merged_after_se.h5ad 或 Tcells_merged_state_pca.h5ad")
    p.add_argument("--output-dir", type=Path, default=None, help="默认与输入文件同目录")
    p.add_argument(
        "--figure-prefix",
        type=str,
        default=None,
        help="输出文件名前缀，默认由输入 stem 推断（去掉 _after_se / _state_pca）",
    )
    p.add_argument("--embed-key", default="X_state", help="在 h5ad 上做 PCA 的 obsm 键（与 --use-pca-key 二选一）")
    p.add_argument(
        "--use-pca-key",
        default=None,
        help="若设置，直接使用 obsm 该矩阵前两列作为 PC1/PC2，不再 fit PCA",
    )
    p.add_argument("--merge-group-key", default="merge_group")
    p.add_argument("--batch-key", default="Batch")
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--title-prefix", default=None, help="图标题前缀，默认用 figure-prefix")
    p.add_argument("--no-per-group", action="store_true", help="不输出 merge_group_only_*.svg")
    p.add_argument(
        "--only-merge-group-splits",
        action="store_true",
        help="只生成 merge_group 拆分的单色图，不写 by_merge_group / by_Batch 总图；不要求 obs 中有 Batch",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    inp = args.input.resolve()
    if not inp.is_file():
        print(f"找不到输入: {inp}", file=sys.stderr)
        return 1

    out_dir = args.output_dir.resolve() if args.output_dir else inp.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.figure_prefix or _figure_prefix_from_stem(inp.stem)
    title_base = args.title_prefix or prefix.replace("_", " ") + " merged · SE embedding · PCA"

    print(f"读取 {inp} ...", flush=True)
    adata = ad.read_h5ad(str(inp))

    if args.merge_group_key not in adata.obs.columns:
        print(f"obs 缺少 {args.merge_group_key!r}", file=sys.stderr)
        return 1
    if not args.only_merge_group_splits and args.batch_key not in adata.obs.columns:
        print(f"obs 缺少 {args.batch_key!r}", file=sys.stderr)
        return 1

    if args.use_pca_key:
        key = args.use_pca_key
        if key not in adata.obsm:
            print(f"obsm 缺少 {key!r}", file=sys.stderr)
            return 1
        mat = np.asarray(adata.obsm[key], dtype=np.float64)
        if mat.shape[1] < 2:
            print(f"obsm[{key!r}] 至少需要 2 列", file=sys.stderr)
            return 1
        Z = mat[:, :2].copy()
        xl, yl = "PC1", "PC2"
        if key == "X_state_pca_merged" and "merged_pca_var_ratio" in adata.uns:
            ev = np.asarray(adata.uns["merged_pca_var_ratio"])
            xl = _pc_axis_label(1, ev)
            yl = _pc_axis_label(2, ev)
    else:
        if args.embed_key not in adata.obsm:
            print(f"obsm 缺少 {args.embed_key!r}（或改用 --use-pca-key）", file=sys.stderr)
            return 1
        Xs = np.asarray(adata.obsm[args.embed_key], dtype=np.float64)
        n_comp = min(2, Xs.shape[1], max(1, Xs.shape[0] - 1))
        pca = PCA(n_components=n_comp, random_state=args.random_state)
        Z = pca.fit_transform(Xs)
        if Z.shape[1] < 2:
            z2 = np.zeros((Z.shape[0], 2))
            z2[:, : Z.shape[1]] = Z
            Z = z2
        ev = pca.explained_variance_ratio_
        xl = _pc_axis_label(1, ev)
        yl = _pc_axis_label(2, ev)

    pad_x = 0.05 * (float(np.ptp(Z[:, 0])) or 1.0)
    pad_y = 0.05 * (float(np.ptp(Z[:, 1])) or 1.0)
    xlim0 = (float(Z[:, 0].min() - pad_x), float(Z[:, 0].max() + pad_x))
    ylim0 = (float(Z[:, 1].min() - pad_y), float(Z[:, 1].max() + pad_y))
    xlim, ylim = _square_xy_limits(xlim0, ylim0)

    mg_order = _merge_group_order(adata.obs[args.merge_group_key])

    if not args.only_merge_group_splits:
        p_group = out_dir / f"{prefix}_SE_PCA_PC12_by_merge_group.svg"
        p_batch = out_dir / f"{prefix}_SE_PCA_PC12_by_Batch.svg"

        _plot_pca_panel(
            Z,
            adata.obs[args.merge_group_key],
            f"{title_base} (by file group)",
            xl,
            yl,
            p_group,
            xlim,
            ylim,
            "merge_group",
            palette=MERGE_GROUP_COLORS,
            hue_order=mg_order,
        )
        print(f"已写 {p_group}", flush=True)

        _plot_pca_panel(
            Z,
            adata.obs[args.batch_key],
            f"{title_base} (by Batch)",
            xl,
            yl,
            p_batch,
            xlim,
            ylim,
            args.batch_key,
        )
        print(f"已写 {p_batch}", flush=True)

    if args.only_merge_group_splits or not args.no_per_group:
        tab = sns.color_palette("tab10", n_colors=max(10, len(mg_order)))
        for i, g in enumerate(mg_order):
            col = MERGE_GROUP_COLORS.get(g)
            if col is None:
                col = mcolors.to_hex(tab[i % len(tab)])
            sub = out_dir / f"{prefix}_SE_PCA_PC12_merge_group_only_{g}.svg"
            _plot_pca_single_merge_group(
                Z,
                adata.obs[args.merge_group_key],
                g,
                col,
                f"{title_base} · merge_group = {g}",
                xl,
                yl,
                sub,
                xlim,
                ylim,
            )
            print(f"已写 {sub}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

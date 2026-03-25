#!/usr/bin/env python3
"""
SE embedding pipeline: 清洗 h5ad → `python -m state emb transform` →
HVG sklearn PCA（图 01）→ X_state PCA（图 02）→ harmonypy Harmony（图 03）。
图默认保存为 .svg；散点层在 SVG 内栅格化（嵌入位图），图例与坐标轴仍为矢量。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import sparse
from sklearn.decomposition import PCA


def _default_state_root() -> Path:
    return Path(__file__).resolve().parent / "state-main"


def _batch_slug(batch_key: str) -> str:
    return batch_key.replace(" ", "_")


def _pc_axis_label(k: int, var_ratio: np.ndarray) -> str:
    if k - 1 < len(var_ratio):
        pct = 100.0 * float(var_ratio[k - 1])
        return f"PC{k} {pct:.1f}%"
    return f"PC{k}"


def _sanitize_counts(adata: ad.AnnData, min_genes: int, min_cells: int) -> None:
    X = adata.X
    if sparse.issparse(X):
        X = X.tocsr()
        X.data = np.nan_to_num(X.data, nan=0.0, posinf=0.0, neginf=0.0)
        X.data = np.maximum(X.data, 0.0)
        adata.X = X
    else:
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = np.maximum(X, 0.0)
        adata.X = X

    if min_genes > 0:
        sc.pp.filter_cells(adata, min_genes=min_genes)
    if min_cells > 0:
        sc.pp.filter_genes(adata, min_cells=min_cells)


def _run_state_emb_transform(
    *,
    python_exe: str,
    state_src_root: Path,
    checkpoint: str | None,
    model_folder: str | None,
    protein_embeddings: str | None,
    embed_batch_size: int | None,
    input_h5ad: str,
    output_h5ad: str,
    embed_key: str,
) -> None:
    src = state_src_root / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"state 源码目录不存在: {src}")

    env = os.environ.copy()
    pp = str(src)
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = pp + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = pp

    cmd: list[str] = [
        python_exe,
        "-m",
        "state",
        "emb",
        "transform",
        "--input",
        input_h5ad,
        "--output",
        output_h5ad,
        "--embed-key",
        embed_key,
    ]
    if checkpoint:
        cmd.extend(["--checkpoint", checkpoint])
    if model_folder:
        cmd.extend(["--model-folder", model_folder])
    if protein_embeddings:
        cmd.extend(["--protein-embeddings", protein_embeddings])
    if embed_batch_size is not None:
        cmd.extend(["--batch-size", str(embed_batch_size)])

    if not checkpoint and not model_folder:
        raise ValueError("必须提供 --checkpoint 或 --model-folder（之一或两者）用于 emb transform")

    print("Running:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(state_src_root), env=env)
    if r.returncode != 0:
        raise RuntimeError(f"state emb transform 失败，退出码 {r.returncode}")


def _plot_scatter(
    x: np.ndarray,
    y: np.ndarray,
    labels: pd.Series | np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    legend_title: str = "batch",
) -> None:
    hue = pd.Series(labels).astype(str)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=x, y=y, hue=hue, s=6, linewidth=0, ax=ax, legend="full")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1), title=legend_title, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = out_path.suffix.lower().removeprefix(".") or "png"
    # SVG/PDF 中散点量极大时文件臃肿；栅格化散点层，图例与坐标轴仍为矢量
    if fmt in ("svg", "pdf", "eps"):
        for coll in ax.collections:
            coll.set_rasterized(True)
    save_kw: dict = {"bbox_inches": "tight", "dpi": 150}
    fig.savefig(out_path, format=fmt, **save_kw)
    plt.close(fig)


def _hvg_sklearn_pca(
    adata: ad.AnnData, n_top_genes: int, random_state: int
) -> tuple[np.ndarray, PCA]:
    a = adata.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(
        a, n_top_genes=min(n_top_genes, a.n_vars), flavor="seurat", subset=False
    )
    mask = a.var["highly_variable"].to_numpy()
    if not np.any(mask):
        raise RuntimeError("未选中任何高变基因，请检查数据或降低 n_top_genes")
    X = a[:, mask].X
    if sparse.issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)
    n_comp = min(2, X.shape[1], max(1, X.shape[0] - 1))
    pca = PCA(n_components=n_comp, random_state=random_state)
    z = pca.fit_transform(X)
    if z.shape[1] < 2:
        z2 = np.zeros((z.shape[0], 2))
        z2[:, : z.shape[1]] = z
        z = z2
    return z, pca


def _state_pca_and_harmony(
    adata: ad.AnnData,
    batch_key: str,
    harmony_n_pcs: int,
    random_state: int,
):
    try:
        import harmonypy as hm
    except ImportError as e:
        raise ImportError("请安装 harmonypy: pip install harmonypy") from e

    if batch_key not in adata.obs.columns:
        raise KeyError(f"obs 中不存在 batch 列: {batch_key!r}")

    Xs = np.asarray(adata.obsm["X_state"], dtype=np.float64)
    n_comp = min(
        harmony_n_pcs,
        Xs.shape[1],
        max(1, Xs.shape[0] - 1),
    )
    pca = PCA(n_components=n_comp, random_state=random_state)
    Z = pca.fit_transform(Xs)

    ho = hm.run_harmony(
        Z.T,
        adata.obs,
        vars_use=[batch_key],
        max_iter_harmony=20,
    )
    Zc = np.asarray(ho.Z_corr, dtype=np.float64)
    n_cells = Z.shape[0]
    # harmonypy 版本间 Z_corr 可能是 (cells, pcs) 或 (pcs, cells)
    if Zc.shape[0] == n_cells:
        Z_h = Zc
    elif Zc.shape[1] == n_cells:
        Z_h = Zc.T
    else:
        raise ValueError(
            f"Harmony Z_corr 形状 {Zc.shape} 与细胞数 {n_cells} 不一致"
        )
    return pca, Z, Z_h


def _unified_limits(a: np.ndarray, b: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    x = np.concatenate([a[:, 0], b[:, 0]])
    y = np.concatenate([a[:, 1], b[:, 1]])
    xr = float(np.ptp(x)) or 1.0
    yr = float(np.ptp(y)) or 1.0
    pad_x = 0.05 * xr
    pad_y = 0.05 * yr
    return (float(x.min() - pad_x), float(x.max() + pad_x)), (float(y.min() - pad_y), float(y.max() + pad_y))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SE emb + HVG PCA + X_state PCA + Harmony + 保存 h5ad/图")
    p.add_argument("--input", required=True, help="输入 .h5ad")
    p.add_argument("--output-dir", required=True, help="输出目录")
    p.add_argument("--state-root", default=None, help="state-main 根目录（默认：与本脚本同级的 state-main）")
    p.add_argument("--python", default=sys.executable, help="用于运行 `python -m state` 的解释器")
    p.add_argument("--checkpoint", default=None, help="SE 模型 .ckpt 路径")
    p.add_argument("--model-folder", default=None, help="含 .ckpt（及可选 protein_embeddings.pt）的目录")
    p.add_argument("--protein-embeddings", default=None, help="覆盖 protein_embeddings.pt")
    p.add_argument("--embed-batch-size", type=int, default=None, help="传给 emb transform 的 batch size")
    p.add_argument("--embed-key", default="X_state", help="obsm 中嵌入矩阵的键名")
    p.add_argument("--num-hvgs", type=int, default=2048, help="HVG 数量（图 01）")
    p.add_argument("--batch-key", default="Batch", help="批次列名（Harmony 与着色）")
    p.add_argument("--harmony-n-pcs", type=int, default=50, help="Harmony 使用的 PCA 维数（基于 X_state）")
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--min-genes", type=int, default=200, help="filter_cells min_genes，0 关闭")
    p.add_argument("--min-cells", type=int, default=3, help="filter_genes min_cells，0 关闭")
    p.add_argument(
        "--plot-01-source",
        choices=("hvg", "raw_obsm"),
        default="hvg",
        help="图 01：hvg 或 raw_obsm",
    )
    p.add_argument("--raw-obsm-key", default="X_pca", help="raw_obsm 时使用的 obsm 键")
    p.add_argument(
        "--unify-emb-pca-harmony-axes",
        action="store_true",
        help="图 02 与 03 共用同一 x/y 轴范围",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    state_root = Path(args.state_root).resolve() if args.state_root else _default_state_root()
    if not state_root.is_dir():
        print(f"错误: state-root 不存在: {state_root}", file=sys.stderr)
        return 1

    inp = Path(args.input).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base = inp.stem
    bk_slug = _batch_slug(args.batch_key)

    print(f"读取 {inp} ...", flush=True)
    adata = ad.read_h5ad(str(inp))
    _sanitize_counts(adata, min_genes=args.min_genes, min_cells=args.min_cells)

    with tempfile.TemporaryDirectory(prefix="se_emb_") as td:
        prep_path = Path(td) / f"{base}_prep_emb.h5ad"
        emb_tmp = Path(td) / f"{base}_emb_out.h5ad"
        adata.write_h5ad(prep_path)

        _run_state_emb_transform(
            python_exe=args.python,
            state_src_root=state_root,
            checkpoint=args.checkpoint,
            model_folder=args.model_folder,
            protein_embeddings=args.protein_embeddings,
            embed_batch_size=args.embed_batch_size,
            input_h5ad=str(prep_path),
            output_h5ad=str(emb_tmp),
            embed_key=args.embed_key,
        )

        adata = ad.read_h5ad(str(emb_tmp))

    if args.embed_key not in adata.obsm:
        raise KeyError(f"嵌入输出缺少 obsm[{args.embed_key!r}]")

    after_se_path = out_dir / f"{base}_after_se.h5ad"
    adata.write_h5ad(after_se_path)
    print(f"已写 {after_se_path}", flush=True)

    if args.plot_01_source == "hvg":
        z01, pca01 = _hvg_sklearn_pca(adata, n_top_genes=args.num_hvgs, random_state=args.random_state)
        ev01 = pca01.explained_variance_ratio_
        xl01 = _pc_axis_label(1, ev01)
        yl01 = _pc_axis_label(2, ev01)
    else:
        key = args.raw_obsm_key
        if key not in adata.obsm:
            raise KeyError(f"--plot-01-source raw_obsm 需要 obsm[{key!r}]")
        mat = np.asarray(adata.obsm[key])
        if mat.shape[1] < 2:
            raise ValueError(f"obsm[{key!r}] 至少需要 2 列")
        z01 = mat[:, :2].copy()
        xl01, yl01 = "PC1", "PC2"

    p01 = out_dir / f"{base}_01_hvgPCA_PC12_{bk_slug}.svg"
    _plot_scatter(
        z01[:, 0],
        z01[:, 1],
        adata.obs[args.batch_key],
        title=f"{base} 01 HVG PCA",
        xlabel=xl01,
        ylabel=yl01,
        out_path=p01,
        legend_title=args.batch_key,
    )
    print(f"已写 {p01}", flush=True)

    pca_s, Z, Z_h = _state_pca_and_harmony(
        adata,
        batch_key=args.batch_key,
        harmony_n_pcs=args.harmony_n_pcs,
        random_state=args.random_state,
    )
    ev_s = pca_s.explained_variance_ratio_

    xlim = ylim = None
    if args.unify_emb_pca_harmony_axes:
        xlim, ylim = _unified_limits(Z[:, :2], Z_h[:, :2])

    p02 = out_dir / f"{base}_02_state_PC12_{bk_slug}.svg"
    _plot_scatter(
        Z[:, 0],
        Z[:, 1],
        adata.obs[args.batch_key],
        title=f"{base} 02 X_state PCA",
        xlabel=_pc_axis_label(1, ev_s),
        ylabel=_pc_axis_label(2, ev_s),
        out_path=p02,
        xlim=xlim,
        ylim=ylim,
        legend_title=args.batch_key,
    )
    print(f"已写 {p02}", flush=True)

    p03 = out_dir / f"{base}_03_harmony_PC12_{bk_slug}.svg"
    _plot_scatter(
        Z_h[:, 0],
        Z_h[:, 1],
        adata.obs[args.batch_key],
        title=f"{base} 03 Harmony (on X_state PCA)",
        xlabel=_pc_axis_label(1, ev_s),
        ylabel=_pc_axis_label(2, ev_s),
        out_path=p03,
        xlim=xlim,
        ylim=ylim,
        legend_title=args.batch_key,
    )
    print(f"已写 {p03}", flush=True)

    adata.obsm["X_state_pca"] = Z
    adata.obsm["X_state_harmony"] = Z_h
    state_emb_path = out_dir / f"{base}_state_emb.h5ad"
    adata.write_h5ad(state_emb_path)
    print(f"已写 {state_emb_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

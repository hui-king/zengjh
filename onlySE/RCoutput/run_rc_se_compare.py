#!/usr/bin/env python3

import argparse
import json
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "RC.h5ad"
DEFAULT_OUTDIR = ROOT / "data" / "RCoutput"
CHECKPOINT = ROOT / "se600m_epoch4.ckpt"
PROTEIN_PT = ROOT / "protein_embeddings.pt"
RNG = 42
EMB_KEY = "X_state"
NORM_TARGET_SUM = 1e4


def load_checkpoint_gene_vocab(protein_embeds_path: Path) -> tuple[set[str], dict[str, str]]:
    raw = torch.load(protein_embeds_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict at {protein_embeds_path}, got {type(raw)}")
    ordered = [str(k) for k in raw.keys()]
    exact = set(ordered)
    lower_to_canon: dict[str, str] = {}
    for gene in ordered:
        lower_to_canon.setdefault(gene.lower(), gene)
    return exact, lower_to_canon


def resolve_gene_to_checkpoint(name: str, exact: set[str], lower_to_canon: dict[str, str]) -> str | None:
    gene = str(name).strip()
    if not gene:
        return None
    if gene in exact:
        return gene
    return lower_to_canon.get(gene.lower())


def align_adata_genes_to_checkpoint(adata: sc.AnnData, protein_embeds_path: Path) -> sc.AnnData:
    exact, lower_to_canon = load_checkpoint_gene_vocab(protein_embeds_path)
    names_index = [str(x).strip() for x in adata.var_names]
    names_alt = (
        [str(x).strip() for x in adata.var["gene_name"].values]
        if "gene_name" in adata.var.columns
        else list(names_index)
    )

    pairs: list[tuple[int, str]] = []
    for i in range(adata.n_vars):
        candidate = resolve_gene_to_checkpoint(names_index[i], exact, lower_to_canon)
        if candidate is None:
            candidate = resolve_gene_to_checkpoint(names_alt[i], exact, lower_to_canon)
        if candidate is not None:
            pairs.append((i, candidate))

    seen: set[str] = set()
    sel_cols: list[int] = []
    canon: list[str] = []
    for col_i, gene in pairs:
        if gene in seen:
            continue
        seen.add(gene)
        sel_cols.append(col_i)
        canon.append(gene)

    if not sel_cols:
        raise RuntimeError("No genes overlap between AnnData and protein_embeddings.pt")

    out = adata[:, sel_cols].copy()
    out.var_names = canon
    out.var["gene_name"] = canon
    return out


def matrix_min_max(X) -> tuple[float, float]:
    if sp.issparse(X):
        if X.nnz == 0:
            return 0.0, 0.0
        data = X.data
        lo, hi = float(data.min()), float(data.max())
        if lo > 0.0:
            lo = 0.0
        return lo, hi
    arr = np.asarray(X)
    return float(arr.min()), float(arr.max())


def clip_nonnegative(adata: sc.AnnData) -> None:
    if sp.issparse(adata.X):
        X = adata.X.tocsr().copy()
        X.data = np.maximum(X.data.astype(np.float64, copy=False), 0.0)
        adata.X = X
    else:
        adata.X = np.maximum(np.asarray(adata.X, dtype=np.float64), 0.0)


def normalize_and_log1p(adata: sc.AnnData) -> None:
    mn, mx = matrix_min_max(adata.X)
    if mn >= -1e-6 and mx <= 40.0:
        return
    clip_nonnegative(adata)
    sc.pp.normalize_total(adata, target_sum=NORM_TARGET_SUM)
    sc.pp.log1p(adata)


def select_hvg(adata: sc.AnnData, n_top: int) -> sc.AnnData:
    batch_key = "Batch" if "Batch" in adata.obs.columns else None
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=min(int(n_top), adata.n_vars),
        flavor="seurat",
        subset=False,
        batch_key=batch_key,
    )
    return adata[:, adata.var["highly_variable"]].copy()


def compute_metrics(X: np.ndarray, labels) -> dict:
    y = labels.astype("category")
    truth = y.cat.codes.to_numpy(dtype=int)
    if (truth < 0).any():
        raise ValueError("Label column contains missing values")
    k = len(y.cat.categories)
    pred = KMeans(n_clusters=k, random_state=RNG, n_init=20).fit_predict(X)
    return {
        "n_clusters": k,
        "ARI": float(adjusted_rand_score(truth, pred)),
        "NMI": float(normalized_mutual_info_score(truth, pred)),
    }


def build_palette(labels) -> dict[str, str]:
    cats = list(labels.astype("category").cat.categories)
    cmap = plt.get_cmap("tab10", max(len(cats), 1))
    palette = {str(cat): matplotlib.colors.to_hex(cmap(i % 10)) for i, cat in enumerate(cats)}
    if "T cells" in palette:
        palette["T cells"] = "#ff0000"
    return palette


def plot_embedding_pca(X: np.ndarray, labels, path: Path, title: str, palette: dict[str, str]) -> None:
    z = PCA(n_components=2, random_state=RNG).fit_transform(X)
    lab = labels.astype(str).to_numpy()
    cats = list(labels.astype("category").cat.categories)
    fig, ax = plt.subplots(figsize=(7, 6))
    for cat in cats:
        mask = lab == str(cat)
        ax.scatter(
            z[mask, 0],
            z[mask, 1],
            s=5,
            alpha=0.7,
            label=str(cat),
            color=palette[str(cat)],
            rasterized=True,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(markerscale=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_metric_compare(pre_metrics: dict, post_metrics: dict, path: Path) -> None:
    metric_names = ["ARI", "NMI"]
    pre_vals = [pre_metrics[m] for m in metric_names]
    post_vals = [post_metrics[m] for m in metric_names]
    x = np.arange(len(metric_names))
    w = 0.34
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars1 = ax.bar(x - w / 2, pre_vals, w, label="Before SE", color="#4c72b0")
    bars2 = ax.bar(x + w / 2, post_vals, w, label="After SE", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, max(pre_vals + post_vals + [1.0]) * 1.12)
    ax.set_ylabel("Score")
    ax.set_title("ARI / NMI Comparison")
    ax.legend()
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.015,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--label-column", default="celltype")
    parser.add_argument("--n-top-genes", type=int, default=2048)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--protein-embeddings", type=Path, default=PROTEIN_PT)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    sample_name = args.input.stem

    args.outdir.mkdir(parents=True, exist_ok=True)
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.protein_embeddings.is_file():
        raise FileNotFoundError(args.protein_embeddings)

    adata = sc.read_h5ad(args.input)
    if adata.raw is not None:
        adata = adata.raw.to_adata()
    if args.label_column not in adata.obs:
        raise KeyError(args.label_column)

    adata = align_adata_genes_to_checkpoint(adata, args.protein_embeddings)
    normalize_and_log1p(adata)
    adata_hvg = select_hvg(adata, args.n_top_genes)
    labels = adata_hvg.obs[args.label_column]
    palette = build_palette(labels)

    adata_pre = adata_hvg.copy()
    sc.pp.scale(adata_pre, max_value=10)
    sc.tl.pca(adata_pre, svd_solver="auto")
    X_pre = np.asarray(adata_pre.obsm["X_pca"], dtype=np.float32)
    pre_metrics = compute_metrics(X_pre, labels)

    sys.path.insert(0, str(ROOT / "state-main" / "src"))
    from state.emb.inference import Inference

    protein_embeds = torch.load(args.protein_embeddings, map_location="cpu", weights_only=False)
    infer = Inference(cfg=None, protein_embeds=protein_embeds)
    infer.load_model(str(args.checkpoint))
    batch_size = args.batch_size or getattr(infer.model.cfg.model, "batch_size", 32)

    with tempfile.TemporaryDirectory(prefix="rcoutput_se_") as tmpdir:
        tmp_input = Path(tmpdir) / "se_input.h5ad"
        adata_hvg.write_h5ad(tmp_input)
        X_state = infer.encode_adata(
            input_adata_path=str(tmp_input),
            output_adata_path=None,
            emb_key=EMB_KEY,
            batch_size=batch_size,
            return_embeddings_only=True,
            concat_dataset_embed=False,
        )

    X_state = np.asarray(X_state, dtype=np.float32)
    post_metrics = compute_metrics(X_state, labels)

    plot_embedding_pca(
        X_pre,
        labels,
        args.outdir / f"{sample_name}_preSE_pca.svg",
        "PCA Before SE",
        palette,
    )
    plot_embedding_pca(
        X_state,
        labels,
        args.outdir / f"{sample_name}_postSE_pca.svg",
        "PCA After SE",
        palette,
    )
    plot_metric_compare(pre_metrics, post_metrics, args.outdir / f"{sample_name}_ari_nmi_compare.svg")

    results = {
        "input": str(args.input),
        "label_column": args.label_column,
        "pre_se": {
            **pre_metrics,
            "embedding": "X_pca",
            "shape": list(X_pre.shape),
        },
        "post_se": {
            **post_metrics,
            "embedding": EMB_KEY,
            "shape": list(X_state.shape),
        },
        "outputs": {
            "pre_pca_svg": str(args.outdir / f"{sample_name}_preSE_pca.svg"),
            "post_pca_svg": str(args.outdir / f"{sample_name}_postSE_pca.svg"),
            "compare_svg": str(args.outdir / f"{sample_name}_ari_nmi_compare.svg"),
        },
    }
    (args.outdir / f"{sample_name}_compare_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

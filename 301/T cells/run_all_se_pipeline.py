#!/usr/bin/env python3
"""
放在「含 .h5ad 的批次目录」内（可与子目录并列），递归运行上级目录中的 run_se_pca.py。

跳过路径中包含 *_ouput 的文件。每个 h5ad 输出到：本目录下 `{basename}_ouput/`。

环境变量（可选）:
  PYTHON              — Python 解释器
  STATE_CHECKPOINT    — 传给 --checkpoint
  STATE_MODEL_FOLDER  — 传给 --model-folder
  STATE_PROTEIN_EMB   — 传给 --protein-embeddings
  STATE_EMB_BATCH_SIZE — 传给 --embed-batch-size
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 若本脚本放在 301/ 根目录（与 run_se_pca.py 同级），则 DIR_301=_SCRIPT_DIR；
# 若放在 301/<细胞>/ 子目录，则 DIR_301 为上级 301。
if os.path.isfile(os.path.join(_SCRIPT_DIR, "run_se_pca.py")):
    DIR_301 = _SCRIPT_DIR
    CELL_ROOT = _SCRIPT_DIR
else:
    DIR_301 = os.path.dirname(_SCRIPT_DIR)
    CELL_ROOT = _SCRIPT_DIR

RUN_SE_PCA = os.path.join(DIR_301, "run_se_pca.py")
STATE_MAIN = os.path.join(DIR_301, "state-main")

_DEFAULT_CELL = "/data2/zengjh/conda/envs/cell/bin/python"


def _python() -> str:
    env = os.environ.get("PYTHON", "").strip()
    if env and os.path.isfile(env):
        return env
    if os.path.isfile(_DEFAULT_CELL):
        return _DEFAULT_CELL
    return sys.executable


def _list_h5ad() -> list[str]:
    paths = sorted(glob.glob(os.path.join(CELL_ROOT, "**", "*.h5ad"), recursive=True))
    out: list[str] = []
    for p in paths:
        norm = os.path.normpath(p)
        parts = norm.split(os.sep)
        if any(part.endswith("_ouput") for part in parts):
            continue
        if any(part == "state-main" for part in parts):
            continue
        out.append(norm)
    return out


def _extra_args() -> list[str]:
    args: list[str] = []
    ck = os.environ.get("STATE_CHECKPOINT", "").strip()
    mf = os.environ.get("STATE_MODEL_FOLDER", "").strip()
    pe = os.environ.get("STATE_PROTEIN_EMB", "").strip()
    bs = os.environ.get("STATE_EMB_BATCH_SIZE", "").strip()
    if ck:
        args.extend(["--checkpoint", ck])
    if mf:
        args.extend(["--model-folder", mf])
    if pe:
        args.extend(["--protein-embeddings", pe])
    if bs:
        args.extend(["--embed-batch-size", bs])
    if not ck and not mf:
        print(
            "警告: 未设置 STATE_CHECKPOINT 或 STATE_MODEL_FOLDER，"
            "run_se_pca.py 将报错。",
            file=sys.stderr,
        )
    return args


def _verify_outputs(base: str, out_dir: str, batch_key: str) -> list[str]:
    bk = batch_key.replace(" ", "_")
    missing: list[str] = []
    expected = [
        os.path.join(out_dir, "run.log"),
        os.path.join(out_dir, f"{base}_after_se.h5ad"),
        os.path.join(out_dir, f"{base}_state_emb.h5ad"),
    ]
    for f in expected:
        if not os.path.isfile(f):
            missing.append(f)
    for idx in ("01", "02", "03"):
        pat = os.path.join(out_dir, f"{base}_{idx}_*_{bk}.png")
        if not glob.glob(pat):
            missing.append(pat)
    return missing


def main() -> int:
    if not os.path.isfile(RUN_SE_PCA):
        print(f"找不到 run_se_pca.py: {RUN_SE_PCA}", file=sys.stderr)
        return 1
    if not os.path.isdir(STATE_MAIN):
        print(f"找不到 state-main: {STATE_MAIN}", file=sys.stderr)
        return 1

    h5ads = _list_h5ad()
    if not h5ads:
        print(f"在 {CELL_ROOT!r} 下未发现 .h5ad", file=sys.stderr)
        return 1

    py = _python()
    extra = _extra_args()
    print(f"Python: {py}")
    print(f"run_se_pca: {RUN_SE_PCA}")
    print(f"state-main: {STATE_MAIN}")
    print(f"发现 {len(h5ads)} 个 h5ad\n")

    failed = 0
    for h5ad in h5ads:
        base = os.path.splitext(os.path.basename(h5ad))[0]
        out_dir = os.path.join(CELL_ROOT, f"{base}_ouput")
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, "run.log")

        cmd = [
            py,
            os.path.abspath(RUN_SE_PCA),
            "--state-root",
            os.path.abspath(STATE_MAIN),
            "--input",
            os.path.abspath(h5ad),
            "--output-dir",
            os.path.abspath(out_dir),
            *extra,
        ]

        header = (
            f"=== {datetime.now().isoformat()} ===\n"
            f"input: {h5ad}\noutput_dir: {out_dir}\n"
            f"command: {' '.join(cmd)}\n\n"
        )
        print(f"--- {base} -> {out_dir}")
        with open(log_path, "w", encoding="utf-8") as logf:
            logf.write(header)
            logf.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=DIR_301,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                logf.write(line)
                logf.flush()
                print(line, end="")
            rcode = proc.wait()
        if rcode != 0:
            print(f"失败 退出码 {rcode}，见 {log_path}", file=sys.stderr)
            failed += 1
            continue

        miss = _verify_outputs(base, out_dir, batch_key="Batch")
        if miss:
            print("警告: 缺少预期输出:", file=sys.stderr)
            for m in miss:
                print(f"  - {m}", file=sys.stderr)
            failed += 1
        else:
            print(f"OK: {out_dir}")

    print(f"\n结束。失败数: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

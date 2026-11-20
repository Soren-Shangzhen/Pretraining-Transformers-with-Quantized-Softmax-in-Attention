#!/usr/bin/env python3
"""
Collect training-dynamics CSVs from <project> runs into one long-form CSV.

One output row = one (run, step, tokens) evaluation/logging point.

Merged sources when present:
  - log.csv
  - attention_stats.csv
  - occupancy.csv

The script is intentionally lossless:
  * it keeps every numeric column found in the source CSVs;
  * schema differences between historical and current runs are tolerated;
  * rows are OUTER-merged on (step, tokens), so missing diagnostics are not dropped;
  * historical attention_stats without row_center_mean get a derived midpoint
        row_center_mean = 0.5 * (row_min_mean + row_max_mean)
    and this is marked in row_center_source;
  * duplicate metric names from occupancy.csv are prefixed with "occ_";
  * no checkpoint is loaded by default, so scanning does not touch multi-GB weights.

Typical use from the phase-B repo:
    python collect_training_dynamics.py \
      --runs-root runs \
      --runs-root <REPO>/training_code/legacy_exact_fp32/runs \
      --output training_dynamics_all.csv

By default, obvious smoke/debug/presanity/partial/tmp runs are excluded.
Use --include-partial to keep them.

Optional:
    --name-regex '100m'
        Only keep run directory names matching the regex.

    --with-checkpoint-metadata
        Load ckpt.pt one run at a time with mmap=True (PyTorch >=2.1) and append
        exact config/status metadata. This can be slower on network/external disks.

Scientific note:
    This collector does NOT compute Delta-NLL relative to Exact. Raw values are
    preserved so later comparisons can enforce matched seed/protocol/token budget.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository



KEYS = ["step", "tokens"]
DEFAULT_EXCLUDE_RE = re.compile(
    r"(smoke|debug|test|presanity|partial|tmp|scratch)",
    re.IGNORECASE,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--runs-root",
        action="append",
        required=True,
        help="Path to a directory containing run subdirectories. Repeat as needed.",
    )
    p.add_argument(
        "--output",
        default="training_dynamics_all.csv",
        help="Output long-form CSV.",
    )
    p.add_argument(
        "--name-regex",
        default=None,
        help="Optional regex applied to run directory basename, e.g. '100m'.",
    )
    p.add_argument(
        "--include-partial",
        action="store_true",
        help="Include names matching smoke/debug/presanity/partial/tmp.",
    )
    p.add_argument(
        "--with-checkpoint-metadata",
        action="store_true",
        help="Read ckpt.pt config/tokens_done using torch.load(..., mmap=True).",
    )
    p.add_argument(
        "--print-only",
        action="store_true",
        help="Only print discovered runs; do not write output.",
    )
    return p.parse_args()


def read_csv_robust(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize key columns when possible.
    for k in KEYS:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce").astype("Int64")
    # Convert all non-key columns to numeric when they are fully/mostly numeric.
    for c in df.columns:
        if c in KEYS:
            continue
        converted = pd.to_numeric(df[c], errors="coerce")
        # If the original has no non-null strings that fail conversion, keep numeric.
        orig_nonnull = df[c].notna().sum()
        conv_nonnull = converted.notna().sum()
        if orig_nonnull == 0 or conv_nonnull >= max(1, int(0.95 * orig_nonnull)):
            df[c] = converted
    return df


def canonicalize_run_name(name: str, source_repo: str) -> Dict[str, object]:
    n = name.lower()

    operator = "unknown"
    K = np.nan
    range_grad = "unknown"
    condition = name
    experiment_role = "other"

    if "exact" in n and "softmax" in n:
        operator = "exact"
        condition = "exact"
        range_grad = "n/a"
        experiment_role = "formal_main" if "100m" in n else "other"
    else:
        if "lerp" in n:
            operator = "lerp"
        elif "nearest" in n:
            operator = "nearest"

        m = re.search(r"(?:^|[_-])q?(\d+)(?:[_-]|$)", n)
        if m:
            K = int(m.group(1))

        if "detachproj" in n or "detach_proj" in n or "detach-proj" in n:
            range_grad = "detach_proj"
            experiment_role = "p1_detach_proj"
        elif "fullgrad" in n or "full_grad" in n:
            range_grad = "full"
            experiment_role = "formal_main"
        elif operator in {"lerp", "nearest"}:
            # Historical code predating --range_grad used detached row min/max.
            range_grad = "detach"
            experiment_role = "historical_detach"

        if operator in {"lerp", "nearest"} and not (isinstance(K, float) and math.isnan(K)):
            suffix = range_grad if range_grad != "unknown" else ""
            condition = f"{operator}_K{int(K)}" + (f"_{suffix}" if suffix else "")

    return {
        "condition": condition,
        "operator": operator,
        "K": K,
        "range_grad_inferred": range_grad,
        "experiment_role": experiment_role,
    }


def merge_sources(run_dir: Path) -> Tuple[pd.DataFrame, Dict[str, bool]]:
    paths = {
        "log": run_dir / "log.csv",
        "stats": run_dir / "attention_stats.csv",
        "occupancy": run_dir / "occupancy.csv",
    }
    present = {k: p.exists() for k, p in paths.items()}

    frames: List[pd.DataFrame] = []

    if present["log"]:
        log = read_csv_robust(paths["log"])
        frames.append(log)

    if present["stats"]:
        stats = read_csv_robust(paths["stats"])
        # Ensure older historical schema has center mean available.
        if (
            "row_center_mean" not in stats.columns
            and "row_min_mean" in stats.columns
            and "row_max_mean" in stats.columns
        ):
            stats["row_center_mean"] = 0.5 * (
                pd.to_numeric(stats["row_min_mean"], errors="coerce")
                + pd.to_numeric(stats["row_max_mean"], errors="coerce")
            )
            stats["row_center_source"] = "derived_midpoint"
        else:
            stats["row_center_source"] = "logged"
        frames.append(stats)

    if present["occupancy"]:
        occ = read_csv_robust(paths["occupancy"])
        # Keep keys untouched; prefix every other occupancy field so nothing
        # silently overwrites the primary attention_stats values.
        occ = occ.rename(
            columns={c: f"occ_{c}" for c in occ.columns if c not in KEYS}
        )
        frames.append(occ)

    if not frames:
        return pd.DataFrame(), present

    # Start from first source and outer-merge everything on available keys.
    out = frames[0]
    for f in frames[1:]:
        common_keys = [k for k in KEYS if k in out.columns and k in f.columns]
        if len(common_keys) == 2:
            out = out.merge(f, on=KEYS, how="outer", validate="one_to_one")
        elif len(common_keys) == 1:
            out = out.merge(f, on=common_keys, how="outer")
        else:
            raise RuntimeError(
                f"Cannot merge sources in {run_dir}: no shared step/tokens key."
            )

    # Canonical sort.
    sort_cols = [c for c in ["tokens", "step"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    return out, present


def checkpoint_metadata(run_dir: Path) -> Dict[str, object]:
    ckpt = run_dir / "ckpt.pt"
    if not ckpt.exists():
        return {
            "ckpt_present": False,
            "ckpt_tokens_done": np.nan,
            "ckpt_step": np.nan,
        }

    try:
        import torch

        kwargs = dict(map_location="cpu", weights_only=False)
        # mmap=True keeps tensor storages mapped instead of eagerly copying.
        try:
            obj = torch.load(ckpt, mmap=True, **kwargs)
        except TypeError:
            obj = torch.load(ckpt, **kwargs)

        cfg = obj.get("config", {}) if isinstance(obj, dict) else {}
        meta = {
            "ckpt_present": True,
            "ckpt_tokens_done": obj.get("tokens_done", np.nan),
            "ckpt_step": obj.get("step", np.nan),
        }

        config_keys = [
            "seed",
            "attn",
            "attn_mode",
            "K",
            "attn_K",
            "approx",
            "attn_approx",
            "range_grad",
            "attn_range_grad",
            "micro_batch",
            "batch_tokens",
            "max_tokens",
            "warmup_tokens",
            "eval_every_tokens",
            "eval_batches",
            "stats_batches",
            "ckpt_every_tokens",
            "grad_checkpointing",
            "block_size",
            "lr",
            "min_lr_ratio",
            "weight_decay",
            "beta1",
            "beta2",
            "grad_clip",
        ]
        for k in config_keys:
            if k in cfg:
                meta[f"cfg_{k}"] = cfg[k]

        # Historical checkpoints predate range_grad; for quant runs that means detach.
        attn_val = cfg.get("attn", cfg.get("attn_mode", None))
        if attn_val == "quant" and "range_grad" not in cfg and "attn_range_grad" not in cfg:
            meta["cfg_range_grad"] = "detach"

        del obj
        gc.collect()
        return meta

    except Exception as e:
        return {
            "ckpt_present": True,
            "ckpt_read_error": repr(e),
            "ckpt_tokens_done": np.nan,
            "ckpt_step": np.nan,
        }


def discover_runs(
    roots: Iterable[Path],
    name_regex: Optional[str],
    include_partial: bool,
) -> List[Tuple[Path, Path]]:
    rx = re.compile(name_regex, re.IGNORECASE) if name_regex else None
    found: List[Tuple[Path, Path]] = []

    for root in roots:
        root = root.expanduser().resolve()
        if not root.exists():
            print(f"[WARN] missing runs root: {root}", file=sys.stderr)
            continue

        for p in sorted(root.iterdir()):
            if not p.is_dir():
                continue
            if not include_partial and DEFAULT_EXCLUDE_RE.search(p.name):
                continue
            if rx and not rx.search(p.name):
                continue
            if not any((p / fn).exists() for fn in ("log.csv", "attention_stats.csv", "occupancy.csv")):
                continue
            found.append((root, p))

    return found


def main():
    args = parse_args()
    roots = [Path(x) for x in args.runs_root]
    runs = discover_runs(roots, args.name_regex, args.include_partial)

    if not runs:
        raise SystemExit("No candidate runs found.")

    print(f"Discovered {len(runs)} run(s):")
    for root, run in runs:
        print(f"  - {run}")

    if args.print_only:
        return

    all_frames: List[pd.DataFrame] = []
    for root, run_dir in runs:
        df, present = merge_sources(run_dir)
        if df.empty:
            continue

        source_repo = str(root.parent)
        inferred = canonicalize_run_name(run_dir.name, source_repo)

        # Provenance / identifiers first.
        df.insert(0, "run_dir", str(run_dir))
        df.insert(0, "source_runs_root", str(root))
        df.insert(0, "source_repo", source_repo)
        df.insert(0, "run_name", run_dir.name)
        df.insert(0, "experiment_role", inferred["experiment_role"])
        df.insert(0, "range_grad_inferred", inferred["range_grad_inferred"])
        df.insert(0, "K_inferred", inferred["K"])
        df.insert(0, "operator", inferred["operator"])
        df.insert(0, "condition", inferred["condition"])

        df["has_log_csv"] = present["log"]
        df["has_attention_stats_csv"] = present["stats"]
        df["has_occupancy_csv"] = present["occupancy"]

        if args.with_checkpoint_metadata:
            meta = checkpoint_metadata(run_dir)
            for k, v in meta.items():
                df[k] = v

        all_frames.append(df)

    if not all_frames:
        raise SystemExit("No readable CSV data found in discovered runs.")

    out = pd.concat(all_frames, ignore_index=True, sort=False)

    # Put common scientific columns near the front; preserve every extra field after.
    preferred = [
        "condition",
        "operator",
        "K_inferred",
        "range_grad_inferred",
        "experiment_role",
        "run_name",
        "source_repo",
        "source_runs_root",
        "run_dir",
        "step",
        "tokens",
        "train_loss",
        "val_loss",
        "lr",
        "grad_norm",
        "tok_per_s",
        "row_min_mean",
        "row_max_mean",
        "row_center_mean",
        "row_center_source",
        "row_center_std",
        "row_span_mean",
        "row_span_std",
        "q_norm_mean",
        "q_norm_std",
        "k_norm_mean",
        "k_norm_std",
        "q_proj_weight_norm_mean",
        "k_proj_weight_norm_mean",
        "interval_width_mean",
        "has_log_csv",
        "has_attention_stats_csv",
        "has_occupancy_csv",
    ]
    cols = [c for c in preferred if c in out.columns] + [
        c for c in out.columns if c not in preferred
    ]
    out = out[cols]

    # Stable ordering for easy diffing and later plotting.
    sort_cols = [c for c in ["experiment_role", "operator", "K_inferred", "run_name", "tokens", "step"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="stable", na_position="last").reset_index(drop=True)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    # Console audit summary.
    print("\n=== Collection summary ===")
    print(f"rows   : {len(out):,}")
    print(f"runs   : {out['run_name'].nunique():,}")
    print(f"columns: {len(out.columns):,}")
    print(f"output : {output}")

    summary_cols = [c for c in ["run_name", "experiment_role", "operator", "K_inferred", "range_grad_inferred"] if c in out.columns]
    audit = (
        out.groupby(summary_cols, dropna=False)
        .agg(
            n_rows=("run_name", "size"),
            min_tokens=("tokens", "min") if "tokens" in out.columns else ("run_name", "size"),
            max_tokens=("tokens", "max") if "tokens" in out.columns else ("run_name", "size"),
        )
        .reset_index()
    )
    print("\nPer-run audit:")
    print(audit.to_string(index=False))

    # Basic duplicate-key audit.
    if all(c in out.columns for c in ["run_name", "step", "tokens"]):
        dup = out.duplicated(["run_name", "step", "tokens"], keep=False)
        if dup.any():
            print(
                f"\n[WARN] {int(dup.sum())} rows share duplicate (run_name,step,tokens) keys.",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stage 0: freeze run-level and checkpoint-level manifests for the primary runs (35 canonical + later additions).

Input : analysis_asset_inventory_20260917/canonical_primary_35.tsv  (authoritative classification)
Output: analysis_main_v1/manifests/
          primary_37_runs.tsv         explicit method axes per run (35 canonical + ADDITIONS)
          primary_37_checkpoints.tsv  every saved trajectory checkpoint, with step/tokens/size
          additions_after_20260917_inventory.tsv   runs added after the canonical inventory was frozen
          eval_jobs.tsv               ordered evaluation job list (endpoints first, then trajectories)

No recursive discovery outside the run_path listed in the canonical manifest.
Only numpy-free stdlib so it runs anywhere.
"""
import csv
import os
import re
import shutil
import sys
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository


ROOT = KI_CLOUD
INV = f"{ROOT}/analysis_asset_inventory_20260917"
OUT = f"{ROOT}/analysis_main_v1/manifests"

SUITE = {
    "A_LONG_124M_2P5B": ("A_longhorizon_124m_2p5b", "124M", 2500),
    "B_FACTORIAL_124M_250M": ("B_factorial_124m_250m", "124M", 250),
    "C_SCALE_1B_100M": ("C_scale_1b_100m", "1B", 100),
}
EXPECTED = {"A_LONG_124M_2P5B": (15, 20), "B_FACTORIAL_124M_250M": (10, 10), "C_SCALE_1B_100M": (12, 5)}

# Runs completed after the 2026-09-17 canonical inventory was frozen. They complete the 1B K4 2x2
# (calibration x backward) on WikiText-103 and were verified checkpoint-by-checkpoint against the 1B protocol
# (32L/24H/1536, seed 1337, 100M tokens, warmup 10M, lr 3e-4, batch_tokens 131072, range_grad=full).
ADDITIONS = [
    dict(analysis_group="C_SCALE_1B_100M", category="PRIMARY_1B_100M",
         root="formal_runs_rerun_wikitext_1b_probste_20260917",
         run_name="formal_1b_s1337_minmax_nearest_k4_probste_100m_wikitext_rerun", seed="1337", scale="1B",
         method="MinMax/Nearest", ste="Prob-STE", K="4", checkpoint_count="5", min_tokens_m="20.1",
         max_tokens_m="100.0", final_checkpoint="ckpt_step000763_100.0m.pt", final_size_bytes="11819179283",
         canonical_reason="2026-09-17 WikiText rerun completing the 1B K4 calibration x backward 2x2",
         dataset_id="wikitext103"),
    dict(analysis_group="C_SCALE_1B_100M", category="PRIMARY_1B_100M",
         root="formal_runs_rerun_wikitext_1b_probste_20260917",
         run_name="formal_qrm_1b_s1337_nearest_k4_tau6_probste_100m_wikitext_rerun", seed="1337", scale="1B",
         method="QRM", ste="Prob-STE", K="4", checkpoint_count="5", min_tokens_m="20.1", max_tokens_m="100.0",
         final_checkpoint="ckpt_step000763_100.0m.pt", final_size_bytes="11819179347",
         canonical_reason="2026-09-17 WikiText rerun completing the 1B K4 calibration x backward 2x2",
         dataset_id="wikitext103"),
]

CODE = {
    "base": f"{ROOT}/formal_code_factorial250_v1/formal_base_v2",
    "lerp_k1": f"{ROOT}/vast_formal_extension_v1/formal_lerp_k1_v1",
    "minmax_probste": f"{ROOT}/formal_code_factorial250_v1/formal_nearest_probste_v1",
    "qrm_weightste": f"{ROOT}/formal_code_factorial250_v1/formal_qrm_v1",
    "qrm_probste": f"{ROOT}/formal_code_factorial250_v1/formal_qrm_probste_v1",
}
CK_RE = re.compile(r"ckpt_step(\d+)_([\d.]+)m\.pt$")


def axes(row):
    method, ste, K = row["method"], row["ste"], row["K"]
    if method == "Exact":
        return dict(label="Exact", calibration="none", reconstruction="exact", backward="native_exact",
                    K="", range_tau="", range_grad="n/a", train_code="base")
    if method == "LERP":
        return dict(label="LERP", calibration="minmax_row_adaptive", reconstruction="lerp",
                    backward="native_differentiable", K=K, range_tau="", range_grad="full",
                    train_code="lerp_k1" if K == "1" else "base")
    if method == "MinMax/Nearest":
        prob = ste == "Prob-STE"
        return dict(label="MinMax-Prob" if prob else "MinMax-Weight", calibration="minmax_row_adaptive",
                    reconstruction="nearest", backward="prob_ste" if prob else "weight_ste", K=K,
                    range_tau="", range_grad="full", train_code="minmax_probste" if prob else "base")
    if method == "QRM":
        prob = ste == "Prob-STE"
        return dict(label="QRM-Prob" if prob else "QRM-Weight", calibration="relevant_zero",
                    reconstruction="nearest", backward="prob_ste" if prob else "weight_ste", K=K,
                    range_tau="6", range_grad="full", train_code="qrm_probste" if prob else "qrm_weightste")
    raise ValueError(f"unknown method {method!r}")


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in ("canonical_primary_35.tsv", "canonical_runs_v2.tsv", "canonical_primary_config_audit.tsv",
              "auxiliary_runs.tsv", "CANONICAL_ASSET_MAP.md"):
        shutil.copy2(f"{INV}/{f}", f"{OUT}/{f}")

    rows = list(csv.DictReader(open(f"{INV}/canonical_primary_35.tsv"), delimiter="\t"))
    for extra in ADDITIONS:
        e = dict(extra)
        e["run_path"] = f"{ROOT}/{e['root']}/{e['run_name']}"
        rows.append(e)
    with open(f"{OUT}/additions_after_20260917_inventory.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[-1].keys()), delimiter="\t")
        w.writeheader()
        w.writerows([dict(r, run_path=f"{ROOT}/{r['root']}/{r['run_name']}") for r in ADDITIONS])
    runs, ckpts, errors = [], [], []
    counts = {g: 0 for g in EXPECTED}

    for r in rows:
        g = r["analysis_group"]
        if g not in SUITE:
            errors.append(f"unexpected group {g} for {r['run_name']}")
            continue
        counts[g] += 1
        suite, scale, horizon = SUITE[g]
        ax = axes(r)
        files = sorted(f for f in os.listdir(r["run_path"]) if CK_RE.match(f))
        if len(files) != EXPECTED[g][1]:
            errors.append(f"{r['run_name']}: {len(files)} checkpoints, expected {EXPECTED[g][1]}")
        if not files or files[-1] != r["final_checkpoint"]:
            errors.append(f"{r['run_name']}: last checkpoint {files[-1] if files else None} != manifest "
                          f"{r['final_checkpoint']}")
        dataset = r["dataset_id"]
        runs.append(dict(suite=suite, analysis_group=g, run_name=r["run_name"], **{k: v for k, v in ax.items()},
                         dataset=dataset, seed=r["seed"], model_scale=scale, token_horizon_m=horizon,
                         n_checkpoints=len(files), final_checkpoint=r["final_checkpoint"],
                         train_code_root=CODE[ax["train_code"]], run_path=r["run_path"]))
        for i, f in enumerate(files):
            m = CK_RE.match(f)
            p = os.path.join(r["run_path"], f)
            ckpts.append(dict(suite=suite, run_name=r["run_name"], label=ax["label"], K=ax["K"],
                              ckpt_index=i, step=int(m.group(1)), tokens_m=float(m.group(2)),
                              is_final=int(f == r["final_checkpoint"]), size_bytes=os.path.getsize(p),
                              dataset=dataset, model_scale=scale, checkpoint_path=p))

    for g, (n, _) in EXPECTED.items():
        if counts[g] != n:
            errors.append(f"group {g}: {counts[g]} runs, expected {n}")
    if errors:
        print("MANIFEST ERRORS:\n  " + "\n  ".join(errors))
        sys.exit(1)

    def dump(name, rows_):
        with open(f"{OUT}/{name}", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(rows_)

    dump("primary_37_runs.tsv", runs)
    dump("primary_37_checkpoints.tsv", ckpts)

    # Job order: all endpoints (B, A, C), then trajectories B, A, C.
    order = {"B_factorial_124m_250m": 0, "A_longhorizon_124m_2p5b": 1, "C_scale_1b_100m": 2}
    finals = sorted((c for c in ckpts if c["is_final"]), key=lambda c: (order[c["suite"]], c["run_name"]))
    traj = sorted((c for c in ckpts if not c["is_final"]),
                  key=lambda c: (order[c["suite"]], c["run_name"], c["ckpt_index"]))
    jobs = [dict(job_id=f"{j:04d}", phase="endpoint" if c["is_final"] else "trajectory", **c)
            for j, c in enumerate(finals + traj)]
    dump("eval_jobs.tsv", jobs)
    for legacy, new in (("primary_35_runs.tsv", "primary_37_runs.tsv"),
                        ("primary_35_checkpoints.tsv", "primary_37_checkpoints.tsv")):
        shutil.copy2(f"{OUT}/{new}", f"{OUT}/{legacy}")   # keep the old filenames working
    print(f"runs={len(runs)} checkpoints={len(ckpts)} jobs={len(jobs)} "
          f"(endpoint={len(finals)}, trajectory={len(traj)})")
    for g in EXPECTED:
        print(f"  {g}: {counts[g]} runs")


if __name__ == "__main__":
    main()

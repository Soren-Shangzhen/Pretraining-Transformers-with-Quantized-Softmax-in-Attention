#!/usr/bin/env python3
"""Independent check of the result tables of Appendix I (the csv files written by rb_dossier.py).  This file does not
import the generator (rb_dossier*.py): it finds the result files itself, resolves every pair label itself, calls the
paper's bootstrap directly, and compares with the csv files at full precision.
Writes CHECK_REPORT.md next to csv/.  usage: rb_dossier_check.py [--pkg DIR]
"""
import argparse, csv, glob, gzip, hashlib, json, os, re, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RB = os.path.abspath(os.path.join(HERE, "..", ".."))
AM = os.path.dirname(RB)
sys.path.insert(0, os.path.join(AM, "downstream_v1", "code"))
import stats as st  # noqa: E402  (the paper's statistics)

EV = os.path.join(RB, "runs", "_eval")
HOST = {"A": "machine-A", "B": "machine-B"}
PAPER = {"A": os.path.join(AM, "eval_native_v1", "A_longhorizon_124m_2p5b"), "B": os.path.join(AM, "eval_native_v1", "B_factorial_124m_250m"),
         "D": os.path.join(AM, "D_robustness_124m_100m_5seed", "eval_native_v1")}
A_RUNS = {"rb-E8a-01", "rb-E8c-01", "rb-E3a-2.5B-01"}
SUITE_A = {"softmax": "formal_124m_s1337_exact_2p5b", "LERP K4": "formal_124m_s1337_lerp_k4_fullgrad_2p5b", "LERP K16": "formal_124m_s1337_lerp_k16_fullgrad_2p5b",
           "FWM-Weight K4": "formal_124m_s1337_qrm_nearest_k4_tau6_fullgrad_2p5b", "FWM-Prob K4": "formal_124m_s1337_qrm_nearest_k4_tau6_probste_2p5b",
           "FWM-Weight K16": "formal_124m_s1337_qrm_nearest_k16_tau6_fullgrad_2p5b"}
problems = []


def rd(p):
    return list(csv.DictReader(open(p)))


def fl(x):
    return None if x in ("", None) else float(x)


def eval_dirs(machine, run):
    out = {}
    for d in glob.glob(os.path.join(EV, "nll", run, "*")):
        if "__fwd" in os.path.basename(d):
            continue
        s = json.load(open(os.path.join(d, "summary.json")))
        if s["host"] != HOST[machine]:
            continue
        out[int(s["step"])] = (np.load(os.path.join(d, "per_block_nll.npy")), s)
    return out


def endpoint_step(run):
    for p in (os.path.join(RB, "manifests", "ckpt_sha256_machine-B.tsv"), os.path.join(RB, "manifests", "ckpt_sha256_machine-A.tsv")):
        for r in csv.DictReader(open(p), delimiter="\t"):
            if r["run"] == run and r["kind"] == "endpoint":
                return int(re.search(r"ckpt_step(\d+)", r["file"]).group(1)), r["sha256"]
    raise KeyError(run)


def run_arr(run, step=None):
    m = "A" if run in A_RUNS else "B"
    ds = eval_dirs(m, run)
    if step is None:
        step = endpoint_step(run)[0]
    return ds[step][0] if step in ds else None


def paper_arr(suite, name, step):
    g = glob.glob(os.path.join(PAPER[suite], name, f"*{step:06d}*", "per_block_nll.npy"))
    return np.load(g[0]) if g else None


def supp_arr(where, name):
    g = glob.glob(os.path.join(AM, "supplement_four_runs_20260923", "eval", where, name, "100m_*", "per_block_nll.npy"))
    return np.load(g[0]) if g else None


def resolve(label, step=None):
    """per-block array (or a float for mean-only pairs) for a run / pair label as written in the dossier."""
    if label.startswith("rb-"):
        return run_arr(label.split(" ")[0], step)
    if label.startswith("suite A") or label.startswith("seed-1337"):
        name = next(k for k in sorted(SUITE_A, key=len, reverse=True) if f" {k} " in f" {label} ")
        st_ = step if step is not None else (4769 if "625M" in label else 19074)
        return paper_arr("A", SUITE_A[name], st_)
    m = re.match(r"suite B FWM-(Weight|Prob) K4", label)
    if m:
        return paper_arr("B", "formal_124m_s1337_qrm_k4_tau6_matched250m" if m.group(1) == "Weight" else "formal_124m_s1337_qrm_k4_tau6_probste_250m", 1908)
    if label.startswith("suite B softmax"):
        return paper_arr("B", "formal_124m_s1337_exact_matched250m", step if step is not None else 1908)
    if label.startswith("suite D softmax s7"):
        return paper_arr("D", "Exact_s7", 763)
    m = re.match(r"suite D LERP K32 s(\d+)", label)
    if m:
        return paper_arr("D", f"LERP_K32_s{m.group(1)}", 763)
    m = re.match(r"App\. F (min|max)only s(\d+)", label)
    if m:
        return supp_arr("rtx3090", f"sp_s{m.group(2)}_lerp_k32_{m.group(1)}only_100m")
    m = re.match(r"D\.5 (BF16-exp|FP8-E4M3 exp) s7", label)
    if m:
        return supp_arr("rtx3090", "sp_s7_softmax_bf16exp_100m" if "BF16" in label else "sp_s7_softmax_fp8e4m3exp_100m")
    if label.startswith("App. C.2 run"):
        return supp_arr("rtx5090", "sp_s7_fwm_nearest_k4_tau6_weightste_strict_100m")
    if label.startswith("App. C.2 strict MinMax-Weight"):
        rv = json.load(open(os.path.join(RB, "checks", "13_2_reeval_machine-B.json")))
        return float(next(r for r in rv if r["label"] == "C2_MMW4S_s7")["got_nll"])
    raise KeyError(label)


def same(a, b, what):
    if a is None and b is None:
        return True
    if a is None or b is None or not (a == b or (abs(a - b) <= 1e-12 * max(1.0, abs(b)))):
        problems.append(f"{what}: dossier {a!r} vs recomputed {b!r}")
        return False
    return True


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pkg", default=os.path.join(RB, "reports", "rb-dossier_20261114")); a = ap.parse_args()
    P = a.pkg; csvd = os.path.join(P, "csv"); counts = {}
    t0 = time.time()
    # 1. run list
    runs = rd(os.path.join(csvd, "runs.csv")); n = 0
    for r in runs:
        arr = run_arr(r["run_id"])
        same(fl(r["endpoint_nll"]), float(arr.mean()), f"runs.csv {r['run_id']} endpoint_nll"); n += 1
        stp, sha = endpoint_step(r["run_id"])
        if r["run_id"] != "rb-E1b-01" and r["endpoint_checkpoint_sha256"] != sha:
            problems.append(f"runs.csv {r['run_id']}: checkpoint SHA-256 differs from the machine's checkpoint list")
        if r["pair"] and r["ci_lo"]:
            b = resolve(r["pair"]); d = st.paired(arr, b, kind="blocks")
            same(fl(r["dnll"]), d["delta"], f"runs.csv {r['run_id']} dnll"); same(fl(r["ci_lo"]), d["ci_lo"], f"runs.csv {r['run_id']} ci_lo"); same(fl(r["ci_hi"]), d["ci_hi"], f"runs.csv {r['run_id']} ci_hi")
        elif r["pair"] and r["dnll"]:
            same(fl(r["dnll"]), float(arr.mean()) - resolve(r["pair"]), f"runs.csv {r['run_id']} dnll (mean only)")
    sm = {l.split(",")[0]: l for l in open(os.path.join(RB, "manifests", "ckpt_sha256_machine-B.tsv"))}  # noqa
    counts["runs.csv rows recomputed"] = n
    # 2. comparisons
    n = 0
    for r in rd(os.path.join(csvd, "comparisons.csv")):
        x = resolve(r["run"]); y = resolve(r["pair"])
        same(fl(r["nll_run"]), float(x.mean()), f"comparisons {r['comparison']} {r['run']} nll_run")
        if isinstance(y, float):
            same(fl(r["dnll"]), float(x.mean()) - y, f"comparisons {r['run']} dnll")
        else:
            d = st.paired(x, y, kind="blocks")
            for k, v in (("dnll", d["delta"]), ("ci_lo", d["ci_lo"]), ("ci_hi", d["ci_hi"])):
                same(fl(r[k]), v, f"comparisons {r['comparison']} {r['run']} {k}")
        n += 1
    counts["comparisons.csv rows recomputed"] = n
    # 3. seed means (t interval)
    n = 0
    for r in rd(os.path.join(csvd, "seed_means.csv")):
        ds = [float(v) for v in r["per_seed_dnll"].split(";")]
        same(fl(r["mean_dnll"]), float(np.mean(ds)), f"seed_means {r['condition']} mean")
        if r["t_lo"]:
            h = 4.302652729911275 * np.std(ds, ddof=1) / np.sqrt(3)
            same(fl(r["t_lo"]), float(np.mean(ds) - h), f"seed_means {r['condition']} t_lo"); same(fl(r["t_hi"]), float(np.mean(ds) + h), f"seed_means {r['condition']} t_hi")
        n += 1
    counts["seed_means.csv rows recomputed"] = n
    # 4. trajectories
    n = 0
    for r in rd(os.path.join(csvd, "trajectories.csv")):
        step = int(r["step"])
        if r["run_id"].startswith("rb-"):
            x = run_arr(r["run_id"], step)
        else:
            name = {"suite A softmax s1337 (paper)": ("A", SUITE_A["softmax"]), "suite A LERP K16 s1337 (paper)": ("A", SUITE_A["LERP K16"]),
                    "suite A MinMax-Weight K4 s1337 (paper)": ("A", "formal_124m_s1337_nearest_k4_fullgrad_2p5b"),
                    "suite B softmax s1337 (paper)": ("B", "formal_124m_s1337_exact_matched250m"),
                    "suite B MinMax-Weight K4 s1337 (paper)": ("B", "formal_124m_s1337_nearest_k4_weightste_250m"),
                    "suite B MinMax-Prob K4 s1337 (paper)": ("B", "formal_124m_s1337_nearest_k4_probste_250m")}[r["run_id"]]
            x = paper_arr(name[0], name[1], step)
        same(fl(r["nll"]), float(x.mean()), f"trajectories {r['run_id']} {step} nll")
        if r["dnll"]:
            y = resolve(r["pair"], step)
            d = st.paired(x, y, kind="blocks")
            for k, v in (("dnll", d["delta"]), ("ci_lo", d["ci_lo"]), ("ci_hi", d["ci_hi"])):
                same(fl(r[k]), v, f"trajectories {r['run_id']} {step} {k}")
        n += 1
    counts["trajectories.csv rows recomputed"] = n
    # 5. probes (layer mean)
    pj = {}
    for p in glob.glob(os.path.join(EV, "probe", "*", "*", "probe.json")):
        d = json.load(open(p)); pj[(d["label"], int(d["step"]))] = d
    n = 0
    for r in rd(os.path.join(csvd, "probes_layer_mean.csv")):
        d = pj[(r["run_id"], int(r["step"]))]
        k = r["metric"]
        if k in d["layer_mean"]:
            v = d["layer_mean"][k]
        elif k == "param_grad_norm_mean":
            v = d["grad"]["param_grad_norm_mean"]
        elif k == "patched_loss_equals_normal_loss":
            v = float(d["grad"]["patched_loss_equals_normal_loss"])
        elif k == "ema_buffer_mean":
            v = float(np.mean([x for L in d["layers"] for kk, x in L.items() if kk.startswith("ema_buffer_head")]))
        elif k == "span_head_mean":
            v = float(np.mean([x for L in d["layers"] for kk, x in L.items() if kk.startswith("span_head")]))
        else:
            problems.append(f"probe metric {k} not found"); continue
        same(fl(r["value"]), float(v), f"probes {r['run_id']} {r['step']} {k}"); n += 1
    counts["probes_layer_mean.csv values recomputed"] = n
    # 6. E6
    n = 0
    for r in rd(os.path.join(csvd, "e6.csv")):
        d = os.path.join(EV, "e6", r["endpoint"], r["setting"])
        x = np.load(os.path.join(d, "per_block_nll.npy")); ref = np.load(os.path.join(EV, "e6", r["endpoint"], "ref", "per_block_nll.npy"))
        same(fl(r["nll_5090"]), float(x.mean()), f"e6 {r['endpoint']} {r['setting']} nll")
        if r["setting"] != "ref":
            dd = st.paired(x, ref, kind="blocks"); same(fl(r["dnll_vs_ref"]), dd["delta"], f"e6 {r['endpoint']} {r['setting']} delta"); same(fl(r["ci_lo"]), dd["ci_lo"], f"e6 {r['endpoint']} {r['setting']} lo")
        l3 = os.path.join(EV, "e6_rtx3090", r["endpoint"], r["setting"], "per_block_nll.npy")
        if os.path.exists(l3):
            same(fl(r["nll_3090"]), float(np.load(l3).mean()), f"e6 {r['endpoint']} {r['setting']} 3090")
        n += 1
    counts["e6.csv rows recomputed"] = n
    # 7. compute
    ev = {}
    for p in (os.path.join(RB, "manifests", "rb-run_log_machine-B.jsonl"),
              os.path.join(RB, "manifests", "rb-run_log_machine-A.jsonl")):
        for l in open(p):
            d = json.loads(l)
            if d.get("run_id") and d["event"] in ("start", "finished"):
                ev.setdefault(d["run_id"], {})[d["event"]] = d["time"]
    n = 0
    for r in rd(os.path.join(csvd, "compute_per_run.csv")):
        same(fl(r["gpu_hours"]), (ev[r["run_id"]]["finished"] - ev[r["run_id"]]["start"]) / 3600, f"compute {r['run_id']}"); n += 1
    counts["compute_per_run.csv rows recomputed"] = n
    n = 0
    for r in rd(os.path.join(csvd, "e2d_endpoints.csv")):
        ids = r["runs"].split(";")
        for col, rid in zip(("ww", "wp", "pp", "pw"), ids):
            same(fl(r[col]), float(run_arr(rid).mean()), f"e2d_endpoints seed {r['seed']} {col}")
        n += 1
    counts["e2d_endpoints.csv rows recomputed"] = n
    # 8. downstream (the paper's own functions, called here directly)
    import rb_paper_stats as PS
    ok, msg = PS.check()
    if not ok:
        problems.append("paper downstream statistics not reproduced: " + msg)
    srcs = {"rb-E1b-01": ("B", "rbE1b01"), "rb-E8a-01": ("B", "rbE8a01"), "rb-E8c-01": ("A", "rbE8c01"), "rb-E3a-2.5B-01": ("A", "rbE3a2.5B01"),
            "LERP K16": ("P", "LERP_K16"), "softmax": ("P", "Exact"), "LERP K4": ("P", "LERP_K4"), "FWM-Prob K4": ("P", "QRMProb_K4")}
    def parse(c):
        a_, b_ = c.split(" − ", 1)
        a_ = a_.split(" ")[0] if a_.startswith("rb-") else a_
        b_ = b_.split(" (")[0].replace(" s1337", "").replace(", seed 1337", "")
        b_ = b_.split(" ")[0] if b_.startswith("rb-") else b_
        return srcs[a_], srcs[b_]
    n = 0
    for r in rd(os.path.join(csvd, "downstream_benchmarks.csv")):
        sa_, sb_ = parse(r["comparison"])
        x = PS.compare(r["config"], r["metric"], sa_, sb_)
        for k, v in (("value_run", x[0]), ("value_pair", x[1]), ("diff", x[2]), ("ci_lo", x[3]), ("ci_hi", x[4])):
            same(fl(r[k]), v, f"benchmarks {r['comparison']} {r['config']} {r['metric']} {k}")
        n += 1
    counts["downstream_benchmarks.csv rows recomputed"] = n
    n = 0
    for r in rd(os.path.join(csvd, "downstream_heldout.csv")):
        a_, b_ = r["comparison"].split(" − ", 1)
        def arr(lab):
            lab = lab.split(" (")[0].replace(" s1337", "").replace(", seed 1337", "").strip()
            if lab.startswith("rb-"):
                run = lab.split(" ")[0]
                return PS.l1_pb("B", run, run, r["corpus"])
            return PS.paper_pb(SUITE_A[lab], r["corpus"])
        x, y = arr(a_), arr(b_)
        cl = PS.clusters(r["corpus"])
        d = st.paired(x, y, kind="clusters" if cl is not None else "blocks", clusters=cl)
        for k, v in (("diff", d["delta"]), ("ci_lo", d["ci_lo"]), ("ci_hi", d["ci_hi"]), ("nll_run", float(x.mean())), ("nll_pair", float(y.mean()))):
            same(fl(r[k]), v, f"heldout {r['comparison']} {r['corpus']} {k}")
        n += 1
    counts["downstream_heldout.csv rows recomputed"] = n
    # report
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    L = ["# Check report of the Appendix I result tables", "", f"Checked {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} in {time.time() - t0:.0f} s by `rb_dossier_check.py` "
         "(independent of the generator's code: own file lookup, own pair resolution, the paper's bootstrap called directly).", "",
         f"**Result: {'PASS — no discrepancy' if not problems else 'FAIL — ' + str(len(problems)) + ' discrepancies'}**", "", "| check | count |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in counts.items()]
    L += [""]
    if problems:
        L += ["## Discrepancies", ""] + [f"- {p}" for p in problems]
    L += ["", "## SHA-256 of the checked files", "", "| file | SHA-256 |", "|---|---|"]
    for f_ in sorted(glob.glob(os.path.join(csvd, "*.csv"))):
        L.append(f"| {os.path.relpath(f_, P)} | {sha(f_)} |")
    open(os.path.join(P, "CHECK_REPORT.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:8 + len(counts)]))
    for p in problems[:30]:
        print("PROBLEM", p)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())

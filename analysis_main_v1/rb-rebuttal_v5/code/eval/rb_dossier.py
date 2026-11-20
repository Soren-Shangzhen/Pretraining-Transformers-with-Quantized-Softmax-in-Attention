#!/usr/bin/env python3
"""Result tables of Appendix I: writes the 20 csv files of reports/rb-dossier_20261114/csv/ (every number at full
precision) from the evaluation results of this package. The rows are computed here and in rb_dossier_tables /
rb_dossier_core; rb_dossier_check.py re-derives the numbers independently and compares them with these files.
usage: rb_dossier.py [--out DIR]   (default reports/rb-dossier_20261114; the files are written to DIR/csv/)
"""
import argparse, csv, glob, json, math, os, re
from collections import OrderedDict
import numpy as np
import rb_dossier_core as C
import rb_dossier_tables as T

OUT = os.path.join(C.RB, "reports", "rb-dossier_20261114")
CHL = os.path.join(C.RB, "checks")


NEUTRAL = [  # wording of recorded decisions: the authors made them (applied to every text cell)
    (re.compile(r"\b(RESOLVED|APPROVED|ACCEPTED|CONFIRMED) by (?:the )?user\b"), r"\1 by the authors"),
    (re.compile(r"\bUser decisions\b"), "Author decisions"), (re.compile(r"\bUser decision\b"), "Author decision"),
    (re.compile(r"\buser decisions\b"), "author decisions"), (re.compile(r"\buser decision\b"), "author decision"),
    (re.compile(r"\b[Uu]ser (rule|request|approval|choice|confirmation)"), r"author \1"), (re.compile(r"\buser-(requested|approved)\b"), r"author-\1"),
    (re.compile(r"\bdecision: user\b"), "decision: authors"), (re.compile(r"\b(?:by the|by) user\b"), "by the authors"),
    (re.compile(r"\bthe user\b"), "the authors"), (re.compile(r"\bThe user\b"), "The authors"), (re.compile(r"\buser confirmed\b"), "authors confirmed"),
    (re.compile(r"\(user\b"), "(authors")]
VERBATIM = ("prediction", "reading_rule", "pending_decisions")   # pre-declared text of the run manifest: never reworded


def neutral(t, keep=()):
    hold = {}
    for i, k in enumerate(sorted({k for k in keep if k}, key=len, reverse=True)):   # verbatim quotes are set aside first
        if k in t:
            tok = f"\x00{i}\x00"; t = t.replace(k, tok); hold[tok] = k
    for a, b in NEUTRAL:
        t = a.sub(b, t)
    for tok, k in hold.items():
        t = t.replace(tok, k)
    return t


# ----------------------------------------------------------------------------- output helper
def write_csv(name, rows):
    os.makedirs(os.path.join(OUT, "csv"), exist_ok=True)
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(os.path.join(OUT, "csv", name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else repr(float(v)) if isinstance(v, (float, np.floating)) else bool(v) if isinstance(v, np.bool_)
                            else int(v) if isinstance(v, np.integer) else neutral(v) if isinstance(v, str) and k not in VERBATIM else v) for k, v in r.items()})
    return f"csv/{name}"


# ----------------------------------------------------------------------------- E2 analysis
def e2_analysis():
    out = []
    W = {t: C.paper("B", "formal_124m_s1337_nearest_k4_weightste_250m", s) for t, s in ((50, 382), (125, 954), (250, 1908))}
    P = {t: C.paper("B", "formal_124m_s1337_nearest_k4_probste_250m", s) for t, s in ((50, 382), (125, 954), (250, 1908))}
    c0b = C.m_end("B", "rb-C0b-01")
    for bp, ids in ((50, ("rb-E2ab-01", "rb-E2ab-02", "rb-E2ab-03", "rb-E2ab-04", "rb-E2ab-05")), (125, ("rb-E2ab-06", "rb-E2ab-07", "rb-E2ab-08", "rb-E2ab-09", "rb-E2ab-10"))):
        ww, wp, wf, pp, pw = (C.m_end("B", x).mean for x in ids)
        g = ww - pp
        fz = C.m_traj("B", ids[2]); base = W[bp].mean
        dense = [s for s in fz if bp * 1e6 < fz[s].tokens <= (bp + 25.5) * 1e6]
        sp = lambda r: C.probe("B", r, max(C.probe("B", r)))["layer_mean"]["span"]
        out.append(OrderedDict(group="E2a/b", branch_point_m=bp, orig_w_250=W[250].mean, orig_p_250=P[250].mean, c0b_250=c0b.mean,
                               ww=ww, wp=wp, freeze=wf, pp=pp, pw=pw, ww_minus_orig_w=ww - W[250].mean, pp_minus_orig_p=pp - P[250].mean,
                               valid=abs(ww - W[250].mean) < 0.01 and abs(pp - P[250].mean) < 0.01,
                               wp_closes_pct=100 * (ww - wp) / g, pw_opens_pct=100 * (pw - pp) / g, freeze_removes_pct=100 * (ww - wf) / g,
                               freeze_minus_pp=wf - pp, span_ww=sp(ids[0]), span_freeze=sp(ids[2]), span_pp=sp(ids[3]), span_wp=sp(ids[1]), span_pw=sp(ids[4]),
                               freeze_first25m_max_rise=max(fz[s].mean - base for s in dense), branch_point_nll=base,
                               ww_peak_nll=None, ww_peak_m=None, orig_peak_nll=None, orig_peak_m=None))
    ot = C.paper_traj("A", T.SUITE_A["MinMax-Weight K4"])
    ot = {s: x for s, x in ot.items() if s <= 4769}
    row = OrderedDict(group="E2c", branch_point_m=250)
    for lab, r in (("ww", "rb-E2c-01"), ("wp", "rb-E2c-02"), ("freeze", "rb-E2c-03")):
        tr = C.m_traj("B", r); pk = max(tr, key=lambda s: tr[s].mean)
        row[f"{lab}_peak_nll"] = tr[pk].mean; row[f"{lab}_peak_m"] = round(tr[pk].tokens / 1e6, 1)
        row[f"{lab}_end_nll"] = tr[max(tr)].mean
        lo = min(tr, key=lambda s: tr[s].mean); row[f"{lab}_min_nll"] = tr[lo].mean; row[f"{lab}_min_m"] = round(tr[lo].tokens / 1e6, 1)
    pk = max(ot, key=lambda s: ot[s].mean)
    row["orig_peak_nll"] = ot[pk].mean; row["orig_peak_m"] = round(ot[pk].tokens / 1e6, 1)
    fz = C.m_traj("B", "rb-E2c-03"); first = min(fz)
    row["freeze_first_ckpt_nll"] = fz[first].mean; row["freeze_first_ckpt_m"] = round(fz[first].tokens / 1e6, 1)
    out.append(row)
    return out


# ----------------------------------------------------------------------------- E0 tables
def e0a(comps):
    sa = T.sa_end()
    def d(run):
        return C.pd(C.paper("A", T.SUITE_A[run], T.A_END), sa)
    e1b = T.endpoint("rb-E1b-01"); e3 = T.endpoint("rb-E3a-2.5B-01")
    acc = OrderedDict()
    for k in ("LERP K4", "LERP K16", "LERP K32", "FWM-Prob K4", "FWM-Weight K4", "FWM-Weight K16"):
        acc[k] = d(k)
    acc["FWM-LERP K4 (E1b)"] = C.pd(e1b, sa); acc["IndexSoftmax E3a-ii (E3a-2.5B)"] = C.pd(e3, sa)
    return [OrderedDict(condition=k, scale="124M @ 2.5B, seed 1337", dnll_vs_softmax=v[0], ci_lo=v[1], ci_hi=v[2]) for k, v in acc.items()]


def e0c():
    rows = []
    for K in (2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64):
        rows.append(OrderedDict(scheme=f"K-interval grid, K={K}", levels=K + 1, index_bits=math.ceil(math.log2(K + 1)), nonzero_levels_in_0_1="", underflow_below_z="",
                                note="K intervals → K+1 levels; bits = ceil(log2(K+1))"))
    fs = {}
    for p in sorted(glob.glob(os.path.join(CHL, "13_3_formats_*.json"))):
        j = json.load(open(p))
        for k, v in j["E0c_levels"].items():
            fs.setdefault(k, set()).add((v["nonzero_levels_in_0_1"], v["underflow_below_z"]))
    import torch
    bits = torch.arange(0x8000, 0xFF80, dtype=torch.int32).to(torch.int16)
    zb = torch.cat([bits.view(torch.bfloat16), torch.zeros(1, dtype=torch.bfloat16)])
    zb = zb[torch.isfinite(zb.float()) & (zb.float() <= 0)]
    wb = torch.exp(zb).float()
    bf_all = int(torch.unique(wb[wb > 0]).numel()); bf_uf = float(zb.float()[wb == 0].max()); bf_20 = int(torch.unique(wb[(wb > 0) & (zb.float() >= -20)]).numel())
    for k, v in sorted(fs.items()):
        (lev, uf), = v if len(v) == 1 else (sorted(v)[0],)
        exact = dict(levels_all_z=bf_all, underflow_all_z=bf_uf, sweep_recount=bf_20) if k == "softmax_bf16exp" else dict(levels_all_z=lev, underflow_all_z=uf, sweep_recount="")
        rows.append(OrderedDict(scheme=k.replace("softmax_", "E5 format "), levels="", index_bits="", nonzero_levels_in_0_1=lev, underflow_below_z=uf, **exact,
                                note="identical in all check records" if len(v) == 1 else "DIFFERS between stacks"))
    return rows


def e0d():
    # inputs: FA4's own roofline per 128^3 tile (forward); the two-pass row adds one QK^T
    mm, ex, sm = 1024, 1024, 768
    qk = mm // 2
    rows = [OrderedDict(implementation="FA4, exact softmax, single pass", matmul=mm, exp_unit=ex, shared_memory=sm, bound=max(mm, ex, sm), relative_to_fa4=1.0, function="exact softmax"),
            OrderedDict(implementation="lookup operator, single pass, grid anchored at the running max", matmul=mm, exp_unit=0, shared_memory=sm, bound=max(mm, 0, sm), relative_to_fa4=max(mm, 0, sm) / max(mm, ex, sm), function="different from the trained operator"),
            OrderedDict(implementation="lookup operator, two passes (full-row max first)", matmul=mm + qk, exp_unit=0, shared_memory=sm + qk, bound=max(mm + qk, 0, sm + qk), relative_to_fa4=max(mm + qk, sm + qk) / max(mm, ex, sm), function="same as the trained operator")]
    return rows


# ----------------------------------------------------------------------------- remaining derived tables
def e4_hist():
    hw = [C.paper("D", f"MinMax-Weight_K4_s{s}", 763) for s in (7, 42, 1337)]
    hp = [C.paper("D", f"MinMax-Prob_K4_s{s}", 763) for s in (7, 42, 1337)]
    hx = [C.paper("D", f"Exact_s{s}", 763) for s in (7, 42, 1337)]
    ew = [C.m_end("B", r) for r in ("rb-E4-01", "rb-E4-02", "rb-E4-03")]; ep = [C.m_end("B", r) for r in ("rb-E4-05", "rb-E4-06", "rb-E4-07")]
    rows = []
    def add(q, vals):
        rows.append(OrderedDict(quantity=q, s7=vals[0], s42=vals[1], s1337=vals[2], mean=float(np.mean(vals)) if vals[0] is not None else None))
    add("historical MinMax-Weight − MinMax-Prob, 100M (suite D, RTX 3090)", [a.mean - b.mean for a, b in zip(hw, hp)])
    add("EMA Weight − EMA Prob, 100M (this round, S1)", [a.mean - b.mean for a, b in zip(ew, ep)])
    add("historical MinMax-Weight − softmax, 100M (suite D, RTX 3090)", [a.mean - b.mean for a, b in zip(hw, hx)])
    hb = C.paper("B", "formal_124m_s1337_nearest_k4_weightste_250m", 1908).mean - C.paper("B", "formal_124m_s1337_nearest_k4_probste_250m", 1908).mean
    rows.append(OrderedDict(quantity="historical MinMax-Weight − MinMax-Prob, 250M (suite B, S1)", s7=None, s42=None, s1337=hb, mean=None))
    rows.append(OrderedDict(quantity="EMA Weight − EMA Prob, 250M (this round, S1)", s7=None, s42=None, s1337=C.m_end("B", "rb-E4-04").mean - C.m_end("B", "rb-E4-08").mean, mean=None))
    return rows


def forwards():
    e3 = C.m_end("A", "rb-E3a-2.5B-01"); sa = T.sa_end()
    rows = []
    for tag, name in (("", "native: IndexSoftmax UINT8 (E3a-ii, as trained)"), ("__fwd-E3a-i-fp", "IndexSoftmax with float normalisation (E3a-i forward)"),
                      ("__fwd-exact-softmax", "exact fp32 softmax"), ("__fwd-native-control", "native forward through the override wrapper (control)")):
        x = e3 if not tag else C.m_end("A", "rb-E3a-2.5B-01", tag)
        dn = C.pd(x, e3) if tag else (None, None, None); ds = C.pd(x, sa)
        rows.append(OrderedDict(forward=name, nll=x.mean, d_native=dn[0], d_native_lo=dn[1], d_native_hi=dn[2], d_softmax=ds[0], d_softmax_lo=ds[1], d_softmax_hi=ds[2],
                                bitwise_equal_native=bool(np.array_equal(x.arr, e3.arr)), checkpoint_sha256=x.sha))
    return rows


def e0b_trained(lm):
    idx = {(r["run_id"], r["step"], r["metric"]): r["value"] for r in lm}
    steps = {}
    for r in lm:
        steps[r["run_id"]] = max(steps.get(r["run_id"], 0), r["step"])
    rows = []
    for run, s in sorted(steps.items()):
        if run not in C.MAN or C.MAN[run]["experiment"] in ("C0", "C0b"):
            continue
        rows.append(OrderedDict(run_id=run, step=s, operator=T.cond_label(run), zeroed=idx.get((run, s, "zeroed_softmax_mass")), clip=idx.get((run, s, "below_window_entry_frac")),
                                rowsum=idx.get((run, s, "rowsum_native")), pmax=idx.get((run, s, "pmax_native"))))
    return rows


def main():
    global OUT
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=OUT); a = ap.parse_args(); OUT = a.out
    os.makedirs(OUT, exist_ok=True)
    R = {}
    R["runs"] = T.runs_table()
    R["comps"] = T.comparisons(); R["sm"] = T.seed_means(R["comps"])
    R["traj"] = T.trajectories(); R["lm"] = T.probes()
    R["bench"], R["held"] = T.downstream(); R["e6"] = T.e6()
    R["e2"] = e2_analysis(); R["e4hist"] = e4_hist(); R["fwd"] = forwards(); R["e2d"] = T.e2d_endpoints()
    R["comp_run"], R["comp_ev"] = T.compute()
    ce = OrderedDict()
    for r in R["comp_run"]:
        k = (r["experiment"], r["machine"]); x = ce.setdefault(k, OrderedDict(experiment=k[0], machine=k[1], runs=0, gpu_hours=0.0, planned_gpu_hours=0.0))
        x["runs"] += 1; x["gpu_hours"] += r["gpu_hours"]; x["planned_gpu_hours"] += r["planned_gpu_hours"]
    tot = OrderedDict(experiment="total", machine="A+B", runs=sum(x["runs"] for x in ce.values()), gpu_hours=sum(x["gpu_hours"] for x in ce.values()),
                      planned_gpu_hours=sum(x["planned_gpu_hours"] for x in ce.values()))
    R["comp_exp"] = list(ce.values()) + [tot]
    R["e0a"] = e0a(R["comps"]); R["e0c"] = e0c(); R["e0d"] = e0d(); R["e0b_trained"] = e0b_trained(R["lm"])
    consts = [OrderedDict(name="t_0.975,2 (paired t interval over 3 seeds)", value=C.T975_2), OrderedDict(name="bootstrap resamples", value=4000),
              OrderedDict(name="moving-block length", value=16), OrderedDict(name="block bootstrap seed (validation NLL)", value=20260918),
              OrderedDict(name="stage C bootstrap seed (benchmarks)", value=20260920), OrderedDict(name="single-seed interpretation threshold (nats)", value=0.01),
              OrderedDict(name="tokens per step", value=C.BT)]
    files = [
        ("runs.csv", R["runs"]),                          # the 98 runs: configuration, code-root hashes, status, endpoint NLL and checkpoint SHA-256, primary pair and ΔNLL, GPU-hours, command
        ("comparisons.csv", R["comps"]),                  # every endpoint comparison (primary pairs and same-seed contrasts) with ΔNLL and 95% CI
        ("seed_means.csv", R["sm"]),                      # means over seeds of the 100M comparisons, paired t intervals
        ("trajectories.csv", R["traj"]),                  # validation NLL of every evaluated checkpoint, with ΔNLL vs the pair at the same step
        ("probes_layer_mean.csv", R["lm"]),               # every attention-probe metric, layer mean, every probed checkpoint
        ("downstream_benchmarks.csv", R["bench"]),        # 7 benchmark configurations, paired differences with the paper's stage C bootstrap
        ("downstream_heldout.csv", R["held"]),            # 3 held-out corpora, paired differences with the paper's layer-1 resampling
        ("e2_analysis.csv", R["e2"]),                     # E2a/b chain analysis and E2c landmarks
        ("e4_historical_gap.csv", R["e4hist"]),           # E4: historical MinMax Weight − Prob gaps next to the EMA runs
        ("e2d_endpoints.csv", R["e2d"]),                  # E2d: the four branches' endpoints side by side, per seed
        ("e3a_2p5b_forwards.csv", R["fwd"]),              # E3a-2.5B endpoint under other forwards (weights unchanged)
        ("e0a_accuracy.csv", R["e0a"]),                   # E0a: measured accuracy at 2.5B
        ("e0b_trained_endpoints.csv", R["e0b_trained"]),  # zeroed mass and clip fraction at the endpoints of the Appendix I runs
        ("e0c_bitwidth.csv", R["e0c"]),                   # E0c: levels, index bits, format levels and underflow thresholds (sweep over z in [−20, 0], and exact over all z <= 0 for BF16)
        ("e0d_roofline.csv", R["e0d"]),                   # E0d: roofline estimate
        ("e6.csv", R["e6"]),                              # E6: all 50 evaluations (RTX 5090) with the RTX 3090 repeat
        ("compute_per_run.csv", R["comp_run"]),           # training GPU-hours per run (start/finish UTC epoch)
        ("compute_per_experiment.csv", R["comp_exp"]),    # training GPU-hours per experiment and machine, with the planned estimate
        ("compute_evaluation.csv", R["comp_ev"]),         # evaluation GPU time recorded in the result files
        ("constants.csv", consts),                        # statistical constants used
    ]
    for name, rows in files:
        write_csv(name, rows)
    print(f"written {len(files)} csv files to {os.path.join(OUT, 'csv')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Row builders of the Appendix I result tables: every table is a list of dicts computed here from the result files
(rb_dossier_core); rb_dossier.py writes them as csv files at full precision."""
import glob, json, os
from collections import OrderedDict
import numpy as np
import rb_dossier_core as C

SEEDS3 = (7, 42, 1337)
C0 = {7: "rb-C0-01", 42: "rb-C0-02", 1337: "rb-C0-03"}
SA_RUN = "formal_124m_s1337_exact_2p5b"
SUITE_A = {"softmax": SA_RUN, "LERP K4": "formal_124m_s1337_lerp_k4_fullgrad_2p5b", "LERP K16": "formal_124m_s1337_lerp_k16_fullgrad_2p5b",
           "LERP K32": "formal_124m_s1337_lerp_k32_fullgrad_2p5b", "FWM-Weight K4": "formal_124m_s1337_qrm_nearest_k4_tau6_fullgrad_2p5b",
           "FWM-Prob K4": "formal_124m_s1337_qrm_nearest_k4_tau6_probste_2p5b", "FWM-Weight K16": "formal_124m_s1337_qrm_nearest_k16_tau6_fullgrad_2p5b",
           "MinMax-Weight K4": "formal_124m_s1337_nearest_k4_fullgrad_2p5b", "MinMax-Prob K4": "formal_124m_s1337_minmax_nearest_k4_probste_2p5b"}
SB_SOFT = "formal_124m_s1337_exact_matched250m"
A_END, B250, D100 = 19074, 1908, 763


def endpoint(run):
    """The run's endpoint evaluation: machine B for B runs and for E1b (evaluated on B); machine A for E8a, E8c, E3a-2.5B."""
    m = "A" if run in ("rb-E8a-01", "rb-E8c-01", "rb-E3a-2.5B-01") else "B"
    return C.m_end(m, run)


def sa_end():
    return C.paper("A", SA_RUN, A_END)


# ----------------------------------------------------------------------------- pairs
def primary_pair(run):
    """(label, NLL of the pair at the run's endpoint) for the run list; None for reference runs."""
    e = C.MAN[run]["experiment"]; s = int(C.MAN[run]["seed"])
    n = int(C.arg(run, "stop_at_tokens", 0) or C.arg(run, "max_tokens"))
    E11 = {"rb-E11-01": "rb-E1-01", "rb-E11-02": "rb-E1-02", "rb-E11-03": "rb-E1-05", "rb-E11-04": "rb-E1-06",
           "rb-E11-05": "rb-E11-06", "rb-E11-07": "rb-E11-08", "rb-E11-09": "rb-E1-04"}
    E2D = {f"rb-E2d-{i:02d}": f"rb-E2d-{j:02d}" for i, j in ((2, 1), (4, 3), (6, 5), (8, 7), (10, 9), (12, 11))}
    if e == "C0":
        return None
    if e == "C0b":
        return "suite B softmax s1337 250M (paper, S1)", C.paper("B", SB_SOFT, B250)
    if run in E11:
        return f"{E11[run]} (full gradient, same seed)", C.m_end("B", E11[run])
    if e == "E2d":
        return (f"{E2D[run]} (same seed, no switch)", C.m_end("B", E2D[run])) if run in E2D else None
    if e == "E2c":
        return "suite A softmax s1337 at 625M (paper, S1)", C.paper("A", SA_RUN, 4769)
    if run == "rb-E7-01":
        return "rb-E11-06 (TF32 on, same root)", C.m_end("B", "rb-E11-06")
    if run == "rb-E7-02":
        r = C.REEVAL["C2_MMW4S_s7"]
        return "App. C.2 strict MinMax-Weight K4 s7 (TF32 on; re-evaluated on B, mean only)", float(r["got_nll"])
    if run == "rb-E8c-01":
        return "rb-E8a-01 (softmax, same seed)", endpoint("rb-E8a-01")
    if run in ("rb-E1b-01", "rb-E3a-2.5B-01", "rb-E8a-01"):
        return "suite A softmax s1337 2.5B (paper, S1)", sa_end()
    if n == 250_000_000:
        return "rb-C0b-01 (same-stack 250M softmax)", C.m_end("B", "rb-C0b-01")
    if n == 100_000_000:
        return f"{C0[s]} (same-stack softmax, same seed)", C.m_end("B", C0[s])
    return None


BT_ = C.BT


def branch_label(run):
    """W→W / W→P / P→P / P→W / W→W freeze-h for the E2 branch runs ('' otherwise)."""
    m = C.MAN[run]
    if not m["experiment"].startswith("E2"):
        return ""
    par = "W" if "weight" in (m["parent_surrogate"] or "").lower() else "P"
    if m["operator"].startswith("resolved"):
        return f"{par}→W freeze-h"
    return f"{par}→{'W' if 'Weight' in m['surrogate'] else 'P'}"


def runs_table():
    rl = C.run_log(); cl = C.ckpt_lists()
    rows = []
    for run, m in C.MAN.items():
        e = endpoint(run)
        n_target = int(C.arg(run, "stop_at_tokens", 0) or C.arg(run, "max_tokens"))
        envp = os.path.join(C.RUNS, run, "env.json")
        env = json.load(open(envp)) if os.path.exists(envp) else None
        man_hash = {"model.py": m["code_sha256_model"], "quant_attention.py": m["code_sha256_quant_attention"], "train_formal.py": m["code_sha256_train"]}
        hash_ok = (env["code_sha256"] == man_hash) if env else None
        fin = [k for k in cl if k[0] == run and cl[k]["kind"] == "endpoint"]
        lg = rl.get(run, {})
        gh = (lg["finished"] - lg["start"]) / 3600 if "finished" in lg and "start" in lg else None
        variant = "u8" if run == "rb-E3a-2.5B-01" else C.arg(run, "attn_variant", "")
        pp = primary_pair(run)
        row = OrderedDict(run_id=run, experiment=m["experiment"], priority=m["priority"], machine=m["machine"],
                          protocol=m["protocol"], seed=int(m["seed"]), operator=m["operator"], attn=C.arg(run, "attn"),
                          attn_variant=variant, K=m["K"], approx=C.arg(run, "approx", ""), range_mode=C.arg(run, "range_mode", ""),
                          surrogate=m["surrogate"], range_grad=m["range_grad"], start_from=m["start_from"], branch=branch_label(run),
                          parent_checkpoint=os.path.basename(os.path.dirname(m["parent_checkpoint"])) + "/" + os.path.basename(m["parent_checkpoint"]) if m["parent_checkpoint"] else "",
                          target_tokens=n_target, final_tokens=e.tokens if e else None, final_step=e.step if e else None,
                          code_root=os.path.basename(m["code_root"]), code_sha256_model=m["code_sha256_model"],
                          code_sha256_quant_attention=m["code_sha256_quant_attention"], code_sha256_train=m["code_sha256_train"],
                          code_hash_equals_trainer_record=hash_ok,
                          status="finished" if ("finished" in lg and fin) else "NOT FINISHED",
                          train_gpu_hours=gh, train_gpu=f"{m['machine']}:GPU{lg.get('gpu')}",
                          endpoint_nll=e.mean if e else None, endpoint_eval_machine=e.src.split(":")[0] if e else "",
                          endpoint_checkpoint_sha256=e.sha if e else "",
                          pair="", pair_nll=None, dnll=None, ci_lo=None, ci_hi=None,
                          command=C.clean_command(m["command"]).replace("{E3A_2P5B_VARIANT}", "u8"))
        if pp:
            row["pair"] = pp[0]
            if isinstance(pp[1], float):
                row["pair_nll"], row["dnll"] = pp[1], e.mean - pp[1]
            elif pp[1] is not None:
                d = C.pd(e, pp[1]); row["pair_nll"] = pp[1].mean; row["dnll"], row["ci_lo"], row["ci_hi"] = d
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------- endpoint comparisons
def comparisons():
    """Every endpoint comparison of the dossier: primary pairs and same-seed contrasts."""
    out = []

    def add(section, label, seed, run, pair, a, b, stack="same stack (S1)"):
        if a is None or b is None:
            return
        if isinstance(b, float):
            out.append(OrderedDict(section=section, comparison=label, seed=seed, run=run, pair=pair, nll_run=a.mean, nll_pair=b,
                                   dnll=a.mean - b, ci_lo=None, ci_hi=None, ci="mean only (no per-block array of the pair)", stack=stack,
                                   run_checkpoint_sha256=a.sha or "", n_blocks=len(a.arr)))
            return
        d = C.pd(a, b)
        out.append(OrderedDict(section=section, comparison=label, seed=seed, run=run, pair=pair, nll_run=a.mean, nll_pair=b.mean,
                               dnll=d[0], ci_lo=d[1], ci_hi=d[2], ci="moving-block bootstrap over validation blocks", stack=stack,
                               run_checkpoint_sha256=a.sha or "", n_blocks=len(a.arr)))

    B = lambda r: C.m_end("B", r)
    for run, m in C.MAN.items():
        pp = primary_pair(run)
        if pp:
            add(m["experiment"], "primary pair", int(m["seed"]), run, pp[0], endpoint(run), pp[1],
                "cross-run reference on S1" if "paper" in pp[0] else "same stack (S1)")
    # E5 seed 7 for BF16-exp / FP8: the paper's D.5 runs, RTX 3090 stack, paired with suite D softmax seed 7 (also 3090)
    for f, lab in (("bf16exp", "BF16-exp"), ("fp8e4m3exp", "FP8-E4M3 exp")):
        add("E5", f"{lab} seed 7 (App. D.5 run)", 7, f"D.5 {lab} s7", "suite D softmax s7 (paper, RTX 3090)",
            C.supp("rtx3090", f"sp_s7_softmax_{f}_100m"), C.paper("D", "Exact_s7", D100), "RTX 3090 stack (both runs)")
    same = [("E3a", "E3a-ii − E3a-i", [(7, "rb-E3a-05", "rb-E3a-01"), (42, "rb-E3a-06", "rb-E3a-02"), (1337, "rb-E3a-07", "rb-E3a-03"), (1337, "rb-E3a-08", "rb-E3a-04")]),
            ("E3b", "E3b-d − E3b-g", [(7, "rb-E3b-01", "rb-E3b-05"), (42, "rb-E3b-02", "rb-E3b-06"), (1337, "rb-E3b-03", "rb-E3b-07"), (1337, "rb-E3b-04", "rb-E3b-08")]),
            ("E3b", "E3b-g − E3a-i", [(7, "rb-E3b-05", "rb-E3a-01"), (42, "rb-E3b-06", "rb-E3a-02"), (1337, "rb-E3b-07", "rb-E3a-03"), (1337, "rb-E3b-08", "rb-E3a-04")]),
            ("E4", "E4 EMA Weight − EMA Prob", [(7, "rb-E4-01", "rb-E4-05"), (42, "rb-E4-02", "rb-E4-06"), (1337, "rb-E4-03", "rb-E4-07"), (1337, "rb-E4-04", "rb-E4-08")]),
            ("E5n", "E5n − BF16-exp (same seed)", [(42, "rb-E5n-02", "rb-E5-01"), (1337, "rb-E5n-03", "rb-E5-02"), (1337, "rb-E5n-04", "rb-E5-03")])]
    for sec, lab, lst in same:
        for s, a, b in lst:
            add(sec, lab + (" (250M)" if C.MAN[a]["protocol"].startswith("suiteB") else ""), s, a, b, B(a), B(b))
    for name, rn in (("FWM-Weight K4", "formal_124m_s1337_qrm_k4_tau6_matched250m"), ("FWM-Prob K4", "formal_124m_s1337_qrm_k4_tau6_probste_250m")):
        add("E1", f"E1-04 (FWM-LERP K4, 250M) − suite B {name}", 1337, "rb-E1-04", f"suite B {name} s1337 250M (paper, S1)", B("rb-E1-04"), C.paper("B", rn, B250),
            "cross-run reference on S1")
    # (2) E2d: cross-parent differences (switched branch vs the branch that always used the target rule)
    for s_, (ww, wp, pp, pw) in ((7, (1, 2, 3, 4)), (42, (5, 6, 7, 8)), (253, (9, 10, 11, 12))):
        r_ = lambda i: f"rb-E2d-{i:02d}"
        add("E2d", "W→P − P→P (different parents)", s_, r_(wp), r_(pp), B(r_(wp)), B(r_(pp)))
        add("E2d", "P→W − W→W (different parents)", s_, r_(pw), r_(ww), B(r_(pw)), B(r_(ww)))
    e1b = endpoint("rb-E1b-01"); e3 = endpoint("rb-E3a-2.5B-01")
    for name in ("LERP K4", "FWM-Weight K4", "FWM-Prob K4"):
        add("E1b", f"E1b − {name} (suite A s1337)", 1337, "rb-E1b-01", f"suite A {name} s1337 2.5B (paper, S1)", e1b, C.paper("A", SUITE_A[name], A_END), "cross-run reference on S1")
    for name in ("FWM-Prob K4", "FWM-Weight K16", "LERP K4"):
        add("E3a-2.5B", f"E3a-2.5B − {name} (suite A s1337)", 1337, "rb-E3a-2.5B-01", f"suite A {name} s1337 2.5B (paper, S1)", e3, C.paper("A", SUITE_A[name], A_END), "cross-run reference on S1")
    add("E8", "seed-1337 counterpart: LERP K16 − softmax (paper)", 1337, "suite A LERP K16 s1337", "suite A softmax s1337",
        C.paper("A", SUITE_A["LERP K16"], A_END), sa_end(), "paper suite A, S1")
    # E11-06 vs the App. C.2 run of the same configuration (identity check, kept as a comparison too)
    add("E11", "E11-06 − App. C.2 FWM-Weight K4 strict s7 (same configuration)", 7, "rb-E11-06", "App. C.2 run (paper, S1)",
        B("rb-E11-06"), C.supp("rtx5090", "sp_s7_fwm_nearest_k4_tau6_weightste_strict_100m"))
    for sd in (7, 42):
        full = C.paper("D", f"LERP_K32_s{sd}", D100)
        for kind in ("minonly", "maxonly"):
            add("E11", f"App. F {kind.replace('only', '_only')} − full (MinMax-LERP K32, 100M)", sd, f"App. F {kind} s{sd}", f"suite D LERP K32 s{sd}",
                C.supp("rtx3090", f"sp_s{sd}_lerp_k32_{kind}_100m"), full, "RTX 3090 stack (both runs)")
    # E2: freeze-h − W→W at 250M and the E2d switch effects come from the primary pairs; add freeze-h contrasts
    for a, b in (("rb-E2ab-03", "rb-E2ab-01"), ("rb-E2ab-08", "rb-E2ab-06"), ("rb-E2c-03", "rb-E2c-01")):
        add(C.MAN[a]["experiment"], f"{branch_label(a)} − {branch_label(b)} (same branch point)", 1337, a, b, B(a), B(b))
    for a, b in (("rb-E2ab-02", "rb-E2ab-01"), ("rb-E2ab-04", "rb-E2ab-01"), ("rb-E2ab-05", "rb-E2ab-04"), ("rb-E2ab-07", "rb-E2ab-06"),
                 ("rb-E2ab-09", "rb-E2ab-06"), ("rb-E2ab-10", "rb-E2ab-09"), ("rb-E2c-02", "rb-E2c-01")):
        add(C.MAN[a]["experiment"], f"{branch_label(a)} − {branch_label(b)} (same branch point)", 1337, a, b, B(a), B(b))
    return out


def cond_label(run):
    m = C.MAN[run]
    v = "u8" if run == "rb-E3a-2.5B-01" else C.arg(run, "attn_variant", "")
    return " | ".join(x for x in (m["operator"], f"K={m['K']}" if m["K"] else "", m["surrogate"], m["range_grad"], v) if x)


def seed_means(comps):
    """Means over seeds of the 100M paired differences: 3 seeds -> paired t interval (t_0.975,2 = 4.303); 2 seeds -> mean only."""
    groups = OrderedDict()
    for r in comps:
        if r["ci_lo"] is None:
            continue
        if r["comparison"] == "primary pair" and r["run"] in C.MAN:
            m = C.MAN[r["run"]]
            if m["experiment"] == "E2d":
                groups.setdefault(("E2d", f"{branch_label(r['run'])} − {branch_label(r['pair'].split(' ')[0])} (same parent)", "same seed"), []).append(r)
                continue
            if not m["protocol"].startswith("suiteD") or m["experiment"] in ("E7", "C0"):
                continue
            key = (m["experiment"], cond_label(r["run"]), r["pair"].split(" (")[0].replace("rb-C0-01", "C0").replace("rb-C0-02", "C0").replace("rb-C0-03", "C0")
                   if not (m["experiment"] == "E11" and not r["pair"].startswith("rb-C0")) else "full gradient, same seed")
        elif r["comparison"].endswith("(App. D.5 run)"):
            key = ("E5", cond_label("rb-E5-01" if "BF16" in r["comparison"] else "rb-E5-04"), "C0")
        elif r["section"] in ("E3a", "E3b", "E4", "E5n", "E2d") and r["comparison"] != "primary pair" and "250M" not in r["comparison"]:
            key = (r["section"], r["comparison"], "same seed")
        elif r["comparison"].startswith("App. F"):
            key = ("E11", r["comparison"], "suite D LERP K32 same seed")
        else:
            continue
        groups.setdefault(key, []).append(r)
    out = []
    for (sec, lab, pair), rs in groups.items():
        seeds = [r["seed"] for r in rs]
        ds = [r["dnll"] for r in rs]
        row = OrderedDict(section=sec, condition=lab, pair=pair, runs=";".join(r["run"] for r in rs), seeds=";".join(str(x) for x in seeds),
                          per_seed_dnll=";".join(repr(x) for x in ds), n_seeds=len(set(seeds)), mean_dnll=float(np.mean(ds)), t_lo=None, t_hi=None,
                          stacks=";".join(sorted(set(r["stack"] for r in rs))))
        if len(rs) == 3 and len(set(seeds)) == 3:
            _, row["t_lo"], row["t_hi"] = C.tint(ds)
        out.append(row)
    return out


# ----------------------------------------------------------------------------- trajectories
def trajectory_pair(run):
    """(label, {step: NLL}) of the run's trajectory pair, if one exists at matching steps."""
    pp = primary_pair(run)
    if pp is None or isinstance(pp[1], float):
        return None
    lab = pp[0]
    if lab.startswith("suite A softmax"):
        return "suite A softmax s1337 (paper, S1), same checkpoint", C.paper_traj("A", SA_RUN)
    if lab.startswith("suite B softmax"):
        return lab, C.paper_traj("B", SB_SOFT)
    if lab.startswith("rb-E8a"):
        return lab, C.m_traj("A", "rb-E8a-01")
    r = lab.split(" ")[0]
    return lab, C.m_traj("B", r)


def trajectories():
    out = []
    for run, m in C.MAN.items():
        mach = "A" if run in ("rb-E8a-01", "rb-E8c-01", "rb-E3a-2.5B-01") else "B"
        tr = C.m_traj(mach, run)
        tp = trajectory_pair(run)
        for step, x in tr.items():
            row = OrderedDict(run_id=run, experiment=m["experiment"], step=step, tokens=x.tokens, tokens_m=round(x.tokens / 1e6, 1),
                              nll=x.mean, eval_machine=mach, checkpoint_sha256=x.sha, pair="", pair_nll=None, dnll=None, ci_lo=None, ci_hi=None)
            if tp and step in tp[1]:
                d = C.pd(x, tp[1][step]); row.update(pair=tp[0], pair_nll=tp[1][step].mean, dnll=d[0], ci_lo=d[1], ci_hi=d[2])
            out.append(row)
    # reference trajectories used by the figures and the E2 / E8 sections (paper records)
    for lab, suite, rn in (("suite A softmax s1337 (paper)", "A", SA_RUN), ("suite A LERP K16 s1337 (paper)", "A", SUITE_A["LERP K16"]),
                           ("suite A MinMax-Weight K4 s1337 (paper)", "A", SUITE_A["MinMax-Weight K4"]),
                           ("suite B softmax s1337 (paper)", "B", SB_SOFT), ("suite B MinMax-Weight K4 s1337 (paper)", "B", "formal_124m_s1337_nearest_k4_weightste_250m"),
                           ("suite B MinMax-Prob K4 s1337 (paper)", "B", "formal_124m_s1337_nearest_k4_probste_250m")):
        sa = C.paper_traj("A", SA_RUN)
        for step, x in C.paper_traj(suite, rn).items():
            row = OrderedDict(run_id=lab, experiment="reference", step=step, tokens=x.tokens, tokens_m=round(x.tokens / 1e6, 1), nll=x.mean,
                              eval_machine="paper (S1)", checkpoint_sha256="", pair="", pair_nll=None, dnll=None, ci_lo=None, ci_hi=None)
            if rn == SUITE_A["LERP K16"] and step in sa:
                d = C.pd(x, sa[step]); row.update(pair="suite A softmax s1337 (paper)", pair_nll=sa[step].mean, dnll=d[0], ci_lo=d[1], ci_hi=d[2])
            out.append(row)
    return out


# ----------------------------------------------------------------------------- probes
def probes():
    """Long format: every probe metric, layer mean, every probed checkpoint (machines A and B)."""
    lm_rows = []
    for mach in ("B", "A"):
        for p in sorted(glob.glob(os.path.join(C.EV, "probe", "*", "*", "probe.json"))):
            d = json.load(open(p))
            if d["host"] != C.HOST[mach]:
                continue
            base = OrderedDict(run_id=d["label"], step=int(d["step"]), tokens_m=round(int(d["tokens_done"]) / 1e6, 1), eval_machine=mach,
                               probes=",".join(d.get("probes", [])), checkpoint_sha256=d["provenance"]["checkpoint"]["sha256"],
                               selfcheck=d.get("selfcheck_P_times_V_equals_output"))
            for k, v in d["layer_mean"].items():
                lm_rows.append(OrderedDict(base, metric=k, value=v))
            ema = [v for L in d["layers"] for k, v in L.items() if k.startswith("ema_buffer_head")]
            if ema:
                lm_rows.append(OrderedDict(base, metric="ema_buffer_mean", value=float(np.mean(ema))))
            spn = [v for L in d["layers"] for k, v in L.items() if k.startswith("span_head")]
            if spn:
                lm_rows.append(OrderedDict(base, metric="span_head_mean", value=float(np.mean(spn))))
            if "grad" in d:
                lm_rows.append(OrderedDict(base, metric="param_grad_norm_mean", value=d["grad"]["param_grad_norm_mean"]))
                lm_rows.append(OrderedDict(base, metric="patched_loss_equals_normal_loss", value=float(d["grad"]["patched_loss_equals_normal_loss"])))
    return lm_rows


# ----------------------------------------------------------------------------- downstream
def downstream():
    PS = C.PS
    bench, held = [], []
    comps = [("rb-E1b-01 − softmax s1337", ("B", "rbE1b01"), ("P", "Exact"), ("B", "rb-E1b-01", "rb-E1b-01"), "softmax"),
             ("rb-E1b-01 − LERP K4 s1337", ("B", "rbE1b01"), ("P", "LERP_K4"), ("B", "rb-E1b-01", "rb-E1b-01"), "LERP K4"),
             ("rb-E8a-01 − softmax s1337 (seed 7 vs seed 1337)", ("B", "rbE8a01"), ("P", "Exact"), ("B", "rb-E8a-01", "rb-E8a-01"), "softmax"),
             ("rb-E8c-01 − rb-E8a-01 (LERP K16 − softmax, seed 7)", ("A", "rbE8c01"), ("B", "rbE8a01"), ("A", "rb-E8c-01", "rb-E8c-01"), ("B", "rb-E8a-01", "rb-E8a-01")),
             ("LERP K16 − softmax, seed 1337 (paper)", ("P", "LERP_K16"), ("P", "Exact"), ("P", SUITE_A["LERP K16"]), "softmax"),
             ("rb-E3a-2.5B-01 − softmax s1337", ("A", "rbE3a2.5B01"), ("P", "Exact"), ("A", "rb-E3a-2.5B-01", "rb-E3a-2.5B-01"), "softmax"),
             ("rb-E3a-2.5B-01 − FWM-Prob K4 s1337", ("A", "rbE3a2.5B01"), ("P", "QRMProb_K4"), ("A", "rb-E3a-2.5B-01", "rb-E3a-2.5B-01"), "FWM-Prob K4")]
    def hpb(spec, corpus):
        if spec[0] == "P":
            return PS.paper_pb(spec[1], corpus)
        return PS.l1_pb(spec[0], spec[1], spec[2], corpus)
    for lab, a, b, ha, hb in comps:
        for cfg, metric in PS.PRIM:
            r = PS.compare(cfg, metric, a, b)
            bench.append(OrderedDict(comparison=lab, config=cfg, metric=metric, unit="percentage points" if metric in PS.S3.PCT else "nats",
                                     value_run=r[0], value_pair=r[1], diff=r[2], ci_lo=r[3], ci_hi=r[4],
                                     ci_excludes_zero=bool(r[3] > 0 or r[4] < 0)))
        for corpus in PS.CORPORA:
            xa = hpb(ha, corpus)
            xb = hpb(("P", SUITE_A[hb]) if isinstance(hb, str) else hb, corpus)
            d = PS.heldout_paired(xa, xb, corpus)
            held.append(OrderedDict(comparison=lab, corpus=corpus, nll_run=float(xa.mean()), nll_pair=float(xb.mean()), diff=d[0], ci_lo=d[1], ci_hi=d[2],
                                    resampling="document clusters" if PS.clusters(corpus) is not None else "moving blocks (L=16)", n_blocks=len(xa),
                                    ci_excludes_zero=bool(d[1] > 0 or d[2] < 0)))
    return bench, held


# ----------------------------------------------------------------------------- E6
def e6():
    out = []
    loc = os.path.join(C.EV, "e6_rtx3090")
    for d in sorted(glob.glob(os.path.join(C.EV, "e6", "*", "*", "summary.json"))):
        s = json.load(open(d)); ep, setting = s["endpoint"], s["setting"]
        arr = np.load(os.path.join(os.path.dirname(d), "per_block_nll.npy"))
        ref = np.load(os.path.join(C.EV, "e6", ep, "ref", "per_block_nll.npy"))
        dd = C.st.paired(arr, ref, kind="blocks") if setting != "ref" else None
        l3 = os.path.join(loc, ep, setting, "per_block_nll.npy")
        a3 = np.load(l3) if os.path.exists(l3) else None
        out.append(OrderedDict(endpoint=ep, setting=setting, operator=json.dumps(s.get("operator")) if not isinstance(s.get("operator"), str) else s["operator"],
                               dataset=s["dataset"], n_blocks=int(s["n_blocks"]), nll_5090=float(s["mean_nll"]), nonfinite=int(s.get("nonfinite", 0)),
                               recorded_native=float(s["recorded_native"]) if s.get("recorded_native") not in (None, "") else None,
                               dnll_vs_ref=None if dd is None else dd["delta"], ci_lo=None if dd is None else dd["ci_lo"], ci_hi=None if dd is None else dd["ci_hi"],
                               bitwise_equal_ref=bool(np.array_equal(arr, ref)),
                               nll_3090=None if a3 is None else float(a3.mean()),
                               dnll_3090_minus_5090=None if a3 is None else float(a3.mean() - arr.mean()),
                               checkpoint_sha256=(s.get("provenance") or {}).get("checkpoint", {}).get("sha256", ""), seconds=s.get("seconds")))
    return out


# ----------------------------------------------------------------------------- compute
L1_MACHINE = {"Exact_s1337_check": "B", "rb-E1b-01": "B", "rb-E8a-01": "B", "rb-E8c-01": "A", "rb-E3a-2.5B-01": "A"}   # machine of each held-out evaluation
def compute():
    rl = C.run_log()
    per_run = []
    for run, m in C.MAN.items():
        lg = rl.get(run, {})
        h = (lg["finished"] - lg["start"]) / 3600 if "finished" in lg else None
        per_run.append(OrderedDict(run_id=run, experiment=m["experiment"], machine=m["machine"], gpu_hours=h, planned_gpu_hours=float(C.CSVR[run]["gpu_h"]),
                                   start_utc=lg.get("start"), finished_utc=lg.get("finished")))
    ev = []
    for mach in ("B", "A"):
        for kind, pat, key in (("validation NLL", "nll/*/*/summary.json", "seconds"), ("attention probes", "probe/*/*/probe.json", "seconds"),
                               ("E6", "e6/*/*/summary.json", "seconds")):
            fs = [f for f in sorted(glob.glob(os.path.join(C.EV, pat))) if json.load(open(f))["host"] == C.HOST[mach]]
            sec = sum(float(json.load(open(f)).get(key) or 0) for f in fs)
            ev.append(OrderedDict(machine=mach, kind=kind, units=len(fs), gpu_hours=sec / 3600))
        fs = [f for f in sorted(glob.glob(os.path.join(C.EV, "downstream", "layer1", "*", "summary.json"))) if L1_MACHINE[os.path.basename(os.path.dirname(f))] == mach]
        sec = sum(float(json.load(open(f)).get("seconds_native") or 0) + float(json.load(open(f)).get("seconds_exact") or 0) for f in fs)
        ev.append(OrderedDict(machine=mach, kind="held-out corpora (native + exact forward)", units=len(fs), gpu_hours=sec / 3600))
    return per_run, ev


def e2d_endpoints():
    out = []
    for s_, ids in ((7, (1, 2, 3, 4)), (42, (5, 6, 7, 8)), (253, (9, 10, 11, 12))):
        e = [C.m_end("B", f"rb-E2d-{i:02d}") for i in ids]
        out.append(OrderedDict(seed=s_, parent_checkpoint_m=40.1, ww=e[0].mean, wp=e[1].mean, pp=e[2].mean, pw=e[3].mean,
                               wp_minus_ww=e[1].mean - e[0].mean, pw_minus_pp=e[3].mean - e[2].mean,
                               wp_minus_pp=e[1].mean - e[2].mean, pw_minus_ww=e[3].mean - e[0].mean,
                               runs=";".join(f"rb-E2d-{i:02d}" for i in ids)))
    return out

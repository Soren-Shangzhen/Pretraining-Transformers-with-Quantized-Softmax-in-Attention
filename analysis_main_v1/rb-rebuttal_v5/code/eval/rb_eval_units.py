#!/usr/bin/env python3
"""Evaluation units for machine B (user decision 2026-11-12; protocol = plan section 4).

Writes <EVAL>/units.jsonl (EVAL = $RB_RUNS/_eval), one unit per line, in priority order:
  phase 1  validation NLL of every run's endpoint (batch 1, TF32 off, the run's own frozen code root;
           100M runs on WikiText-103, 250M/625M runs on FineWebEdu; per-block NLL saved by rb_eval.py)
  phase 3  validation NLL of the intermediate checkpoints (all checkpoints; E5/E5n 100M and E7: endpoint only),
           plus the E2 branch points (parents) with their own code roots
Runs whose final checkpoint does not exist yet are skipped; rerun the builder after training (units are
deduplicated by unit_id, so a rerun only appends what is new).
Phase 2 (machine A's evaluations), 4 (probes, E0b) and 5 (E6) are added by --phase 2/4/5 once their inputs exist.
A unit: {unit_id, phase, kind, run_id, cmd, requires: [paths that must exist], jobs: [...]}.  The queue
(rb_eval_queue.py) starts a unit only when its requirements exist; rb_eval.py skips outputs already written,
so a unit can be rerun safely.
usage (on B, after `source rb_env.sh`):  rb_eval_units.py --phase 1,3 [--dry-run]
"""
import argparse, csv, glob, json, os, re, shlex

HERE = os.path.dirname(os.path.abspath(__file__))
RB = os.path.abspath(os.path.join(HERE, "..", ".."))
MAN = os.path.join(RB, "manifests", "rb-runs_v5_manifest.json")
FIELDS = ["label", "eval_root", "checkpoint", "dataset", "tokens_m", "role", "surrogate"]
ENDPOINT_ONLY = {"E7"}                       # user: E7 endpoint only; E5/E5n: 100M endpoint only, 250M all


def ex(s):
    return os.path.expandvars(s)


def ckpts(rd):
    """All complete checkpoints of a run directory, by step (ckpt.pt, a link to the latest, is skipped)."""
    out = []
    for p in glob.glob(os.path.join(rd, "ckpt_step*_*.pt")):
        m = re.match(r"ckpt_step(\d+)_([\d.]+)m\.pt$", os.path.basename(p))
        if m:
            out.append((int(m.group(1)), float(m.group(2)), p))
    return sorted(out)


def arg(cmd, name):
    m = re.search(rf"--{name}\s+(\S+)", cmd)
    return m.group(1) if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="1,3")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    phases = {int(x) for x in a.phase.split(",")}
    runs_dir, ev = os.environ["RB_RUNS"], os.path.join(os.environ["RB_RUNS"], "_eval")
    jobs_dir = os.path.join(ev, "jobs")
    os.makedirs(jobs_dir, exist_ok=True)
    man = json.load(open(MAN))["runs"]
    py, evpy = os.environ["RB_PY"], os.path.join(HERE, "rb_eval.py")
    units, unf = [], os.path.join(ev, "units.jsonl")
    have = set()
    if os.path.exists(unf):
        have = {json.loads(l)["unit_id"] for l in open(unf) if l.strip()}

    def nll_unit(uid, phase, run_id, root, rows):
        tsv = os.path.join(jobs_dir, f"{uid}.tsv")
        if not a.dry_run:
            with open(tsv, "w", newline="") as f:
                w = csv.writer(f, delimiter="\t"); w.writerow(FIELDS)
                for r in rows:
                    w.writerow([r[k] for k in FIELDS])
        cmd = f"{py} {evpy} --jobs {tsv} --out {ev}/nll --root {root}"
        units.append(dict(unit_id=uid, phase=phase, kind="nll", run_id=run_id, cmd=cmd,
                          requires=sorted({r["checkpoint"] for r in rows} | {root}), n_jobs=len(rows)))

    b_runs = [r for r in man if r["machine"] == "B"]
    later = []
    for r in b_runs:
        rid, exp = r["run_id"], r["experiment"]
        rd = os.path.join(runs_dir, rid)
        cs = ckpts(rd)
        cmd = r["command"]
        end = int(arg(cmd, "stop_at_tokens") or arg(cmd, "max_tokens"))
        if not cs or cs[-1][0] != -(-end // 131072):            # final checkpoint: step ceil(end / 131072)
            print(f"not finished yet (rerun the builder later): {rid}")
            continue
        root = ex(r["code_root"])
        ds = os.path.basename(arg(cmd, "data_dir"))
        sur = arg(cmd, "nearest_surrogate")
        endpoint_only = exp in ENDPOINT_ONLY or (exp in ("E5", "E5n") and ds == "wikitext103")
        row = lambda c: dict(label=rid, eval_root=root, checkpoint=c[2], dataset=ds, tokens_m=f"{c[1]:.1f}",
                             role=exp, surrogate=sur)
        if 1 in phases:
            nll_unit(f"p1_nll_end_{rid}", 1, rid, root, [row(cs[-1])])
        if 3 in phases and not endpoint_only and len(cs) > 1:
            later.append((f"p3_nll_all_{rid}", 3, rid, root, [row(c) for c in cs[:-1]]))
    if 3 in phases:
        # E2 branch points with their own frozen roots (same-machine reference for the branch curves)
        parents = {}
        for r in b_runs:
            ini, proot = arg(r["command"], "init_from"), arg(r["command"], "parent_code_root")
            if ini:
                parents[ex(ini)] = (ex(proot), r["experiment"], os.path.basename(arg(r["command"], "data_dir")))
        for p, (proot, exp, ds) in sorted(parents.items()):
            m = re.search(r"ckpt_step(\d+)_([\d.]+)m\.pt$", p)
            lab = "parent_" + os.path.basename(os.path.dirname(p))
            later.append((f"p3_nll_{lab}_{m.group(1)}", 3, lab, proot,
                          [dict(label=lab, eval_root=proot, checkpoint=p, dataset=ds, tokens_m=f"{float(m.group(2)):.1f}",
                                role=f"{exp} branch point", surrogate="")]))
        # longest first inside phase 3 (more checkpoints and FineWebEdu first), so the tail is short
        later.sort(key=lambda u: -(len(u[4]) * (4 if u[4][0]["dataset"] == "finewebedu_3b" else 1)))
        for u in later:
            nll_unit(*u)
    if 2 in phases:
        # Plan section 3 / user 2026-11-12: machine A's runs evaluated on B.  Gate first: the suite A seed-1337
        # softmax endpoint re-evaluated with its own root must match the S1 record; every other phase-2 unit
        # requires the gate's PASS file.
        cloud = os.environ.get("RB_CLOUD", "<CLOUD_DISK>")
        sa = f"{cloud}/formal_runs_scale_v1/formal_124m_s1337_exact_2p5b/ckpt_step019074_2500.1m.pt"
        sa_root = f"{cloud}/formal_code_factorial250_v1/formal_base_v2"
        lab = "suiteA_s1337_softmax_2500m"
        tsv = os.path.join(jobs_dir, f"p2_gate_{lab}.tsv")
        if not a.dry_run:
            with open(tsv, "w", newline="") as f:
                w = csv.writer(f, delimiter="\t"); w.writerow(FIELDS)
                w.writerow([lab, sa_root, sa, "finewebedu_3b", "2500.1", "suite A softmax (pairing, S1 gate)", ""])
        gate = os.path.join(ev, "gates", f"{lab}.PASS.json")
        units.append(dict(unit_id=f"p2_gate_{lab}", phase=2, kind="gate", run_id=lab, requires=[sa, sa_root], n_jobs=1,
                          cmd=f"{py} {evpy} --jobs {tsv} --out {ev}/nll --root {sa_root} && {py} "
                              f"{os.path.join(HERE, 'rb_eval_gate.py')} --name {lab} --summary-dir {ev}/nll/{lab} "
                              f"--recorded 3.1927719292093495 --recorded-source "
                              + shlex.quote("eval_native_v1/A_longhorizon_124m_2p5b/formal_124m_s1337_exact_2p5b/"
                                            "ckpt_step019074_2500.1m/summary.json (RTX 5090, torch 2.13.0)")))
        rb_root = os.path.join(RB, "code", "rb-root_v1")
        ao = os.environ.get("RB_A_EVAL", "<POD_WORKSPACE>/A_eval_model_only")
        for rid in ("rb-E8a-01", "rb-E1b-01"):
            files = sorted(glob.glob(os.path.join(ao, rid, "ckpt_step*.model_only.pt")))
            exp = os.path.join(ao, rid, "ckpt_step019074_2500.1m.model_only.pt")
            row = lambda p: dict(label=rid, eval_root=rb_root, checkpoint=p, dataset="finewebedu_3b", role=rid[3:6],
                                 tokens_m=re.search(r"_([\d.]+)m\.model_only", p).group(1), surrogate="")
            nll_unit(f"p2_nll_end_{rid}", 2, rid, rb_root, [row(exp)])
            units[-1]["requires"].append(gate)
            if rid == "rb-E1b-01":
                rest = [f for f in files if f != exp]
                if len(rest) == 19:
                    nll_unit(f"p2_nll_all_{rid}", 2, rid, rb_root, [row(f) for f in rest])
                    units[-1]["requires"].append(gate)
                else:
                    print(f"{rid}: {len(rest)} of 19 intermediate checkpoints on B so far (rerun --phase 2 later)")
        # downstream (rb_downstream.py, separate lm-eval 0.4.13 environment): the suite A softmax endpoint first as
        # a same-machine check against the paper's S1 results, then E8a and E1b (after the NLL gate)
        dpy = os.environ.get("RB_DS_PY", "<POD_WORKSPACE>/envs/rb-ds/bin/python")
        dsr = os.path.join(HERE, "rb_downstream.py")
        ready = os.path.join(os.path.dirname(os.path.dirname(dpy)), "READY")
        pre = os.path.join(ev, "downstream", "PREFETCH_OK")
        l1 = [f"{os.environ['RB_TRAIN_ROOT']}/analysis_main_v1/downstream_v1/data/layer1/{c}.bin"
              for c in ("wikitext103_test", "ptb_test", "c4_validation")]
        targets = [("Exact_s1337_check", sa, [])] + [(rid, os.path.join(ao, rid, "ckpt_step019074_2500.1m.model_only.pt"),
                                                      [gate]) for rid in ("rb-E8a-01", "rb-E1b-01")]
        for what in ("layer1", "stagec"):
            for cond, ck, extra in targets:
                req = [ready, ck, dsr] + extra + (l1 if what == "layer1" else [pre])
                units.append(dict(unit_id=f"p2_ds_{what}_{cond}", phase=2, kind=f"downstream_{what}", run_id=cond,
                                  n_jobs=1, requires=req,
                                  cmd=f"HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 {dpy} {dsr} --what {what} --cond {cond}"))
    if 4 in phases:
        # probes (rb_probe.py) per the user's list of 2026-11-12: all checkpoints except E5/E5n 100M (endpoint);
        # none for E2d and E7.  C0 also gets the E3a forwards on its own scores (E3a-i / E3a-ii / C0 comparison);
        # E11 the score-gradient probe, and E1 as its full-gradient LERP counterpart.
        prb = os.path.join(HERE, "rb_probe.py")
        flags = {"C0": "cf_e3a", "E11": "grad", "E1": "grad"}
        pl = []
        for r in b_runs:
            rid, exp = r["run_id"], r["experiment"]
            if exp in ("E2d", "E7") or exp.startswith("E2d"):
                continue
            cs, cmd = ckpts(os.path.join(runs_dir, rid)), r["command"]
            end = int(arg(cmd, "stop_at_tokens") or arg(cmd, "max_tokens"))
            if not cs or cs[-1][0] != -(-end // 131072):
                continue
            ds = os.path.basename(arg(cmd, "data_dir"))
            sel = cs[-1:] if exp in ("E5", "E5n") and ds == "wikitext103" else cs
            root = ex(r["code_root"])
            rows = [dict(label=rid, eval_root=root, checkpoint=c[2], dataset=ds, tokens_m=f"{c[1]:.1f}", role=exp,
                         surrogate="", probes=flags.get(exp, "")) for c in sel]
            pl.append((f"p4_probe_{rid}", rid, root, rows))
        pl.sort(key=lambda u: -(len(u[3]) * (4 if u[3][0]["dataset"] == "finewebedu_3b" else 1)))
        for uid, rid, root, rows in pl:
            tsv = os.path.join(jobs_dir, f"{uid}.tsv")
            if not a.dry_run:
                with open(tsv, "w", newline="") as f:
                    w = csv.writer(f, delimiter="\t"); w.writerow(FIELDS + ["probes"])
                    for x in rows:
                        w.writerow([x[k] for k in FIELDS + ["probes"]])
            units.append(dict(unit_id=uid, phase=4, kind="probe", run_id=rid, n_jobs=len(rows),
                              cmd=f"{py} {prb} --jobs {tsv} --out {ev}/probe --root {root}",
                              requires=sorted({x["checkpoint"] for x in rows} | {root})))
    if 5 in phases:
        # E6 (rb_e6.py): six existing Nearest endpoints, every evaluation setting; the same script runs on the local 3090
        import importlib.util
        spec = importlib.util.spec_from_file_location("rb_e6", os.path.join(HERE, "rb_e6.py"))
        e6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e6)
        for name, (ck, ds, root, kind, _, _) in e6.ENDPOINTS.items():
            units.append(dict(unit_id=f"p5_e6_{name}", phase=5, kind="e6", run_id=name, n_jobs=1,
                              cmd=f"{py} {os.path.join(HERE, 'rb_e6.py')} --endpoint {name} --out {ev}/e6",
                              requires=[ck, f"{e6.ROOTS}/native_{root}", f"{e6.ROOTS}/strict_{root}"]))
    new = [u for u in units if u["unit_id"] not in have]
    print(f"{len(units)} units built, {len(new)} new; jobs: {sum(u['n_jobs'] for u in new)}")
    for u in new[:5] + new[-3:]:
        print(" ", u["unit_id"], u["n_jobs"], "jobs")
    if not a.dry_run and new:
        with open(unf, "a") as f:
            for u in new:
                f.write(json.dumps(u) + "\n")
        print("appended to", unf)


if __name__ == "__main__":
    main()

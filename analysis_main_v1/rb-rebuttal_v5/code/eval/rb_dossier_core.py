#!/usr/bin/env python3
"""Data access and statistics for the result tables of Appendix I: every number is computed by script from the result
files.

Sources (all read-only):
  the evaluation outputs of machines A and B (runs/_eval; the field "host" of each record names the machine);
  the paper's historical evaluations (eval_native_v1 suites A and B on the S1 stack; suite D and the App. C.2 / D.5 / F
  runs from the supplement evaluations); the run manifest, its recorded rules and decisions, the check records.
Statistics: the paper's own code (downstream_v1/code/stats.py; rb_paper_stats for the downstream methods).
"""
import csv, glob, gzip, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RB = os.path.abspath(os.path.join(HERE, "..", ".."))
AM = os.path.abspath(os.path.join(RB, ".."))
RUNS = os.path.join(RB, "runs")                    # output directory of the training and evaluation code (RB_RUNS)
EV = os.path.join(RUNS, "_eval")
HOST = {"A": "machine-A", "B": "machine-B"}        # field "host" of the evaluation records
sys.path.insert(0, os.path.join(AM, "downstream_v1", "code"))
sys.path.insert(0, HERE)
import stats as st  # noqa: E402
import rb_paper_stats as PS  # noqa: E402

BT = 131072
T975_2 = 4.302652729911275          # t_{0.975, 2}
MANJ = json.load(open(os.path.join(RB, "manifests", "rb-runs_v5_manifest.json")))
MAN = {r["run_id"]: r for r in MANJ["runs"]}
RULES = MANJ["check_rules"]
CSVR = {k: {"run_id": k, "gpu_h": r["gpu_h_plan"]} for k, r in MAN.items()}        # planned GPU-hours (manifest field gpu_h_plan)
REEVAL = {r["label"]: r for r in json.load(open(os.path.join(RB, "checks", "13_2_reeval_machine-B.json")))}
SUITE = {"A": os.path.join(AM, "eval_native_v1", "A_longhorizon_124m_2p5b"),
         "B": os.path.join(AM, "eval_native_v1", "B_factorial_124m_250m"),
         "D": os.path.join(AM, "D_robustness_124m_100m_5seed", "eval_native_v1")}
SUPP = os.path.join(AM, "supplement_four_runs_20260923", "eval")


# ----------------------------------------------------------------------------- evaluation outputs
class NLL:
    """One evaluated checkpoint: per-block NLL and its summary (machine A or B, or a paper record)."""
    def __init__(self, arr, src, tokens=None, step=None, sha=None, meta=None):
        self.arr, self.src, self.tokens, self.step, self.sha, self.meta = arr, src, tokens, step, sha, meta or {}

    @property
    def mean(self):
        return float(self.arr.mean())


def _nll_dirs(machine, run, tag=""):
    out = []
    for d in sorted(glob.glob(os.path.join(EV, "nll", run, "*"))):
        name = os.path.basename(d)
        if ("__fwd" in name) != bool(tag) or (tag and not name.endswith(tag)):
            continue
        if json.load(open(os.path.join(d, "summary.json")))["host"] != HOST[machine]:
            continue
        out.append(d)
    return out


def _load_dir(d, machine):
    s = json.load(open(os.path.join(d, "summary.json")))
    prov = (s.get("provenance") or {}).get("checkpoint", {})
    return NLL(np.load(os.path.join(d, "per_block_nll.npy")), f"{machine}:nll/{os.path.basename(os.path.dirname(d))}/{os.path.basename(d)}",
               tokens=int(s["tokens_done"]), step=int(s["step"]), sha=prov.get("sha256"), meta=s)


def m_traj(machine, run, tag=""):
    """Every evaluated checkpoint of a run on one machine, keyed by step."""
    out = {}
    for d in _nll_dirs(machine, run, tag):
        x = _load_dir(d, machine)
        out[x.step] = x
    return dict(sorted(out.items()))


def m_end(machine, run, tag=""):
    t = m_traj(machine, run, tag)
    return t[max(t)] if t else None


def paper(suite, run, step):
    """The paper's per-block NLL of one checkpoint (suites A, B on S1; suite D on the RTX 3090 stack)."""
    g = glob.glob(os.path.join(SUITE[suite], run, f"*{step:06d}*", "per_block_nll.npy"))
    if not g and suite == "D" and step == 763:
        g = glob.glob(os.path.join(SUITE[suite], run, "100m_ckpt", "per_block_nll.npy"))
    if not g:
        return None
    sj = os.path.join(os.path.dirname(g[0]), "summary.json")
    meta = json.load(open(sj)) if os.path.exists(sj) else {}
    return NLL(np.load(g[0]), f"paper suite {suite}:{run}@{step}", step=step, tokens=step * BT, meta=meta)


def paper_traj(suite, run):
    out = {}
    for g in glob.glob(os.path.join(SUITE[suite], run, "*", "per_block_nll.npy")):
        m = re.search(r"ckpt_step(\d+)", g)
        if m:
            out[int(m.group(1))] = NLL(np.load(g), f"paper suite {suite}:{run}@{m.group(1)}", step=int(m.group(1)), tokens=int(m.group(1)) * BT)
    return dict(sorted(out.items()))


def supp(where, run):
    g = glob.glob(os.path.join(SUPP, where, run, "100m_*", "per_block_nll.npy"))
    return NLL(np.load(g[0]), f"paper supplement ({'RTX 3090' if where == 'rtx3090' else 'S1'}):{run}", step=763, tokens=763 * BT) if g else None


def probe(machine, run, step=None):
    out = {}
    for p in glob.glob(os.path.join(EV, "probe", run, "*", "probe.json")):
        d = json.load(open(p))
        if d["host"] == HOST[machine]:
            out[int(d["step"])] = d
    out = dict(sorted(out.items()))
    if step is None:
        return out
    return out.get(step)


# ----------------------------------------------------------------------------- statistics
def pd(a, b):
    """Paired a − b over the same validation blocks, circular moving-block bootstrap (L = 16, 4000, seed 20260918)."""
    r = st.paired(a.arr if isinstance(a, NLL) else a, b.arr if isinstance(b, NLL) else b, kind="blocks")
    return r["delta"], r["ci_lo"], r["ci_hi"]


def tint(xs):
    xs = np.asarray(xs, float)
    m, s = xs.mean(), xs.std(ddof=1)
    h = T975_2 * s / np.sqrt(len(xs))
    return float(m), float(m - h), float(m + h)


# ----------------------------------------------------------------------------- run metadata
def arg(run, name, default=None):
    m = re.search(rf"--{name}\s+(\S+)", MAN[run]["command"])
    return m.group(1) if m else default


def clean_command(cmd):
    """The training command with machine-specific locations replaced by neutral placeholders."""
    c = re.sub(r"^cd \S+ && \S+ ", "", cmd)
    c = re.sub(r"\$\{RB_DATA\}/", "<data>/", c)
    c = re.sub(r"\$\{RB_RUNS\}", "<runs>", c)
    c = re.sub(r"\$\{RB_PARENTS\}/", "<parents>/", c)
    c = re.sub(r"\$\{RB_CLOUD\}/\S*/", "<cloud>/", c)
    c = re.sub(r"\$\{RB_CODE\}/\.\./\.\./", "<training-tree>/", c)
    c = re.sub(r"\$\{RB_[A-Z0-9_]+\}", "<path>", c)
    c = re.sub(r"<CLOUD_DISK>/", "<paper-storage>/", c)
    c = re.sub(r"<POD_WORKSPACE>/", "<machine>/", c)
    return c


def run_log():
    """start / finished times per run (UTC epoch) on both machines."""
    ev = {}
    for machine, p in (("B", os.path.join(RB, "manifests", "rb-run_log_machine-B.jsonl")), ("A", os.path.join(RB, "manifests", "rb-run_log_machine-A.jsonl"))):
        for l in open(p):
            d = json.loads(l)
            if d.get("run_id") and d["event"] in ("start", "finished"):
                ev.setdefault(d["run_id"], {})[d["event"]] = d["time"]
                ev[d["run_id"]]["machine"] = machine
                if d["event"] == "start":
                    ev[d["run_id"]]["gpu"] = d.get("gpu")
    return ev


def ckpt_lists():
    """The machines' own checkpoint SHA-256 manifests (machine B: every checkpoint that existed; machine A)."""
    out = {}
    for machine, p in (("B", os.path.join(RB, "manifests", "ckpt_sha256_machine-B.tsv")),
                       ("A", os.path.join(RB, "manifests", "ckpt_sha256_machine-A.tsv"))):
        for r in csv.DictReader(open(p), delimiter="\t"):
            out[(r["run"], r["file"])] = dict(r, machine=machine)
    return out


def fmt(x, d=4, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def ci(t, d=4):
    return "—" if t is None else f"{t[0]:+.{d}f} [{t[1]:+.{d}f}, {t[2]:+.{d}f}]"

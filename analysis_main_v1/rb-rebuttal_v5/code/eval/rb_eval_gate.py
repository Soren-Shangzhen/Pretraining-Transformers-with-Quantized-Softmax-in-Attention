#!/usr/bin/env python3
"""Gate for evaluating machine A's runs on machine B (user decision 2026-11-12): the suite A seed-1337 softmax
endpoint, re-evaluated on B with its own frozen root, must match the recorded S1 value (|diff| <= 5e-4, the 13.2.2
rule; bitwise equality is reported separately).  Writes <EVAL>/gates/<name>.PASS.json or .FAIL.json; exit 1 on FAIL,
so the queue logs the unit as failed and every unit that requires the PASS file never starts.
usage: rb_eval_gate.py --name NAME --summary-dir DIR --recorded VALUE --recorded-source TEXT
"""
import argparse, glob, json, math, os, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--summary-dir", required=True)
    ap.add_argument("--recorded", type=float, required=True)
    ap.add_argument("--recorded-source", default="")
    ap.add_argument("--tol", type=float, default=5e-4)
    a = ap.parse_args()
    s = glob.glob(os.path.join(a.summary_dir, "*", "summary.json"))
    got = json.load(open(s[0]))["mean_nll"] if len(s) == 1 else float("nan")
    d = got - a.recorded
    ok = math.isfinite(got) and abs(d) <= a.tol
    res = dict(name=a.name, got=got, recorded=a.recorded, recorded_source=a.recorded_source, diff=d, tol=a.tol,
               bitwise_equal=got == a.recorded, passed=ok, summary=s)
    gd = os.path.join(os.environ["RB_RUNS"], "_eval", "gates")
    os.makedirs(gd, exist_ok=True)
    json.dump(res, open(os.path.join(gd, f"{a.name}.{'PASS' if ok else 'FAIL'}.json"), "w"), indent=1)
    print(json.dumps(res))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

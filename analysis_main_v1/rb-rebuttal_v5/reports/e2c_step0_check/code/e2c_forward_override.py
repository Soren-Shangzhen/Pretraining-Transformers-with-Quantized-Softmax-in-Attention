#!/usr/bin/env python3
"""Validation NLL of a parent checkpoint under the frozen-step forward of its branch, weights unchanged.

Derived from code/eval/rb_eval_forward.py (the forward-override wrapper used for the E3a-2.5B forward swaps), which is
left unchanged.  rb_eval.py runs unchanged as well (976 FineWebEdu blocks, batch 1, TF32 off, bf16 autocast outside
the fp32 operator, per-block NLL, provenance with the checkpoint SHA-256 read at evaluation time).  Two differences,
both needed because the three parents are historical MinMax-Weight checkpoints written by an older code root:

  1. rb_eval builds a historical checkpoint (no tail_policy in its config) with common.build_model, so the hook on
     rb_eval.build_rb_model that rb_eval_forward.py installs would never run.  Here both builders are routed to
     rb_eval.build_rb_model (the rb-root_v1 constructor the branch runs used), with the checkpoint config completed
     by the rb-root_v1 defaults for the fields an old config lacks and then overridden by --set.
  2. The frozen-step operator needs resolved_h0 (per-layer constant step), tail_policy and range_tau, which are
     added to the fields that may be overridden.  The values come from the branch run's own checkpoint config.

--set with the parent's own operator (range_mode=minmax) is the control: the parent through rb-root_v1 with its
native MinMax forward.  forward_override.json next to summary.json records the override, the forward actually
built (per-layer window K*h0 read back from the built module) and this script's SHA-256.
usage: e2c_forward_override.py --set k=v [--set ...] --tag TAG --jobs J --out O --root <rb-root_v1>
"""
import argparse, csv, hashlib, json, os, sys
from pathlib import Path

RB = Path(__file__).resolve().parents[3]                       # analysis_main_v1/rb-rebuttal_v5
sys.path.insert(0, str(RB / "code" / "eval"))
import rb_eval  # noqa: E402

DEFAULTS = {"range_mode": "minmax", "range_tau": 6.0, "tail_policy": "zero", "resolved_h0": "", "attn_variant": "",
            "exaq_bits": 3, "ema_decay": 0.999}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True, help="config field=value (operator fields only)")
    ap.add_argument("--tag", required=True)
    a, rest = ap.parse_known_args()
    over = dict(s.split("=", 1) for s in a.set)
    allowed = {"attn", "attn_variant", "K", "approx", "range_mode", "range_tau", "range_grad", "tail_policy",
               "exaq_bits", "resolved_h0", "nearest_surrogate"}
    if set(over) - allowed:
        raise SystemExit(f"only operator fields may be overridden: {sorted(set(over) - allowed)}")
    orig, built = rb_eval.build_rb_model, []

    def patched(mm, ck, device):
        c = dict(ck["config"])
        for k, v in DEFAULTS.items():
            if c.get(k) is None:
                c[k] = v
        c.update(over)
        model, cfg, op = orig(mm, {**ck, "config": c}, device)
        att = model.transformer.h[0].attn
        taus = [getattr(blk.attn, "range_tau", None) for blk in model.transformer.h]
        built.append(dict(checkpoint_config={k: ck["config"].get(k) for k in over}, forward_config=op,
                          module_mode=att.mode, module_range_mode=getattr(att, "range_mode", None),
                          module_tail_policy=getattr(att, "tail_policy", None),
                          module_range_tau_per_layer=[float(x) if x is not None else None for x in taus]))
        return model, cfg, op

    rb_eval.build_rb_model = patched
    rb_eval.build_model = patched                               # historical checkpoints take this path in rb_eval
    sys.argv = [rb_eval.__file__] + rest + ["--tag", a.tag]
    rb_eval.main()
    p = argparse.ArgumentParser(); p.add_argument("--jobs"); p.add_argument("--out"); p.add_argument("--root")
    r, _ = p.parse_known_args(rest)
    me = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    jobs = [j for j in csv.DictReader(open(r.jobs), delimiter="\t") if j["eval_root"] == r.root]
    for j, b in zip(jobs, built):
        out = Path(r.out) / j["label"] / f"{int(round(float(j['tokens_m']))):04d}m_{Path(j['checkpoint']).stem}{a.tag}"
        if out.is_dir() and not (out / "forward_override.json").exists():
            (out / "forward_override.json").write_text(json.dumps(dict(override=over, **b, script=os.path.abspath(__file__),
                                                                        script_sha256=me, weights="unchanged"), indent=1))


if __name__ == "__main__":
    main()

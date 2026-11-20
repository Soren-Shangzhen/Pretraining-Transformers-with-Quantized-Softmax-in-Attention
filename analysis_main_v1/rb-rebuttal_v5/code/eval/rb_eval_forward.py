#!/usr/bin/env python3
"""Validation NLL of a checkpoint under a different attention forward, weights unchanged (user request 2026-11-14,
item 3: the E3a-2.5B endpoint with the E3a-i forward, IndexSoftmax with floating-point normalisation, and with exact
softmax).  rb_eval.py runs unchanged; the only difference is that the checkpoint config handed to
rb_eval.build_rb_model has the operator fields given by --set replaced (the weights, the 976 FineWebEdu blocks, batch 1,
TF32 off, bf16 autocast and the provenance block with the checkpoint SHA-256 read at evaluation time are rb_eval.py's).
The output directory carries --tag; forward_override.json next to summary.json records the override, the forward
actually built and this script's SHA-256.  --set with the checkpoint's own values is the control: it must reproduce
the native result bit for bit.
usage: rb_eval_forward.py --set attn_variant=fp --tag __fwd-E3a-i --jobs J --out O --root R
"""
import argparse, csv, hashlib, json, os, sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rb_eval  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True, help="config field=value (operator fields only)")
    ap.add_argument("--tag", required=True)
    a, rest = ap.parse_known_args()
    over = dict(s.split("=", 1) for s in a.set)
    allowed = {"attn", "attn_variant", "K", "approx", "range_mode", "range_tau", "range_grad", "tail_policy", "exaq_bits"}
    if set(over) - allowed:
        raise SystemExit(f"only operator fields may be overridden: {sorted(set(over) - allowed)}")
    orig, built = rb_eval.build_rb_model, []

    def patched(mm, ck, device):
        ck2 = dict(ck)
        ck2["config"] = {**ck["config"], **over}
        model, cfg, op = orig(mm, ck2, device)
        built.append(dict(checkpoint_config={k: ck["config"].get(k) for k in over}, forward_config=op,
                          module_mode=model.transformer.h[0].attn.mode, module_variant=getattr(model.transformer.h[0].attn, "variant", None)))
        return model, cfg, op

    rb_eval.build_rb_model = patched
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

#!/usr/bin/env python3
"""Addendum analysis (manifests/addendum_nearest_fwd_centroid_bwd.json): forward/backward decomposition of the
FW-Centroid K4 training gap with the hard-Nearest-forward + Centroid-backward run (NfCb).

  B = NfCb - FWM-Prob K4 strict      (backward effect: same strict Nearest forward, Centroid vs LERP Jacobian)
  F = FW-Centroid - NfCb             (forward effect: same Centroid backward, Centroid vs Nearest forward)
  total = FW-Centroid - FWM-Prob K4 strict = B + F;   backward share = B / total
Circular moving-block bootstrap (block 16, 4000 resamples, RNG 20260920), the same resample indices for every
per-block array so that B, F, total and the share are jointly resampled.  Output: results/addendum/addendum.json
"""
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_training as at  # noqa: E402

C = at.C
OUT = os.path.join(C, "results", "addendum")
RUN = "cv1b_s7_fw_nearestfwd_centroidbwd_k4_tau6_100m"
at.SER["NfCb (Nearest fwd + Centroid bwd)"] = f"{C}/eval/rtx3090/{RUN}"
at.LOGS["NfCb (Nearest fwd + Centroid bwd)"] = f"{at.PB}/centroid_runs_20261013/{RUN}/log.csv"
NF = "NfCb (Nearest fwd + Centroid bwd)"


def idx_blocks(n, B=4000, blk=16, seed=20260920):
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / blk))
    st = rng.integers(0, n, size=(B, nb))
    return (st[:, :, None] + np.arange(blk)[None, None, :]).reshape(B, -1)[:, :n] % n


def main():
    os.makedirs(OUT, exist_ok=True)
    nf, fc, ps = at.pb(NF), at.pb("FW-Centroid K4"), at.pb("FWM-Prob K4 strict")
    if nf is None:
        raise SystemExit("addendum run not evaluated yet")
    idx = idx_blocks(len(nf))
    Bv, Fv, Tv = nf - ps, fc - nf, fc - ps
    bb, fb, tb = Bv[idx].mean(1), Fv[idx].mean(1), Tv[idx].mean(1)
    share = Bv.mean() / Tv.mean()
    share_b = bb / tb

    def ci(x):
        return [float(v) for v in np.percentile(x, [2.5, 97.5])]

    res = dict(
        B_backward_effect=dict(delta=float(Bv.mean()), ci95=ci(bb)),
        F_forward_effect=dict(delta=float(Fv.mean()), ci95=ci(fb)),
        total=dict(delta=float(Tv.mean()), ci95=ci(tb)),
        backward_share=dict(value=float(share), ci95=ci(share_b)))
    b_excl = res["B_backward_effect"]["ci95"][0] > 0 or res["B_backward_effect"]["ci95"][1] < 0
    f_excl = res["F_forward_effect"]["ci95"][0] > 0 or res["F_forward_effect"]["ci95"][1] < 0
    res["verdict"] = ("backward_dominated" if (b_excl and share >= 0.75) else
                      "forward_dominated" if (f_excl and share <= 0.25) else "mixed")
    contrasts = {}
    for b in ("softmax", "FWM-Prob K4 strict", "FWM-Prob K4 (hist)", "FWM-Weight K4 (hist)", "FW-Centroid K4",
              "LERP K4"):
        m, lo, hi = at.boot(nf - at.pb(b))
        contrasts[f"NfCb - {b}"] = dict(delta=m, ci95=[lo, hi], label=at.label(m, lo, hi))
    res["contrasts_100m"] = contrasts
    res["trajectory_vs_softmax"] = {
        k: {t: float(at.pb(k, t).mean() - at.pb("softmax", t).mean()) for t in at.T if at.pb(k, t) is not None}
        for k in (NF, "FW-Centroid K4", "FWM-Prob K4 (hist)", "FWM-Weight K4 (hist)")}
    res["abs_nll"] = {t: (float(at.pb(NF, t).mean()) if at.pb(NF, t) is not None else None) for t in at.T}
    res["geometry_100m"] = {k: at.geom(k) for k in (NF, "FW-Centroid K4", "FWM-Prob K4 (hist)", "softmax")}
    res["log"] = at.logstats(NF)
    json.dump(res, open(os.path.join(OUT, "addendum.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("B_backward_effect", "F_forward_effect", "total", "backward_share",
                                           "verdict")}, indent=1))
    for k, v in contrasts.items():
        print(f"{k:40s} {v['delta']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] {v['label']}")
    for k, v in res["trajectory_vs_softmax"].items():
        print(k, {t: round(x, 4) for t, x in v.items()})


if __name__ == "__main__":
    main()

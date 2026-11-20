#!/usr/bin/env python3
"""Stage B: pre-registered power screen (§1.7) over the nine Stage-A tasks.

Read-only. Consumes Stage-A Exact-only summaries and the per-task character/token conversion, and assigns
each task to one of three bands. No condition-vs-Exact comparison is made here.

Bands (as specified):
  primary    the detection floor can carry good-band magnitudes   (LM NLL gap < 0.01 nats)
  reference  the detection floor can carry only pathological-band magnitudes (0.13-0.89 nats)
  unusable   Exact does not score above the chance baseline computed from the actual option count

Units. Stage A's continuous metric for multiple-choice tasks is a CHARACTER-normalised log-likelihood margin
(the harness acc_norm convention: log-likelihood divided by the continuation's character length). To compare a
floor against a per-token LM NLL target it is multiplied by the task's measured characters-per-token. LAMBADA's
metric is the summed target NLL, so it is divided by the mean target token count instead.

Transfer coefficient. A uniform per-token LM NLL shift of delta cancels in a gold-minus-best-wrong margin to the
extent that it is common to both options; only the differential part survives. The floors below are therefore
compared against the raw target magnitudes as an UPPER BOUND on sensitivity (kappa = 1). The realised kappa is
unknown before Stage C and is expected to be < 1 for multiple-choice margins and = 1 for LAMBADA target NLL.

Pairing. The declared floor 2*SD/sqrt(n) corresponds to a paired SD equal to the marginal SD, i.e. a
condition-Exact correlation of rho = 0.5. Good-band conditions will correlate far more strongly, so realised
paired floors will be smaller; Stage C must report the realised paired SD rather than this bound.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)          # downstream_v1, wherever this copy lives

SA = f"{BASE}/results/layer3/stageA"
OUT = f"{BASE}/results/layer3/stageB"
GOOD_BAND = 0.01            # LM NLL gap of the good tier (nats/token)
PATHO_LO, PATHO_HI = 0.13, 0.89
THROUGHPUT = 88.0           # scored requests per second, measured in Stage A

# Paired shrinkage s = SD(cond - Exact) / SD(Exact), measured on data already in hand
# (layer3_stageB_pairing.py). The pre-registered floor is s = 1, i.e. rho = 0.5. On natural text
# (layer 1 per-block NLL) the realised medians are far smaller, so the floor is reported at three
# declared values and the band is assigned twice: as pre-registered, and under the natural-text
# calibration. Stage C must replace this with the realised paired SD.
SHRINK = {"declared (s=1, rho=0.5)": 1.00,
          "natural text, pathological tier (s=0.20)": 0.20,
          "natural text, good tier (s=0.09)": 0.09}


def load():
    rows = [json.load(open(f"{SA}/{f}")) for f in sorted(os.listdir(SA))
            if f.endswith("_summary.json") and not f.startswith("stageA")]
    ct = json.load(open(f"{SA}/chars_per_token.json"))
    d = pd.DataFrame(rows)
    d["chars_per_token"] = [ct[t]["chars_per_token"] for t in d.task]
    d["mean_cont_tokens"] = [ct[t]["mean_tokens"] for t in d.task]
    return d


def screen(d):
    # convert the floor into per-token LM-NLL-equivalent nats
    conv = np.where(d.task == "lambada_openai", 1.0 / d.mean_cont_tokens, d.chars_per_token)
    d["floor_nats_per_token"] = d.detect_floor_metric * conv
    d["metric_sd_nats_per_token"] = d.metric_sd * conv
    d["unit_conversion"] = np.where(d.task == "lambada_openai",
                                    "summed target NLL / " + d.mean_cont_tokens.round(3).astype(str) + " tok",
                                    "char-normalised margin x " + d.chars_per_token.round(3).astype(str) + " chars/tok")

    # above-chance test on accuracy (one-sample, single model)
    d["acc_z_vs_chance"] = (d.acc - d.chance) / d.acc_se
    d["above_chance"] = d.acc_z_vs_chance > 2.0

    band, why = [], []
    for r in d.itertuples():
        if not r.above_chance:
            band.append("unusable")
            why.append(f"Exact acc {r.acc:.4f} is not above the chance baseline {r.chance:.3f} "
                       f"(z = {r.acc_z_vs_chance:+.1f}); the task carries no usable signal in this model")
        elif r.floor_nats_per_token <= GOOD_BAND:
            band.append("primary")
            why.append(f"floor {r.floor_nats_per_token:.4f} nats/token <= {GOOD_BAND}, so a good-band gap is "
                       f"within reach at kappa = 1")
        elif r.floor_nats_per_token <= PATHO_LO:
            band.append("reference")
            why.append(f"floor {r.floor_nats_per_token:.4f} nats/token is above {GOOD_BAND} but below the "
                       f"pathological band lower edge {PATHO_LO}")
        else:
            band.append("reference (lower edge only)")
            why.append(f"floor {r.floor_nats_per_token:.4f} nats/token exceeds the pathological lower edge "
                       f"{PATHO_LO}; only gaps toward {PATHO_HI} are resolvable")
    d["band"] = band
    d["band_reason"] = why

    # sensitivity of the floor to the paired shrinkage, and the band under the natural-text calibration
    for name, sv in SHRINK.items():
        d[f"floor_s{sv:.2f}"] = d.floor_nats_per_token * sv
    cal = []
    for r in d.itertuples():
        f = r.floor_nats_per_token * SHRINK["natural text, pathological tier (s=0.20)"]
        if not r.above_chance:
            cal.append("unusable")
        elif f <= GOOD_BAND:
            cal.append("primary")
        elif f <= PATHO_LO:
            cal.append("reference")
        else:
            cal.append("reference (lower edge only)")
    d["band_calibrated"] = cal

    # accuracy channel: what an 0.01-nat gap could move is far below every accuracy floor
    d["acc_channel"] = np.where(d.above_chance, "pathological band only", "not usable")
    d["stageC_seconds_per_condition"] = (d.n_requests / THROUGHPUT).round(0)
    return d


def main():
    os.makedirs(OUT, exist_ok=True)
    d = screen(load()).sort_values(["band_calibrated", "floor_nats_per_token"])
    cols = ["task", "subset", "n_examples", "n_requests", "mean_options", "chance", "acc", "acc_norm",
            "acc_z_vs_chance", "acc_se", "detect_floor_acc_pp", "metric_name", "metric_mean", "metric_sd",
            "metric_se", "detect_floor_metric", "unit_conversion", "floor_nats_per_token",
            "truncation_rate", "band", "band_calibrated", "band_reason",
            "floor_s1.00", "floor_s0.20", "floor_s0.09", "acc_channel",
            "stageC_seconds_per_condition"]
    d[cols].to_csv(f"{OUT}/stageB_power_screen.csv", index=False)
    show = ["task", "n_examples", "chance", "acc", "acc_z_vs_chance", "acc_se", "detect_floor_acc_pp",
            "metric_sd", "detect_floor_metric", "floor_s1.00", "floor_s0.20", "floor_s0.09",
            "band", "band_calibrated"]
    print(d[show].round(4).to_string(index=False))
    tot = d.stageC_seconds_per_condition.sum()
    print(f"\nStage C cost at {THROUGHPUT:.0f} req/s: {tot/3600:.2f} h per condition-arm; "
          f"15 conditions x native = {15*tot/3600:.1f} h; adding an exact-forward arm doubles it.")
    for col, title in [("band", "as pre-registered (s = 1)"), ("band_calibrated", "natural-text calibrated (s = 0.20)")]:
        print(f"\n--- bands {title} ---")
        for b in ["primary", "reference", "reference (lower edge only)", "unusable"]:
            sub = d[d[col] == b]
            if len(sub):
                print(f"  {b}: {', '.join(sub.task)}")
    json.dump(d[cols].to_dict("records"), open(f"{OUT}/stageB_power_screen.json", "w"), indent=2)


if __name__ == "__main__":
    main()

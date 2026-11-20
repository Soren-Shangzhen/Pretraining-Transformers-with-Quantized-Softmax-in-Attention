#!/usr/bin/env python3
"""Round-2 Task 2, fixed-upstream (per-layer) view of the score-gradient decomposition.

In the full model (E1 root, range_grad=full) each layer's score tensor `att` and the upstream
u = dL/dP (captured with a read-only hook on the normalized weights P) are recorded on the first 8 WikiText-103
validation blocks (loss = mean NLL, loss_b / 8 per block).  For every layer the local VJP of the layer's own
operator P = w/sum(w), w = _adaptive_interval_reconstruct(att, ..., range_grad=mode) is then computed with that
fixed u for mode in {full, detach, min_only, max_only}; P1 local = row projection of the detach VJP.  With a
fixed upstream the decomposition is exact by construction:

    g_full = self term (detach VJP, all keys)  +  M term (max_only - detach, on argmax keys)
                                               +  m term (min_only - detach, on argmin keys)

The local full VJP is checked bit for bit against the in-model score gradient of the same layer.
Deterministic algorithms on, bf16 autocast outside fp32 attention, TF32 off.
"""
import csv, json, os, sys
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["SUP_LOCAL"] = "1"
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SUP = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, f"{SUP}/code/checks")
sys.path.insert(0, f"{SUP}/code/eval")
sys.path.insert(0, HERE)
from sup_common import LocalsCapture, compare, env_record, load_root, root_hashes, ROOTS, sha256_file  # noqa: E402
from common import ValBlocks  # noqa: E402
from grad_decomp import CKPTS, NB, RUN, SEED, VAL, build, project_p1  # noqa: E402

OUT = f"{SUP}/round2/task2" + ("" if SEED == 7 else f"_s{SEED}")
LMODES = ["full", "detach", "min_only", "max_only"]
CMP = ["detach", "min_only", "max_only", "p1_local"]


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    blocks = ValBlocks(VAL, 1024)
    T = 1024
    causal = torch.ones(T, T, dtype=torch.bool, device="cuda").tril()
    valid = causal.view(1, 1, T, T)
    qa, _, _ = load_root("new_e1")
    report = {"script": os.path.abspath(__file__), "script_sha256": sha256_file(__file__), "env": env_record(),
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "checkpoints": {}}
    rows_out = []
    for tok, name in CKPTS:
        ck = torch.load(f"{RUN}/{name}", map_location="cpu", weights_only=False, mmap=True)
        assert ck["config"]["seed"] == SEED and ck["config"]["range_grad"] == "full" and ck["config"]["K"] == 32
        os.makedirs(OUT, exist_ok=True)
        model = build("new_e1", "full", ck["model"])
        local_full_eq = {"all_bitwise": True, "mismatch": 0, "max_abs": 0.0}
        acc = {l: {m: {"d2": 0.0, "d2_argmax": 0.0, "d2_argmin": 0.0, "rows": []} for m in CMP}
               | {"full2": 0.0, "self2": 0.0, "M2": 0.0, "m2": 0.0, "resid2": 0.0, "M_on_argmax": 0.0, "m_on_argmin": 0.0,
                  "rowratio_M": [], "rowratio_m": [], "rowratio_self": []} for l in range(12)}
        for b in range(NB):
            x, y = blocks.batch(b, b + 1, "cuda")
            atts, ups, sgs = [], [], []

            def cap(fn, d):
                atts.append(d["att"].detach())
                d["P"].register_hook(lambda g: ups.append(g.detach().clone()) and None)
                d["att"].register_hook(lambda g: sgs.append(g.detach().clone()) and None)
            model.zero_grad(set_to_none=True)
            with LocalsCapture(["_quant_attention"], ["att", "P"], cap), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
            loss = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))
            (loss / NB).backward()
            ups, sgs = list(reversed(ups)), list(reversed(sgs))
            for l in range(12):
                g = {}
                for mode in LMODES:
                    a = atts[l].clone().requires_grad_(True)
                    w = qa._adaptive_interval_reconstruct(a, causal, 32, "lerp", range_grad=mode)[0]
                    P = w / w.sum(dim=-1, keepdim=True)
                    P.backward(ups[l])
                    g[mode] = a.grad.detach()
                c = compare(g["full"], sgs[l])
                local_full_eq["all_bitwise"] &= c["eq_rawbits"]
                local_full_eq["mismatch"] += c["mismatch_count"]
                local_full_eq["max_abs"] = max(local_full_eq["max_abs"], c["max_abs"])
                g["p1_local"] = project_p1(g["detach"], valid)
                att = atts[l]
                is_max = valid & (att == att.masked_fill(~valid, -torch.inf).amax(-1, keepdim=True))
                is_min = valid & (att == att.masked_fill(~valid, torch.inf).amin(-1, keepdim=True)) & ~is_max
                gfv = g["full"].masked_fill(~valid, 0.0).double()
                nf_row = gfv.pow(2).sum(-1).sqrt()
                keep = nf_row > 0
                A = acc[l]
                A["full2"] += float(gfv.pow(2).sum())
                for m in CMP:
                    d2 = (g[m].masked_fill(~valid, 0.0).double() - gfv).pow(2)
                    A[m]["d2"] += float(d2.sum())
                    A[m]["d2_argmax"] += float(d2[is_max.expand_as(d2)].sum())
                    A[m]["d2_argmin"] += float(d2[is_min.expand_as(d2)].sum())
                    A[m]["rows"].append((d2.sum(-1).sqrt() / nf_row.clamp_min(1e-300))[keep].float().cpu())
                s_t = g["detach"].masked_fill(~valid, 0.0).double()
                M_t = g["max_only"].masked_fill(~valid, 0.0).double() - s_t
                m_t = g["min_only"].masked_fill(~valid, 0.0).double() - s_t
                A["self2"] += float(s_t.pow(2).sum())
                A["M2"] += float(M_t.pow(2).sum())
                A["m2"] += float(m_t.pow(2).sum())
                A["resid2"] += float((gfv - s_t - M_t - m_t).pow(2).sum())
                A["M_on_argmax"] += float(M_t.pow(2)[is_max.expand_as(M_t)].sum())
                A["m_on_argmin"] += float(m_t.pow(2)[is_min.expand_as(m_t)].sum())
                A["rowratio_M"].append((M_t.pow(2).sum(-1).sqrt() / nf_row.clamp_min(1e-300))[keep].float().cpu())
                A["rowratio_m"].append((m_t.pow(2).sum(-1).sqrt() / nf_row.clamp_min(1e-300))[keep].float().cpu())
                A["rowratio_self"].append((s_t.pow(2).sum(-1).sqrt() / nf_row.clamp_min(1e-300))[keep].float().cpu())
                del g, s_t, M_t, m_t, gfv
            del atts, ups, sgs
            torch.cuda.empty_cache()
        rec = {"checkpoint": f"{RUN}/{name}", "local_full_vjp_vs_in_model_score_grad": local_full_eq, "layers": {}}
        for l in range(12):
            A = acc[l]
            nf = np.sqrt(A["full2"])
            L = {"mode_error": {m: {"row_median_rel": float(np.median(torch.cat(A[m]["rows"]).double().numpy())),
                                    "aggregate_rel": float(np.sqrt(A[m]["d2"] / A["full2"])),
                                    "frac_err2_argmax": A[m]["d2_argmax"] / A[m]["d2"],
                                    "frac_err2_argmin": A[m]["d2_argmin"] / A[m]["d2"],
                                    "frac_err2_other": 1 - (A[m]["d2_argmax"] + A[m]["d2_argmin"]) / A[m]["d2"]} for m in CMP},
                 "full_decomposition": {"norm_full": float(nf), "norm_self": float(np.sqrt(A["self2"])),
                                        "norm_M": float(np.sqrt(A["M2"])), "norm_m": float(np.sqrt(A["m2"])),
                                        "ratio_self": float(np.sqrt(A["self2"]) / nf), "ratio_M": float(np.sqrt(A["M2"]) / nf),
                                        "ratio_m": float(np.sqrt(A["m2"]) / nf), "resid_rel": float(np.sqrt(A["resid2"]) / nf),
                                        "M_mass_on_argmax": A["M_on_argmax"] / A["M2"], "m_mass_on_argmin": A["m_on_argmin"] / A["m2"],
                                        "row_median_ratio_self": float(np.median(torch.cat(A["rowratio_self"]).numpy())),
                                        "row_median_ratio_M": float(np.median(torch.cat(A["rowratio_M"]).numpy())),
                                        "row_median_ratio_m": float(np.median(torch.cat(A["rowratio_m"]).numpy()))}}
            rec["layers"][l] = L
            row = {"tokens_m": tok, "layer": l}
            for m, v in L["mode_error"].items():
                row.update({f"{m}__{k}": x for k, x in v.items()})
            row.update({f"decomp__{k}": x for k, x in L["full_decomposition"].items()})
            rows_out.append(row)
        report["checkpoints"][f"{tok}M"] = rec
        print(tok, "local full == in-model:", local_full_eq, flush=True)
        del model
        torch.cuda.empty_cache()
    json.dump(report, open(f"{OUT}/grad_decomp_local.json", "w"), indent=1, default=str)
    with open(f"{OUT}/grad_decomp_local_layers.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

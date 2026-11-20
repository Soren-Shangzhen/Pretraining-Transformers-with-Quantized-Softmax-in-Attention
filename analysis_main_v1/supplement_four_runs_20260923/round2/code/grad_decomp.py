#!/usr/bin/env python3
"""Round-2 Task 2: score-gradient decomposition at the seed-7 LERP K32 full checkpoints (no training).

Checkpoints : p3_s7_lerp_k32_fullgrad_100m at 20M and 40M (local).
Inputs      : the first 8 blocks of the WikiText-103 validation split (1024 tokens each).
Loss        : mean NLL over the 8 blocks.  Each block is one forward/backward with loss_b / 8 (the 8-block
              batch does not fit on the 24 GB card); summed over blocks this is the gradient of the mean NLL.
              No optimizer step.
Modes       : full, detach, min_only, max_only  -> E1 root (code/e1_lerp_range_grad_single), range_grad flag.
              P1 (historical)                   -> frozen phaseB root, range_grad="detach_proj" (the code of run
                                                   q32_lerp_detachproj_100m; hook on the score tensor).
              P1 (new, definition, in-graph)    -> E1 root in detach mode plus a hook on the score tensor that
                                                   applies g'_ij = g_ij - (1/(i+1)) sum_{k<=i} g_ik (masked positions
                                                   untouched) inside the backward, where the historical code applies
                                                   it; compared bit for bit with P1 (historical).
              P1 (local)                        -> the same projection applied after the fact to the detach
                                                   score-gradient (fixed detach upstream).  NOT the training rule:
                                                   in-graph, projected gradients of later layers change the upstream
                                                   of earlier layers.  Reported separately.
Memory      : one model per code root; the mode is switched through each layer's `range_grad` attribute (read
              by the frozen code at forward time); per-mode parameter gradients are accumulated on CPU.
              All modes load the same checkpoint weights; forward identity is checked bit for bit.
Capture     : dL/ds on each layer's scaled score tensor `att` (before calibration / detach), via a
              read-only hook registered after the forward.  For historical P1 the capture hook runs after the
              projection hook, i.e. it sees the projected gradient.
Precision   : bf16 autocast outside the fp32 attention kernel, TF32 off (evaluation protocol);
              torch.use_deterministic_algorithms(True) in this diagnostic process (allowed by the task).
"""
import csv, json, os, sys, time
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["SUP_LOCAL"] = "1"
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SUP = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, f"{SUP}/code/checks")
sys.path.insert(0, f"{SUP}/code/eval")
from sup_common import LocalsCapture, compare, env_record, load_root, root_hashes, ROOTS, sha256_file  # noqa: E402
from common import ValBlocks  # noqa: E402
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository


TR = KI_REPO + ""
PB = f"{TR}/training_code/phaseB_base"
VAL = f"{PB}/data/wikitext103/val.bin"
SEED = int(os.environ.get("SEED", "7"))
RUN = f"{PB}/runs/p3_s{SEED}_lerp_k32_fullgrad_100m"
CKPTS = [(20, "ckpt_step000153_20.1m.pt"), (40, "ckpt_step000306_40.1m.pt")]
NB = 8
MODES = [("full", "e1", "full"), ("full_repeat", "e1", "full"), ("detach", "e1", "detach"), ("min_only", "e1", "min_only"),
         ("max_only", "e1", "max_only"), ("p1_new", "e1", "detach"), ("p1_hist", "old", "detach_proj")]
CMP = ["detach", "min_only", "max_only", "p1_hist", "p1_new", "p1_local"]
OUT = f"{SUP}/round2/task2" + ("" if SEED == 7 else f"_s{SEED}")


def project_p1(g, valid):
    """P1 by its definition (row mean over causal-valid keys; masked positions untouched)."""
    gv = g.masked_fill(~valid, 0.0)
    n = valid.sum(dim=-1, keepdim=True).to(g.dtype)
    return torch.where(valid, g - gv.sum(dim=-1, keepdim=True) / n, g)


def build(key, mode, sd):
    qa, mm, _ = load_root(key)
    cfg = mm.GPTConfig(block_size=1024, n_layer=12, n_head=12, n_embd=768, attn_mode="quant", attn_K=32,
                       attn_approx="lerp", attn_range_grad=mode)
    m = mm.GPT(cfg)
    m.load_state_dict(sd)
    m.to("cuda").eval()
    m.grad_checkpointing = False
    return m


def set_mode(model, mode):
    for blk in model.transformer.h:
        blk.attn.range_grad = mode


def run_block(model, x, y, keep_att=False, p1_graph=False, valid=None):
    grads, atts = [], []

    def cap(fn, d):
        a = d.get("att")
        if a is not None:
            if keep_att:
                atts.append(a.detach())
            if p1_graph:
                a.register_hook(lambda g: project_p1(g, valid))
            a.register_hook(lambda g: grads.append(g.detach().clone()) and None)
    model.zero_grad(set_to_none=True)
    with LocalsCapture(["_quant_attention"], ["att"], cap), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, _ = model(x)
    loss = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))
    (loss / NB).backward()
    pgrad = {n: p.grad.detach().to("cpu", copy=True) for n, p in model.named_parameters()}
    return logits.detach(), loss.detach(), atts, list(reversed(grads)), pgrad


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    os.makedirs(OUT, exist_ok=True)
    blocks = ValBlocks(VAL, 1024)
    T = 1024
    valid = torch.ones(T, T, dtype=torch.bool, device="cuda").tril().view(1, 1, T, T)
    report = {"script": os.path.abspath(__file__), "script_sha256": sha256_file(__file__), "env": env_record(),
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "precision": "bf16 autocast outside fp32 attention, TF32 off, eval mode (no dropout exists)",
              "roots": {k: {"path": ROOTS[k], "sha256": root_hashes(ROOTS[k])} for k in ("new_e1", "old_phaseB_base")},
              "val_sha256": sha256_file(VAL), "checkpoints": {}}
    layer_rows = []
    for tok, name in CKPTS:
        path = f"{RUN}/{name}"
        ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        assert ck["config"]["seed"] == SEED and ck["config"]["range_grad"] == "full" and ck["config"]["K"] == 32
        sd = ck["model"]
        models = {"e1": build("new_e1", "full", sd), "old": build("old_phaseB_base", "detach_proj", sd)}
        fwd_eq = {m: {"logits": True, "loss": True} for m, _, _ in MODES if m != "full"}
        p1_eq = {"all_bitwise": True, "mismatch": 0, "max_abs": 0.0}
        rep_eq = True
        pacc = {m: None for m, _, _ in MODES}
        acc = {l: {m: {"d2": 0.0, "d2_argmax": 0.0, "d2_argmin": 0.0, "rows": []} for m in CMP}
               | {"full2": 0.0, "self2": 0.0, "M2": 0.0, "m2": 0.0, "resid2": 0.0, "M_on_argmax": 0.0, "m_on_argmin": 0.0,
                  "rowratio_M": [], "rowratio_m": [], "rowratio_self": []} for l in range(12)}
        for b in range(NB):
            x, y = blocks.batch(b, b + 1, "cuda")
            res = {}
            for m, root, mode in MODES:
                set_mode(models[root], mode)
                lg, ls, atts, sg, pg = run_block(models[root], x, y, keep_att=(m == "full"), p1_graph=(m == "p1_new"), valid=valid)
                res[m] = (lg, ls, atts, sg)
                pacc[m] = pg if pacc[m] is None else {n: pacc[m][n] + pg[n] for n in pg}
            for m in fwd_eq:
                fwd_eq[m]["logits"] &= compare(res[m][0], res["full"][0])["eq_rawbits"]
                fwd_eq[m]["loss"] &= compare(res[m][1], res["full"][1])["eq_rawbits"]
            for l in range(12):
                att = res["full"][2][l]
                gf = res["full"][3][l]
                rep_eq &= compare(res["full_repeat"][3][l], gf)["eq_rawbits"]
                gd = res["detach"][3][l]
                c = compare(res["p1_new"][3][l], res["p1_hist"][3][l])
                p1_eq["all_bitwise"] &= c["eq_rawbits"]
                p1_eq["mismatch"] += c["mismatch_count"]
                p1_eq["max_abs"] = max(p1_eq["max_abs"], c["max_abs"])
                is_max = valid & (att == att.masked_fill(~valid, -torch.inf).amax(-1, keepdim=True))
                is_min = valid & (att == att.masked_fill(~valid, torch.inf).amin(-1, keepdim=True)) & ~is_max
                gfv = gf.masked_fill(~valid, 0.0).double()
                nf_row = gfv.pow(2).sum(-1).sqrt()
                keep = nf_row > 0
                A = acc[l]
                A["full2"] += float(gfv.pow(2).sum())
                gm = {"detach": gd, "min_only": res["min_only"][3][l], "max_only": res["max_only"][3][l],
                      "p1_hist": res["p1_hist"][3][l], "p1_new": res["p1_new"][3][l], "p1_local": project_p1(gd, valid)}
                for m, g in gm.items():
                    d2 = (g.masked_fill(~valid, 0.0).double() - gfv).pow(2)
                    A[m]["d2"] += float(d2.sum())
                    A[m]["d2_argmax"] += float(d2[is_max.expand_as(d2)].sum())
                    A[m]["d2_argmin"] += float(d2[is_min.expand_as(d2)].sum())
                    A[m]["rows"].append((d2.sum(-1).sqrt() / nf_row.clamp_min(1e-300))[keep].float().cpu())
                    del d2
                self_t = gd.masked_fill(~valid, 0.0).double()
                M_t = res["max_only"][3][l].masked_fill(~valid, 0.0).double() - self_t
                m_t = res["min_only"][3][l].masked_fill(~valid, 0.0).double() - self_t
                A["self2"] += float(self_t.pow(2).sum())
                A["M2"] += float(M_t.pow(2).sum())
                A["m2"] += float(m_t.pow(2).sum())
                A["resid2"] += float((gfv - self_t - M_t - m_t).pow(2).sum())
                A["M_on_argmax"] += float(M_t.pow(2)[is_max.expand_as(M_t)].sum())
                A["m_on_argmin"] += float(m_t.pow(2)[is_min.expand_as(m_t)].sum())
                A["rowratio_M"].append((M_t.pow(2).sum(-1).sqrt() / nf_row.clamp_min(1e-300))[keep].float().cpu())
                A["rowratio_m"].append((m_t.pow(2).sum(-1).sqrt() / nf_row.clamp_min(1e-300))[keep].float().cpu())
                A["rowratio_self"].append((self_t.pow(2).sum(-1).sqrt() / nf_row.clamp_min(1e-300))[keep].float().cpu())
                del self_t, M_t, m_t, gfv
            del res
            torch.cuda.empty_cache()
        gfull = {n: v.double() for n, v in pacc["full"].items()}
        nfull = float(torch.sqrt(sum(v.pow(2).sum() for v in gfull.values())))
        qk = [n for n in gfull if n.endswith("attn.c_attn.weight")]
        nq = float(torch.sqrt(sum(gfull[n].pow(2).sum() for n in qk)))
        pgerr = {}
        for m in ("detach", "min_only", "max_only", "p1_hist", "p1_new"):
            g = pacc[m]
            pgerr[m] = {"all_params_rel_l2": float(torch.sqrt(sum((g[n].double() - gfull[n]).pow(2).sum() for n in gfull))) / nfull,
                        "attn_c_attn_rel_l2": float(torch.sqrt(sum((g[n].double() - gfull[n]).pow(2).sum() for n in qk))) / nq}
        p1_params_bitwise = all(torch.equal(pacc["p1_new"][n], pacc["p1_hist"][n]) for n in gfull)
        full_rep_params = all(torch.equal(pacc["full"][n], pacc["full_repeat"][n]) for n in gfull)
        rec = {"checkpoint": path, "checkpoint_sha256": sha256_file(path), "step": ck["step"], "tokens": ck["tokens_done"],
               "forward_identity_vs_full": fwd_eq, "p1_new_vs_hist_score_grad": p1_eq,
               "p1_new_vs_hist_param_grad_bitwise": p1_params_bitwise,
               "full_repeat_score_grad_bitwise": rep_eq, "full_repeat_param_grad_bitwise": full_rep_params,
               "param_grad_rel_error_vs_full": pgerr, "layers": {}}
        for l in range(12):
            A = acc[l]
            L = {"mode_error": {}, "full_decomposition": {}}
            for m in CMP:
                rows = torch.cat(A[m]["rows"]).double().numpy()
                d2 = A[m]["d2"]
                L["mode_error"][m] = {"row_median_rel": float(np.median(rows)), "aggregate_rel": float(np.sqrt(d2 / A["full2"])),
                                      "frac_err2_argmax": A[m]["d2_argmax"] / d2 if d2 > 0 else float("nan"),
                                      "frac_err2_argmin": A[m]["d2_argmin"] / d2 if d2 > 0 else float("nan"),
                                      "frac_err2_other": 1 - (A[m]["d2_argmax"] + A[m]["d2_argmin"]) / d2 if d2 > 0 else float("nan")}
            nf = np.sqrt(A["full2"])
            L["full_decomposition"] = {
                "norm_full": float(nf), "norm_self": float(np.sqrt(A["self2"])), "norm_M": float(np.sqrt(A["M2"])),
                "norm_m": float(np.sqrt(A["m2"])), "ratio_self": float(np.sqrt(A["self2"]) / nf),
                "ratio_M": float(np.sqrt(A["M2"]) / nf), "ratio_m": float(np.sqrt(A["m2"]) / nf),
                "resid_rel": float(np.sqrt(A["resid2"]) / nf),
                "M_mass_on_argmax": A["M_on_argmax"] / A["M2"] if A["M2"] > 0 else float("nan"),
                "m_mass_on_argmin": A["m_on_argmin"] / A["m2"] if A["m2"] > 0 else float("nan"),
                "row_median_ratio_self": float(np.median(torch.cat(A["rowratio_self"]).numpy())),
                "row_median_ratio_M": float(np.median(torch.cat(A["rowratio_M"]).numpy())),
                "row_median_ratio_m": float(np.median(torch.cat(A["rowratio_m"]).numpy()))}
            rec["layers"][l] = L
            row = {"tokens_m": tok, "layer": l}
            for m, v in L["mode_error"].items():
                for k2, v2 in v.items():
                    row[f"{m}__{k2}"] = v2
            for k2, v2 in L["full_decomposition"].items():
                row[f"decomp__{k2}"] = v2
            layer_rows.append(row)
        report["checkpoints"][f"{tok}M"] = rec
        print(tok, "fwd", fwd_eq, "p1", p1_eq, "p1 params", p1_params_bitwise, "repeat", rep_eq, full_rep_params, flush=True)
        del models, pacc
        torch.cuda.empty_cache()
    json.dump(report, open(f"{OUT}/grad_decomp.json", "w"), indent=1, default=str)
    with open(f"{OUT}/grad_decomp_layers.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(layer_rows[0].keys()))
        w.writeheader()
        w.writerows(layer_rows)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Assemble supplement_operator_checks.json from the individual check outputs in checks/.
Keeps compact verdicts and key numbers; every section names the full source file."""
import glob, json, os
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository


SUP = KI_REPO + "/analysis_main_v1/supplement_four_runs_20260923"
C = f"{SUP}/checks"


def J(name):
    return json.load(open(f"{C}/{name}"))


def e2_ops_summary(r):
    strict_keys = ["fwd_newprob_vs_directPN", "fwd_newprob_vs_strict_weight_ref", "fwd_repeat_newprob",
                   "vjp_newprob_vs_lerp", "vjp_oldprob_vs_lerp", "vjp_lerp_newroot_vs_oldroot"]
    out = {"env": r["env"], "precision_note": r.get("precision_note"), "cases": {}}
    for k, c in r["cases"].items():
        if "fd_lerp" in k:
            out["cases"][k] = c
            continue
        out["cases"][k] = {
            **{s: c[s]["eq_rawbits"] for s in strict_keys},
            "vjp_newprob_vs_lerp_max_abs": c["vjp_newprob_vs_lerp"].get("max_abs"),
            "hard_branch_requires_grad": c["hard_branch_requires_grad"],
            "all_finite": all(c["finite"].values()), "masked_P_zero": all(c["masked_P_zero"].values()),
            "old_weight_w_vs_direct_wN": {x: c["w_oldweight_vs_direct_wN"].get(x) for x in ("n", "mismatch_count", "max_abs")},
            "old_weight_P_vs_direct_PN": {x: c["fwd_oldweight_vs_directPN"].get(x) for x in ("mismatch_count", "max_abs")},
            "old_prob_P_vs_direct_PN": {x: c["fwd_oldprob_vs_directPN"].get(x) for x in ("mismatch_count", "max_abs")},
            "vjp_old_weight_vs_strict_weight_ref_rel_l2": c["vjp_oldweight_vs_strict_weight_ref"].get("rel_l2"),
            "ratio_PN_over_PL_nonzero_keys": c["ratio_PN_over_PL"],
            "ratio_wN_over_wL_nonzero_keys": c["ratio_wN_over_wL"],
        }
    return out


def main():
    e2v1, e2v2, e2det = J("ops_e2_v1.json"), J("ops_e2_v2.json"), J("ops_e2_v1_cuda_deterministic.json")
    e1c, e1det, e1loc = J("ops_e1_v1.json"), J("ops_e1_v1_cuda_deterministic.json"), J("ops_e1_local3090_v1.json")
    s0e1, s0e1d, s0e1l = J("step0_e1_v1.json"), J("step0_e1_v1_deterministic.json"), J("step0_e1_local3090_v1.json")
    s0e2, s0e2w = J("step0_e2_v1.json"), J("step0_e2w_v1_deterministic.json")
    cpuw = J("local_cpu_old_weight_ckpt_parity.json")

    def cpu_gate(r):
        g = r["cpu_gate"]
        return {"env": r["env"], "threads": r["env"]["num_threads"], "dtypes": ["float32 (real _quant_attention path)", "float64 (dtype-generic reconstruct + source normalization)"],
                "roots": r["roots"], "fp64_note": r["fp64_note"],
                "inputs": {k: {"att_sha256": v["att_sha256"], "upstream_sha256": v["upstream_sha256"]} for k, v in g.items()},
                "old_self_repeat_bitwise": all(v["old_self_repeat"][m][x]["eq_rawbits"] for v in g.values() for m in ("full", "detach") for x in ("fwd", "grad"))
                                           and all(v["fp64"]["old_self_repeat"][m][x]["eq_rawbits"] for v in g.values() for m in ("full", "detach") for x in ("fwd", "grad")),
                "new_vs_old_fp32_bitwise": {m: all(v["new_vs_old"][m][x]["eq_rawbits"] for v in g.values() for x in ("fwd", "score_grad")) for m in ("full", "detach")},
                "new_vs_old_fp64_bitwise": {m: all(v["fp64"]["new_vs_old"][m][x]["eq_rawbits"] for v in g.values() for x in ("fwd", "score_grad")) for m in ("full", "detach")},
                "four_mode_forward_identity_fp32": all(x["eq_rawbits"] for k, v in r["forward_identity"].items() if k.startswith("cpu/") for x in v.values()),
                "four_mode_forward_identity_fp64": all(x["eq_rawbits"] for v in g.values() for x in v["fp64"]["forward_identity"].values()),
                "criterion": "raw IEEE bit equality of every element (int view), all values finite"}

    def gpu_aux(s0, label):
        e = s0["e1"]
        return {"file": label, "env": s0["env"], "run_mode": s0["run_mode"], "diagnostic_determinism": s0.get("diagnostic_determinism"),
                "canonical_init": s0["canonical_init"], "first_step_batch": s0["first_step_batch"],
                "repeats": "3 old + 3 new per mode, all 128 microbatches of the first update, no optimizer step",
                "forward_identity_all_microbatches": e["forward_identity"]["vs_new_full_all_128_microbatches"],
                "step0_loss_acc": e["forward_identity"]["loss_acc"],
                "backward": {m: {k: v for k, v in e["backward_regression"][m].items() if k in ("status", "old_self_bitwise", "old_new_bitwise", "finite", "n_exceeding_old_old_scale")}
                             | {"param_grad_max_rel_l2": {p: v["max_rel_l2"] for p, v in e["backward_regression"][m]["param_grad_pairs"].items()},
                                "score_grad_layers": e["backward_regression"][m]["score_grad_layers"]}
                             for m in ("full", "detach")},
                "single_channel_grads_finite_not_none": e["param_grads_single_pass"]["finite_and_not_none"],
                "grad_checkpointing_old_detach_on_vs_off_bitwise": e["grad_checkpointing_old_detach"]["on_vs_off"]["all_eq_rawbits"]}

    out = {
        "generated_from": sorted(os.path.basename(p) for p in glob.glob(f"{C}/*.json")),
        "definitions": {"eq_rawbits": "identical IEEE bit patterns element-wise (torch.equal on int views); NaN/Inf never pass",
                        "exact_feed": "k = I, D = T with sqrt(D) a power of two, V = I, TF32 off: att = q/sqrt(D) and y = P exactly, dL/ds = sqrt(D) q.grad exactly",
                        "step0": "the initial forward/backward before any optimizer update, all 128 microbatches of the first update"},
        "e2_operator": {"v1_nondeterministic": e2_ops_summary(e2v1), "v2_exact_fwm_boundaries": e2_ops_summary(e2v2),
                        "v1_cuda_deterministic_diagnostic": e2_ops_summary(e2det),
                        "fwm_cpu_vs_gpu_note": ("GPU exp(-4.5) = 0x3c360283 (odd mantissa) vs CPU 0x3c360282 (even): with an even-mantissa e_lo "
                                                "the old Weight-STE round trip w_L + (w_N - w_L) is exact for FWM K4 (w_L < 2.74 e_lo); with the "
                                                "odd one it can land +-1 ulp away, which is why FWM old Weight mismatches appear on GPU only "
                                                "(code/checks/diag_fwm_cpu_gpu.py)")},
        "old_weight_reuse_verification": {
            "verdict": "FAILED - old Weight-STE forward is not the strict hard forward on the GPU training path at trained score scales; hardware/software also differ (RTX 3090 torch 2.5.1 vs RTX 5090 torch 2.13)",
            "consequence": "user authorized (2026-09-23) new strict Weight-STE runs for both calibrations; the old Weight controls are reported only as historical references, never as backward-only contrasts",
            "model_level_cpu_diagnostic_old_weight_checkpoints": {k: {"old_weight_vs_strict_P_layers_equal": sum(v["old_weight__vs__strict_weight_ref"]["P_layers_eq"]),
                                                                     "logits_max_abs": v["old_weight__vs__strict_weight_ref"]["logits"]["max_abs"],
                                                                     "block0_loss_old_strict": v["old_weight__vs__strict_weight_ref"]["loss_values"],
                                                                     "new_prob_equals_strict_all_layers": all(v["new_prob__vs__strict_weight_ref"]["P_layers_eq"])}
                                                                 for k, v in cpuw["cases"].items()},
        },
        "e1_operator": {"analytic_single_channel_fp64": e1c["analytic"], "ties_and_edges": e1c["edge"],
                        "zero_sum_rho_random_upstream_structure_check": e1c["zero_sum"],
                        "note_zero_sum": "rho on fixed diagnostic upstreams is a structural check, not a loss-gradient residual"},
        "step0": {
            "initial_state_hash": s0e1["canonical_init"]["hash_by_root"]["old_phaseB_base"],
            "initial_state_identical_across_roots": s0e1["canonical_init"]["all_equal"],
            "batch_hash": s0e1["first_step_batch"],
            "rng": s0e1["run_mode"]["rng"],
            "historical_reconstruction_limit": s0e1["canonical_init"]["note"],
            "e2_forward_identity_5090": {c: {"vs_new_prob_all_microbatches": v["comparisons_vs_new_prob_all_microbatches"],
                                            "loss_acc": v["loss_acc_train_py_style"]} for c, v in s0e2["e2_forward_identity"].items()},
            "e2_step0_backward_5090_deterministic_diagnostic": {c: {"param_grads": {p: {"bitwise": x["all_eq_rawbits"], "max_rel_l2": x["max_rel_l2"]} for p, x in v["param_grads"].items()},
                                                                   "score_grad": v["score_grad"], "loss_acc": v["loss_acc"]}
                                                               for c, v in s0e2w["e2_step0_backward"].items()},
            "e1_backward_regression": {
                "cpu_operator_gate": {"gpu-host-A_torch2.13": cpu_gate(e1c), "local_torch2.5.1": cpu_gate(e1loc)},
                "gpu_operator_auxiliary": {"gpu-host-A": {k: {"old_old_full_grad_bitwise": v["old_self_repeat"]["full"]["grad"]["eq_rawbits"],
                                                         "new_old_full_grad_bitwise": v["new_vs_old"]["full"]["score_grad"]["eq_rawbits"],
                                                         "new_old_detach_grad_bitwise": v["new_vs_old"]["detach"]["score_grad"]["eq_rawbits"]}
                                                     for k, v in e1c["gpu_aux"].items()}},
                "gpu_model_auxiliary": {"gpu-host-A_rtx5090": gpu_aux(s0e1, "step0_e1_v1.json"),
                                        "local_rtx3090_training_gpu": gpu_aux(s0e1l, "step0_e1_local3090_v1.json")},
                "diagnostic_determinism": {"operator_gpu_host_A": {"all_new_vs_old_bitwise": all(v["new_vs_old"][m]["score_grad"]["eq_rawbits"] for v in e1det["gpu_aux"].values() for m in ("full", "detach"))},
                                           "model_gpu_host_A": gpu_aux(s0e1d, "step0_e1_v1_deterministic.json"),
                                           "used_in_training": False},
                "launch_eligible": True,
                "reason": ("CPU operator gate bit-identical (fp32 and fp64, both environments); analytic single-channel formulas pass "
                           "(max abs err 3.3e-16); four-mode forward bit-identical on CPU and GPU incl. all 128 step-0 microbatches; "
                           "GPU full-mode differences are at the old-vs-old scale and vanish under deterministic algorithms "
                           "(diagnostic process only); detach bit-identical without it; no None/non-finite gradients; step-0 loss on the "
                           "local 3090 (10.934344) equals the historical 3090 log of the controls (10.93434)"),
            },
        },
    }
    json.dump(out, open(f"{SUP}/reports/supplement_operator_checks.json", "w"), indent=1, default=str)
    print("wrote", f"{SUP}/reports/supplement_operator_checks.json")


if __name__ == "__main__":
    main()

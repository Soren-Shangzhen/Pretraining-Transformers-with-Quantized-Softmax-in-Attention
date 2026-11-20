#!/usr/bin/env python3
"""Build the rb- run manifest from runs_v5.csv (the per-run configuration authority).

Every row of runs_v5.csv becomes one manifest row with: run_id, experiment, priority, machine, slot,
protocol, seed, code root + its SHA-256, the exact train command (with ${RB_DATA}/${RB_RUNS}/${RB_CODE}
placeholders resolved on the machine), environment variables, pairing, checkpoint schedule, parent
checkpoint for branches, prediction and reading rule (plan v5, Section 7), and the decisions that are
still pending (plan 13.5).  Nothing here starts a run.

usage: python build_run_manifest.py  (writes manifests/rb-runs_v5_manifest.{tsv,json})
"""
import csv, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RB = os.path.abspath(os.path.join(HERE, "..", ".."))
CSV = os.path.join(RB, "manifests", "runs_v5.csv")
ADDENDUM = os.path.join(RB, "manifests", "runs_v5_addendum.csv")   # user-approved additions (2026-11-10)
ROOT_LOCAL = os.path.join(RB, "code", "rb-root_v1")
SUP = os.path.abspath(os.path.join(RB, "..", "supplement_four_runs_20260923", "code"))
C2_MINMAX_LOCAL = os.path.join(SUP, "e2_minmax_probste_strict")

M = 1_000_000
STEP = 131072


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


ROOT_SHA = {f: sha(os.path.join(ROOT_LOCAL, f)) for f in ("model.py", "quant_attention.py", "train_formal.py")}
C2_SHA = {f: sha(os.path.join(C2_MINMAX_LOCAL, f)) for f in ("model.py", "quant_attention.py", "train_formal.py")}

COMMON = ("--n_layer 12 --n_head 12 --n_embd 768 --block_size 1024 --batch_tokens 131072 --micro_batch 1 "
          "--lr 6e-4 --min_lr_ratio 0.1 --weight_decay 0.1 --beta1 0.9 --beta2 0.95 --grad_clip 1.0 --device cuda")
# protocol per suite: from the frozen checkpoint configs (suite D) and training logs (suites A, B)
PROTO = {
    "suiteD 124M@100M WikiText-103": dict(data="wikitext103", warmup=10 * M, max=100 * M, eval_every=5 * M,
                                          ckpt_every=20 * M, eval_batches=100, weights_only=True),
    "suiteB 124M@250M FineWebEdu": dict(data="finewebedu_3b", warmup=25 * M, max=250 * M, eval_every=25 * M,
                                        ckpt_every=25 * M, eval_batches=100, weights_only=True),
    "suiteA 124M@2.5B FineWebEdu": dict(data="finewebedu_3b", warmup=25 * M, max=2500 * M, eval_every=50 * M,
                                        ckpt_every=125 * M, eval_batches=100, weights_only=False),
}

# parents of the branches (cloud paths from manifests/primary_37_checkpoints.tsv and the suite-D manifest)
CLOUD = "<CLOUD_DISK>"
PARENT = {
    ("suite B MinMax-Weight K4", 50): (f"{CLOUD}/vast_formal_250m_v1/formal_runs_fbmatch_v1/formal_124m_s1337_nearest_k4_weightste_250m/ckpt_step000382_50.1m.pt", "weight_hist", "formal_code_factorial250_v1/formal_base_v2"),
    ("suite B MinMax-Weight K4", 125): (f"{CLOUD}/vast_formal_250m_v1/formal_runs_fbmatch_v1/formal_124m_s1337_nearest_k4_weightste_250m/ckpt_step000954_125.0m.pt", "weight_hist", "formal_code_factorial250_v1/formal_base_v2"),
    ("suite B MinMax-Prob K4", 50): (f"{CLOUD}/vast_formal_250m_v1/formal_runs_fbmatch_v1/formal_124m_s1337_nearest_k4_probste_250m/ckpt_step000382_50.1m.pt", "prob_hist", "formal_code_factorial250_v1/formal_nearest_probste_v1"),
    ("suite B MinMax-Prob K4", 125): (f"{CLOUD}/vast_formal_250m_v1/formal_runs_fbmatch_v1/formal_124m_s1337_nearest_k4_probste_250m/ckpt_step000954_125.0m.pt", "prob_hist", "formal_code_factorial250_v1/formal_nearest_probste_v1"),
    ("suite A MinMax-Weight K4", 250): (f"{CLOUD}/formal_runs_scale_v1/formal_124m_s1337_nearest_k4_fullgrad_2p5b/ckpt_step001908_250.1m.pt", "weight_hist", "formal_code_factorial250_v1/formal_base_v2"),
}
for s in (7, 42, 253):
    PARENT[(f"suite D MinMax-Weight K4 seed {s}", 40)] = (
        f"${{RB_PARENTS}}/p3_s{s}_nearest_k4_fullgrad_100m/ckpt_step000306_40.1m.pt", "weight_hist",
        "D_robustness_124m_100m_5seed/code_snapshots/local_phaseB_base")
    PARENT[(f"suite D MinMax-Prob K4 seed {s}", 40)] = (
        f"${{RB_PARENTS}}/formal_124m_s{s}_minmax_nearest_k4_probste_100m/ckpt_step000306_40.1m.pt", "prob_hist",
        "D_robustness_124m_100m_5seed/code_snapshots/formal_code_probste_5seeds_100m_v1/formal_nearest_probste_v1")

# per-layer h0 (nats) for freeze-h: native width h of the paper's geometry probe at the branch checkpoint
# (first 8 validation blocks, rows i >= 1, layer mean), eval_native_v1/<suite>/<run>/<ckpt>/geometry_layers.csv
H0 = {
    ("suite B MinMax-Weight K4", 50): "1.398775,4.924105,2.506495,4.201291,8.273577,8.624490,7.414962,6.529430,4.627206,3.793493,2.602424,2.196885",
    ("suite B MinMax-Weight K4", 125): "6.038305,16.599321,3.600654,17.188646,22.334167,18.092990,16.626243,13.298043,9.027258,6.422319,4.293497,3.708741",
    ("suite A MinMax-Weight K4", 250): "25.021593,60.573802,20.024882,59.742705,62.155911,54.775483,39.413426,23.187504,14.432136,12.114406,9.700631,10.564618",
}

PREDICTION = {
    "C0": ("same-stack softmax control for every new 100M condition (no prediction)",
           "13.2.3: step-1 train loss of seed 1337 against the suite-D softmax log (11.01438)"),
    "C0b": ("same-stack 250M softmax control (plan 4.3; no prediction)",
            "every 250M rb run is paired with it (same machine, same stack); the historical suite-B softmax is kept as "
            "background only"),
    "E1": ("no direction pre-set",
           "FWM-LERP vs LERP at matched K: if not worse than LERP at 2.5B, a fixed window with the exact "
           "Jacobian is the best coarse-K configuration and needs no STE; if it stays near the Nearest plateau, "
           "zero-tail truncation has a price for small weights. 100M: 3-seed paired t interval vs C0; "
           "single-seed differences < 0.01 nats are not interpreted"),
    "E1b": ("no direction pre-set",
            "125M checkpoint evaluated first (suite A LERP K4 at 125M: +0.086). Gap > +0.3 or a non-finite value: "
            "pause and inspect the implementation (decision: user); otherwise report as measured"),
    "E11": ("two outcomes pre-declared",
            "both seeds separated at 100M (CI lower bound > 0.01): the origin term must be kept on a static grid; "
            "neither 100M nor 250M separated: the MinMax failure depends on the data-dependent width, scope "
            "narrowed. Readout detach - full at 20/40/60/80/100M: NLL, span, gradient norm, row-gradient sum"),
    "E2a/b": ("chain 1 (surrogate is the cause): after W->P span stops growing and the gap converges to P's "
              "trajectory; after P->W span starts growing and the gap opens. chain 2 (width is the mediator): "
              "freeze-h weakens span growth and gap widening under Weight-STE",
              "W->P gap does not converge: irreversible damage in the weights; freeze-h no effect: width is not "
              "the mediator; either narrows Section 5.4. Valid only if W->W and P->P stay within the original "
              "trajectory's noise. Readout every 25M (every 5M for the first 25M): dNLL, span, h, TV, clip fraction"),
    "E2c": ("W->W should reproduce the collapse near 375M",
            "if W->W does not reproduce the collapse after restoring, the group is descriptive only"),
    "E2d": ("switch effect has the same sign on seeds 7/42/253",
            "loss only (span growth is small at 100M); paired across seeds"),
    "E5": ("none pre-set (formats are fp32 simulations of the format rule on e^z; not the recipes' real use)",
           "3-seed paired t interval vs C0 at 100M; 250M single seed vs suite-B softmax (same-machine re-eval)"),
    "E3a": ("E3a-ii may train poorly (UINT8 drops keys with p < 1/510; uniform rows over 510+ keys become zero); "
            "no direction pre-set for E3a-i",
            "E3a-2.5B rule: E3a-ii if its 250M dNLL <= +0.05, else (or if non-finite) E3a-i; decided when E8a ends; "
            "fallback FWM-Weight K16 seed 7 if GPU alignment or the 250M result is missing then"),
    "E3a-2.5B": ("as E3a", "variant chosen by the pre-declared rule (E3a-ii if its 250M dNLL <= +0.05, otherwise or if "
                 "non-finite E3a-i), decided when E8a ends on A card 2 (changed 2026-11-10 from 'Nov 12 noon', before "
                 "any E3a result); fallback FWM-Weight K16 seed 7 (strict) if the GPU alignment or the 250M result is "
                 "missing at that time"),
    "E3b": ("E3b-d worse than E3b-g, gap widening with training length",
            "both not separated at 250M (< 0.01): 'no calibration-gradient effect observed at EXAQ's tensor-level "
            "calibration'. Also record sigma, lo and the mass on clamped tail keys"),
    "E4": ("none pre-set",
           "(1) does the MinMax-Weight gap persist when calibration becomes semi-static; (2) does the EMA follow "
           "span growth (250M runs)"),
    "E5n": ("none pre-set", "BF16 full chain vs BF16-exp: isolates the precision of row sum and division"),
    "E7": ("none pre-set", "TF32-off training vs the same-root TF32-on control"),
    "E8a": ("none pre-set", "second softmax seed at 2.5B: noise reference per benchmark"),
    "E8c": ("BLiMP pairwise difference of LERP K16 vs softmax stays negative on seed 7",
            "seed-7 BLiMP paired difference negative and bootstrap CI excludes 0: 'reproduced on a second seed'; "
            "otherwise 'not reproduced, single-seed difference treated as noise'. Both reported"),
}


def ckpt_schedule(parent_tokens, end_tokens, grid, relative):
    """dense every 5M for the first 25M after the branch, then the original checkpoint grid:
    absolute multiples of `grid` (suites A/B) or parent + k*grid (suite D, accumulating schedule).
    Targets that the same optimizer step would hit are merged."""
    dense = [parent_tokens + j * 5 * M for j in range(1, 6)]
    if relative:
        on_grid = [parent_tokens + k * grid for k in range(1, 100) if parent_tokens + k * grid <= end_tokens]
    else:
        on_grid = [k * grid for k in range(1, end_tokens // grid + 1)]
    on_grid = [x for x in on_grid if x > parent_tokens + 25 * M - 2 * STEP]
    steps = {}
    for x in sorted(dense + on_grid + [end_tokens]):
        if parent_tokens < x <= end_tokens:
            steps.setdefault(-(-x // STEP), x)       # first step whose token count reaches x
    return sorted(steps.values())


# scheduling decisions after runs_v5.csv (user, 2026-11-10); configuration is unchanged
SLOT_OVERRIDE = {"rb-E8c-01": "card1-slot2", "rb-E3a-2.5B-01": "card2-slot2"}
for _i in range(1, 9):
    SLOT_OVERRIDE[f"rb-E3a-0{_i}"] = "1"      # E3a moved to the front of B wave 1
SCHEDULE_NOTE = {
    "rb-E8c-01": "A card 1 after E1b (swap approved 2026-11-10)",
    "rb-E3a-2.5B-01": "A card 2 after E8a (swap approved 2026-11-10); card 2 after E3a-2.5B left open (user decides Nov 13)",
}
CHECK_RULES = {
    "gpu_gradient_rule_v2_2026-11-10": "13.3 root equivalence: CPU bitwise equality is the gate; GPU gradients pass if for every "
        "parameter tensor max|g_rb - g_base| <= max(1e-4 * max|g_base|, 3 x the largest difference among 6 repeats of the "
        "same root). Reason: atomic-add backward makes identical GPU code differ in low-order bits; the relative-only rule "
        "(1e-4) failed on torch 2.5.1 for a tensor with max|g| ~ 0.02 whose same-root repeats differ by the same 3.78e-6.",
    "cpu_bitwise_single_thread_2026-11-10": "13.3 CPU bitwise gates run single-threaded (OMP_NUM_THREADS=1, "
        "MKL_NUM_THREADS=1, torch.set_num_threads(1)), locally and on the cloud machines. Evidence "
        "(checks/13_3_cpu_thread_determinism_*.json): with default threads the r3 root does not reproduce itself (one of "
        "three identical 3-step CPU runs differs in low-order bits of the final weights); single-threaded, all three r3 "
        "runs and all three rb-root_v1 runs are bitwise identical.",
    "e11_clamp_gate_2026-11-10": "E11 is queued only on machines where torch.clamp passes gradient 1 at its bounds (CPU and every GPU).",
    "e3a_2p5b_decision_time_2026-11-10": "E3a-2.5B variant decided when E8a ends (was: Nov 12 noon); criterion unchanged; "
        "changed before any E3a result existed.",
    "step1_gate_per_stack_2026-11-10": "Step-1 train-loss gate on a rented machine: a reference recorded on S1 (RTX 5090, "
        "torch 2.13.0+cu132) must be matched exactly; a reference recorded on the RTX 3090 stack (torch 2.5.1) passes if "
        "|diff| <= 1e-3. Reason: on machine B the WikiText seed-42 step-1 loss was 10.97603 against the 3090 record 10.97600 "
        "(the old rule, exact equality for every reference, stopped B_checks there). Cross-stack evidence required in "
        "addition (stop if not identical): SHA-256 of the 128 step-1 token sequences (x and y) and of every initial weight "
        "tensor, machine vs local 3090, seeds 7/42/1337 (tools/step1_fingerprint.py); weights may differ only by "
        "max|diff| < 1e-6 (recorded). Result on B: tokens identical, initial weights identical (76/76 tensors) for all "
        "three seeds (checks/step1_fingerprint_B.json, checks/step1_fingerprint_local3090.json).",
    "step1_gate_comparison_2026-11-10": [
        dict(data="finewebedu_3b", seed=1337, reference_stack="S1", B_train=11.00188, B_val=11.00167, B_gnorm=15.216,
             ref_train=11.00188, ref_val=11.00167, ref_gnorm=15.216, rule="exact", passed=True),
        dict(data="wikitext103", seed=7, reference_stack="RTX 3090", B_train=10.93436, B_val=10.93425, B_gnorm=15.637,
             ref_train=10.93436, ref_val=10.93430, ref_gnorm=15.638, rule="|diff| <= 1e-3", passed=True),
        dict(data="wikitext103", seed=42, reference_stack="RTX 3090", B_train=10.97603, B_val=10.97868, B_gnorm=15.832,
             ref_train=10.97600, ref_val=10.97869, ref_gnorm=15.829, rule="|diff| <= 1e-3", passed=True,
             note="failed the former exact rule; the local 3090 reproduces all three 3090 rows exactly"),
    ],
    "B_wave1_hold_2026-11-10": "Machine B: the 8 E3a runs start once their gates pass (torch version, data SHA-256, E3 "
        "reference comparison of E3a, surrogate gradients, step-1 token hashes, initial weights); every later run waits at "
        "a queue hold point ('@wait' in queues/B_wave1.txt, rb_queue.py) until the remaining B checks have passed. "
        "rb-C0b-01 is the first run after the hold (250M pairing reference of the E3a-2.5B dNLL rule).",
    "B_queue_continuous_2026-11-10": "Machine B keeps one run per GPU (packing ratio 0.83 on B, 1.15 on A, threshold "
        "1.6). Waves 2-4 run in one follow-on runner (rb_queue.py --start-after B_wave1.txt --gpu-free-check) that "
        "starts once every wave-1 run has been dispatched and fills each GPU as soon as it is free, so no GPU waits "
        "at a wave boundary. User decision: inside wave 4 the longest runs go first (projected durations on B), so "
        "the 8 GPUs finish together; run set and configurations unchanged. Projection at the change: all B runs "
        "done 26.8 h -> 25.1 h after 2026-11-10 23:34Z, idle GPU-h in the tail 16.3 -> 2.7.",
    "e3a_2p5b_variant_2026-11-10": "E3a-2.5B variant = E3a-ii (u8), the outcome of the pre-declared rule, confirmed by the "
        "user. Input (full FineWebEdu validation, 976 blocks, batch 1, TF32 off, rb-root_v1, machine B): rb-C0b-01 "
        "3.888275, rb-E3a-04 (E3a-i) 3.889673 (dNLL +0.00140), rb-E3a-08 (E3a-ii) 3.888553 (dNLL +0.00028 <= +0.05), "
        "no non-finite values (checks/e3a_250m_dnll.json). The command keeps the {E3A_2P5B_VARIANT} placeholder; it is "
        "filled at launch (rb_queue.py --resolve E3A_2P5B_VARIANT=u8) and recorded in the run's start record. Launch: "
        "A card 2 (GPU 1) as soon as rb-E8a-01 ends (queues/A_card2_next.txt, follow-on runner).",
    "E8d_cancelled_2026-11-12": "No third 2.5B seed (user decision): rb-E8d-01 (softmax seed 42 @ 2.5B, suite A protocol) and "
        "rb-E8d-02 (LERP K=16 seed 42 @ 2.5B) are cancelled and never started. Reason: a LERP K=16 2.5B run takes about 47 h on "
        "machine A (measured: rb-E8c-01 at ~15k tok/s), too late for the revised manuscript. Consequence: E8a/E8c give two seeds "
        "for the 2.5B LERP K=16 vs softmax comparison (no t interval). Machine A's remaining work is evaluation only: E3a-2.5B "
        "(card 2, when it ends) and E8c plus a re-evaluation of E8a's validation NLL on A (card 1, when E8c ends).",
    "e3b_alignment_accepted_2026-11-10": "E3b (EXAQ) alignment evidence accepted by the user: the training forward is "
        "bitwise equal to the IntAttention reproduction on the same device (local CPU and RTX 3090). On the rented 5090 "
        "machines (A and B, CPU AMD EPYC 7K62 = AVX2) the comparison with the locally made reference differs by at most "
        "2.0e-6 with no grid-index change: on CUDA only sigma (a whole-tensor reduction) differs 5090 vs 3090, and with "
        "the 3090's lo/delta injected all 14 cases are bitwise; on CPU window and softmax input are bitwise and "
        "torch.softmax differs by kernel width (AVX2 vs local AVX512), reproduced 14/14 locally with "
        "ATEN_CPU_CAPABILITY=avx2 under torch 2.5.1 and 2.14.1. Evidence: checks/13_3_e3_reference_<host>.json, "
        "checks/13_3_exaq_inject_B_*.json, checks/13_3_exaq_hashes_*.json, code/e3_ref/exaq_stats_diag.py.",
}


def build_row(r):
    p = PROTO[r["protocol"]]
    run, exp, seed = r["run_id"], r["experiment"], int(r["seed"])
    op = r["operator"]; K = r["K"]; sur = r["surrogate"]; rg = r["range_grad"]
    env = {}
    root = "${RB_CODE}/rb-root_v1"; root_sha = ROOT_SHA; script = "train_formal.py"
    a = []
    if op == "softmax":
        a += ["--attn exact_fp32_softmax"]
    elif op.startswith("FWM-LERP"):
        a += [f"--attn quant --approx lerp --K {K} --range_mode relevant_zero --range_tau 6",
              "--range_grad " + ("detach" if rg.startswith("detach") else "full")]
    elif op.startswith("FWM-Nearest") or op.startswith("FWM-Weight"):
        a += [f"--attn quant --approx nearest --K {K} --range_mode relevant_zero --range_tau 6 --nearest_surrogate weight_strict",
              "--range_grad " + ("detach" if rg.startswith("detach") else "full")]
    elif op == "MinMax-LERP":
        a += [f"--attn quant --approx lerp --K {K} --range_mode minmax --range_grad full"]
    elif op == "MinMax-Nearest":
        a += [f"--attn quant --approx nearest --K {K} --range_mode minmax --range_grad full",
              "--nearest_surrogate " + ("prob" if sur.startswith("Prob") else "weight_strict")]
    elif op == "resolved-Nearest":
        a += [f"--attn quant --approx nearest --K {K} --range_mode relevant_zero --tail_policy resolved "
              f"--range_grad full --nearest_surrogate weight_strict --resolved_h0 {{H0}}"]
    elif op in ("BF16-exp", "FP8-E4M3 rounded exp", "MXFP8-style", "NVFP4-style", "UINT8 uniform", "BF16 full-chain softmax"):
        mode = {"BF16-exp": "softmax_bf16exp", "FP8-E4M3 rounded exp": "softmax_fp8e4m3exp",
                "MXFP8-style": "softmax_mxfp8", "NVFP4-style": "softmax_nvfp4", "UINT8 uniform": "softmax_uint8",
                "BF16 full-chain softmax": "softmax_bf16chain"}[op]
        a += [f"--attn {mode} --range_grad full"]
    elif op.startswith("IndexSoftmax E3a-i "):      # trailing space: "E3a-ii" also starts with "E3a-i"
        a += ["--attn indexsoftmax --attn_variant fp --range_grad full"]
    elif op.startswith("IndexSoftmax E3a-ii "):
        a += ["--attn indexsoftmax --attn_variant u8 --range_grad full"]
    elif op.startswith("IndexSoftmax (E3a-ii if"):
        a += ["--attn indexsoftmax --attn_variant {E3A_2P5B_VARIANT} --range_grad full"]
    elif op.startswith("EXAQ"):
        a += ["--attn exaq --exaq_bits 3 --range_grad full --attn_variant " + ("detach" if "E3b-d" in op else "graph")]
    elif op.startswith("EMA-window"):
        a += [f"--attn quant --approx nearest --K {K} --range_mode ema --ema_decay 0.999 --range_grad full",
              "--nearest_surrogate " + ("prob" if sur.startswith("Prob") else "weight_strict")]
    else:
        raise ValueError(f"{run}: unknown operator {op!r}")

    if exp == "E7":
        env["NVIDIA_TF32_OVERRIDE"] = "0"
        if op == "MinMax-Nearest":
            # control = the Appendix C.2 strict MinMax-Weight seed-7 run; same frozen root, TF32 off only
            root = "${RB_CODE}/c2_roots/e2_minmax_probste_strict"; root_sha = C2_SHA
            a = [f"--attn quant --approx nearest --K {K} --range_grad full --nearest_surrogate weight_strict"]

    proto = [COMMON, f"--data_dir ${{RB_DATA}}/{p['data']}", "--out_dir ${RB_RUNS}", f"--run_name {run}",
             f"--seed {seed}", f"--warmup_tokens {p['warmup']}", f"--max_tokens {p['max']}",
             f"--eval_every_tokens {p['eval_every']}", f"--eval_batches {p['eval_batches']}", "--stats_batches 1"]
    rb_only = root.endswith("rb-root_v1")
    if rb_only:
        proto += ["--health_every_steps 1", "--halt_on_nonfinite"]
    parent = h0 = parent_sur = parent_root = ""
    if r["start_from"] == "scratch":
        if p["data"] == "finewebedu_3b":
            # suites A/B saved at absolute multiples (steps 191, 382, ... / 954, 1908, ...)
            proto += ["--ckpt_tokens " + ",".join(str(k * p["ckpt_every"]) for k in range(1, p["max"] // p["ckpt_every"] + 1))]
        else:
            proto += [f"--ckpt_every_tokens {p['ckpt_every']}"]   # suite D: accumulating (steps 153, 306, 459, ...)
        if p["weights_only"] and rb_only:
            proto += ["--ckpt_weights_only"]
    else:
        sf = r["start_from"]
        key = None
        for (name, tok), val in PARENT.items():
            if sf.startswith(name + " ") and f" at {tok}M" in sf:
                assert key is None, (run, sf)
                key = (name, tok)
        if key is None:
            raise ValueError(f"{run}: no parent for {sf!r}")
        parent, parent_sur, parent_root = PARENT[key]
        if key[0].startswith("suite D"):
            parent_root = "${RB_CODE}/../../" + parent_root      # RB_CODE = analysis_main_v1/rb-rebuttal_v5/code
        else:
            parent_root = f"{CLOUD}/{parent_root}"
        ptok = {50: 382, 125: 954, 250: 1908, 40: 306}[key[1]] * STEP
        stop = {"E2a/b": 250 * M, "E2c": 625 * M, "E2d": 100 * M}[exp]
        grid = {"E2a/b": 25 * M, "E2c": 25 * M, "E2d": 20 * M}[exp]
        sched = ckpt_schedule(ptok, stop, grid, relative=exp == "E2d")
        proto[proto.index(f"--eval_every_tokens {p['eval_every']}")] = f"--eval_every_tokens {5 * M}"
        proto += [f"--init_from {parent}", "--branch_override", f"--parent_surrogate {parent_sur}",
                  f"--parent_code_root {parent_root}", f"--ckpt_tokens {','.join(map(str, sched))}",
                  "--ckpt_weights_only"]
        if stop < p["max"]:
            proto += [f"--stop_at_tokens {stop}"]
        if op == "resolved-Nearest":
            h0 = H0[(key[0], key[1])]
    cmd = " ".join([f"cd {root} &&", "${RB_PY}", script] + proto + a).replace("{H0}", h0)
    pred, rule = PREDICTION[exp]
    pair = r["pair_with"]
    if "250M" in r["protocol"] and exp != "C0b":
        pair = "rb-C0b-01 (same-stack 250M softmax, plan 4.3); " + pair
    pending = []
    if exp == "E4":
        pending.append("13.5: E4 defaults (span = mean M-m over rows i>=1 per head; decay 0.999; update once per "
                       "training forward, before quantising; zero tail; buffer frozen at evaluation)")
    if exp == "E3a-2.5B":
        pending.append("variant by the pre-declared rule when E8a ends (threshold +0.05 kept, 13.5)")
    if exp == "E1b":
        pending.append("13.5: E1b +0.3 inspection line at 125M")
    return dict(run_id=run, experiment=exp, priority=r["priority"], machine=r["machine"],
                slot=SLOT_OVERRIDE.get(run, r["wave_or_slot"]), slot_in_csv=r["wave_or_slot"],
                schedule_note=SCHEDULE_NOTE.get(run, ""),
                protocol=r["protocol"], seed=seed, operator=op, K=K, surrogate=sur, range_grad=rg,
                start_from=r["start_from"], parent_checkpoint=parent, parent_surrogate=parent_sur,
                parent_code_root=parent_root, resolved_h0=h0, gpu_h_plan=r["gpu_h"], pair_with=pair,
                new_code=r["new_code"], note=r["note"], code_root=root,
                code_sha256_quant_attention=root_sha["quant_attention.py"], code_sha256_model=root_sha["model.py"],
                code_sha256_train=root_sha["train_formal.py"], env=json.dumps(env), command=cmd,
                prediction=pred, reading_rule=rule, pending_decisions=" | ".join(pending), status="registered")


def main():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
    add = list(csv.DictReader(open(ADDENDUM, encoding="utf-8-sig")))
    out = [build_row(r) for r in rows + add]
    assert len(out) == 97 + len(add) and len({o["run_id"] for o in out}) == len(out)
    assert all(o["run_id"].startswith("rb-") for o in out)
    mdir = os.path.join(RB, "manifests")
    with open(os.path.join(mdir, "rb-runs_v5_manifest.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(out)
    json.dump(dict(source=CSV, source_sha256=sha(CSV), addendum=ADDENDUM, addendum_sha256=sha(ADDENDUM),
                   rb_root_sha256=ROOT_SHA, c2_minmax_root_sha256=C2_SHA, check_rules=CHECK_RULES,
                   runs=out), open(os.path.join(mdir, "rb-runs_v5_manifest.json"), "w"), indent=1)
    print(f"{len(out)} runs; rb-root_v1 quant_attention {ROOT_SHA['quant_attention.py'][:12]}")


if __name__ == "__main__":
    main()

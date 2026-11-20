# REPRODUCTION_INDEX

Index of this repository, the released code of the paper *Pretraining Transformers with Quantized Softmax in
Attention* (arXiv:2609.33591). Appendix letters and table and figure numbers are those of version 3 of the arXiv paper
(arXiv:2609.33591v3; Appendices A–J, Tables S1–S57, Figures S1–S15). It lists the manifests, code and result files of every run the paper
reports (section 2), maps each table and figure to its generator and source files (section 4; Appendix I tables and
figures to their runs in section 5), and gives the commands for evaluation, analysis, tables, figures and checks
(section 6). Every path is relative to the root of this repository. Model checkpoints and tokenized data are not included
(see `README.md`).

Placeholders in manifests, records and scripts: `<REPO>` is the root of this repository; `<CLOUD_DISK>` is the shared disk that
held the training runs, checkpoints and tokenized data of the cloud runs (not included); `<POD_WORKSPACE>` is the
working directory on the two RTX 5090 machines of Appendix I; `<LOCAL_HOME>`, `<CLOUD_HOME>`, `<LOCAL_TMP>` and
`<CLOUD_ENV>` are home directories, a temporary directory and Python environments; `<THIRD_PARTY>` is a local directory with the IntAttention release (not
included); `<user>` replaces a user name. Machine labels occur as values in records (for example the field `host`) and
as the suffix of the file names of records that are kept per machine: `local3090` is the local workstation with an RTX
3090 (stack S3 of Table S4 for the Appendix I checks); `machine-A` and `machine-B` are the 2-GPU and the 8-GPU RTX 5090
machines of Appendix I; `gpu-host-A` and `pod8` are neutral labels of rented machines used for the earlier suites.
Directories that hold evaluations on different GPUs are named after the GPU (`rtx5090`, `rtx3090`).

## 1. Names in the released code

Result files, run names, run directories and code fields keep their original identifiers. They map to the names used
in the paper as follows.

| Identifier in files | Name in the paper |
|---|---|
| `QRM`, `qrm`, `FW`, `fw`, `range_mode=relevant_zero` | FWM (fixed window at the row maximum, τ = 6) |
| `nearest` | Nearest |
| `lerp` | LERP |
| `minmax`, `MinMax`, `MM`, `range_mode=minmax` (default) | MinMax |
| `exact`, `Exact`, `exact_fp32_softmax`, `softmax` | softmax (the exact fp32 softmax control) |
| `_fullgrad`, `range_grad=full` | the full calibration gradient; it does not identify the surrogate |
| `detach`, `range_grad=detach` | the detached calibration gradient (Appendix F) |
| `max_only`, `min_only` | single-channel calibration gradient (Appendix F) |
| `probste`, `Prob` | Prob-STE |
| `weightste`, `Weight`, and runs of the roots listed as Weight-STE in section 2 | Weight-STE |
| `strict` | strict hard forward of Appendix C.2 |
| `tau6`, `range_tau=6` | τ = 6 |
| `k4`, `K4`, `_K4_` | K = 4 (likewise for other K) |
| `s7`, `s42`, `s253`, `s1337`, `s2026` | seeds |
| `2p5b`, `250m`, `100m` | 2.5B, 250M and 100M training tokens |
| `formal_124m_…`, `formal_1b_…`, `formal_qrm_1b_…` | runs of suites A–C (124M and 1B models) |
| `p3_…`, `qrm_…`, `formal_124m_s*_…_100m` in `analysis_main_v1/D_robustness_124m_100m_5seed/` | runs of suite D |
| `sp_…`, `C-…`, `E1-…`, `E2-…` in `analysis_main_v1/supplement_four_runs_20260923/` | runs and controls of Appendices C.2 (strict forward), D.5 (reduced precision) and F (single channel); unrelated to the codes of Appendix I |
| `rb_s1337_tail_resolved_{prob,weight}_100m` | tail-policy runs (Appendix D.4) |
| `rb_s{7,42}_lerp_k32_detach_100m`, `LERP-Detach_K32_s{7,42}` | detach replication (Appendix F) |
| `rb-<code>-NN` in `analysis_main_v1/rb-rebuttal_v5/` | runs of Appendix I; the code is the manifest field `experiment` (section 2.2) |
| `E3a-i`, `E3a-ii`, `attn_variant=fp`, `attn_variant=u8` | IndexSoftmax with (i) float normalization and (ii) UINT8 probabilities |
| `E3b-d`, `E3b-g`, `attn_variant=detach`, `attn_variant=graph` | EXAQ with the window statistics detached (d) or in the graph (g) |
| `W→W`, `W→P`, `P→P`, `P→W`, `freeze-h` | surrogate before → after the switch, frozen-step branch (Appendix I.3) |
| `centroid`, `Centroid`, `cv1_…_centroid_…` in `analysis_main_v1/centroid_v1_20261013/` | bin-mean reconstruction (Appendix J) |
| `centroid2`, `Centroid-2` | second variant of Appendix J.1 (log-mean-exp from the mean and variance of a bin) |
| `nearest_centroid_bwd`, `NfCb`, `cv1b_s7_fw_nearestfwd_centroidbwd_k4_tau6_100m` | hard Nearest forward with the bin-mean backward (Appendix J.3) |
| `cells`, `--cells nearest` | the K+1 bins of Appendix J.1 (the rounding bins of Nearest) |

The surrogate of a run is identified by its saved configuration together with the SHA-256 of its attention code
(`quant_attention.py`). The manifests of suites A–D give the code root of each run; the SHA-256 of each code root is
listed in this section; every `summary.json` also records the SHA-256 of the code used for its evaluation. The run
manifest of Appendix I records the SHA-256 per run.

| Code root | Operators | SHA-256 of `quant_attention.py` |
|---|---|---|
| `training_code/phaseB_base/` (also `training_code/formal_code_factorial250_v1/formal_base_v2/`) | softmax, LERP, MinMax–Weight | `9d03d7a6d240c80c4ac01eddaaec3581a65aec952daf6d8101205bc832af0704` |
| `training_code/phaseB_base/code/phaseB_rr/` (also `training_code/formal_code_factorial250_v1/formal_qrm_v1/`) | FWM–Weight | `a402460d4545a76e19a65f858d4bee247fec26309431a94fd112670176546367` |
| `training_code/formal_code_factorial250_v1/formal_nearest_probste_v1/` | MinMax–Prob | `0e20aea0750abcaacae9eef17f7f3ffc3cfb36a2cfe25f23ac3af7b407176dce` |
| `training_code/formal_code_factorial250_v1/formal_qrm_probste_v1/` | FWM–Prob | `ce2bd02bc34defb1997f8233e6fc2bf054818b7288fe5c8592b3d28ed0fa3c2d` |
| `training_code/legacy_exact_fp32/` | historical detach root | `3da199f854b97c6460577283af09ee4fa5bde1e1036c1585263d616801f0d4de` |
| `training_code/formal_lerp_k1_v1/` | LERP K = 1 | `b72a1569aa6b18b72beed67aea88a137cdf9bd16a31f7f5519ecaea5d5b19b85` |
| `analysis_main_v1/supplement_four_runs_20260923/code/e1_lerp_range_grad_single/` | single channel | `c1f8d2ceffa46b41f81d75e508565bdc827164646ece40b06f4c91b2f919bd6e` |
| `analysis_main_v1/supplement_four_runs_20260923/code/e2_fwm_probste_strict/` | strict forward, FWM (Prob-STE and Weight-STE by argument) | `98fe5b931f83ba5f4621bdff6a5aa4a033a0b910a0fbffaebdc20cd40244e338` |
| `analysis_main_v1/supplement_four_runs_20260923/code/e2_minmax_probste_strict/` | strict forward, MinMax (Prob-STE and Weight-STE by argument) | `92291324e4b9d90441d42b0c2824b744ec1896628b372628684d241673df4b88` |
| `analysis_main_v1/supplement_four_runs_20260923/code/r3_lowprec_softmax/` | reduced-precision baselines | `075e19c746ba45fa1b79c3b521d3ca8c5c58239a8db5990d83b59ed7f7a68f33` |
| `training_code/phaseB_base/code/rb_tail_prob/` | tail policy, Prob-STE | `7e42c921e1d969aba0184d18d8ef1e5013b60afc6ddf35c3846d45d74c223764` |
| `training_code/phaseB_base/code/rb_tail_weight/` | tail policy, Weight-STE | `02edd7846d6699758ebca1acd6f13771aa2d9f98c275dae0add4410a2c0eb401` |
| `analysis_main_v1/rb-rebuttal_v5/code/rb-root_v1/` | merged root of Appendix I | `d563382c6a50d0e342e63deb1d90867bc05f4aaa29095aa247ae13ea607613e7` |
| `analysis_main_v1/centroid_v1_20261013/code/centroid_root/` | bin-mean reconstruction and its second variant (Appendix J); the other operators as in `training_code/phaseB_base/code/phaseB_rr/` | `23f92c06b13e9a589b3cfe732a4f0b616791e46a62228cc49f5595ed9d1357b0` |
| `analysis_main_v1/centroid_v1_20261013/code/centroid_root_v2/` | hard Nearest forward with the bin-mean backward (Appendix J.3) | `612e7442a8688ab7c6eaff09ba37cc63408234008bb3c6df72d57f96dfd1c7e8` |

## 2. Runs: manifests, code and result files

### 2.1 Runs of Table S26 (202 training runs)

| Group (paper) | Runs | Manifest | Training code | Result files (per checkpoint: `per_block_nll.npy`, `summary.json`, `geometry_layers.csv`) |
|---|---|---|---|---|
| Suite A: 124M @ 2.5B, FineWebEdu-3B, seed 1337 | 15 | `analysis_main_v1/manifests/primary_37_runs.tsv`, `primary_37_checkpoints.tsv`, `eval_jobs.tsv` | `training_code/formal_code_factorial250_v1/` (four roots), `training_code/formal_lerp_k1_v1/` | `analysis_main_v1/eval_native_v1/A_longhorizon_124m_2p5b/<run>/<checkpoint>/` |
| Suite B: 124M @ 250M, FineWebEdu-3B, seed 1337 | 10 | same | `training_code/formal_code_factorial250_v1/` | `analysis_main_v1/eval_native_v1/B_factorial_124m_250m/` |
| Suite C: 1B @ 100M, WikiText-103, seed 1337 | 12 | same | `training_code/formal_code_factorial250_v1/` | `analysis_main_v1/eval_native_v1/C_scale_1b_100m/` |
| Suite D: 124M @ 100M, WikiText-103, five seeds | 130 | `analysis_main_v1/D_robustness_124m_100m_5seed/manifests/five_seed_wikitext_130.tsv`, `five_seed_checkpoints_130.tsv` | `training_code/phaseB_base/` (softmax, LERP, MinMax–Weight), `training_code/phaseB_base/code/phaseB_rr/` (FWM–Weight), `training_code/formal_code_factorial250_v1/formal_{nearest,qrm}_probste_v1` (Prob-STE; evaluation snapshots in `analysis_main_v1/D_robustness_124m_100m_5seed/code_snapshots/`) | `analysis_main_v1/D_robustness_124m_100m_5seed/eval_native_v1/<label>_K<k>_s<seed>/` |
| Historical detach intervention (App. F) | 16 | `analysis_main_v1/E_detach_history/detach_history_endpoints.tsv` | `training_code/legacy_exact_fp32/`, `training_code/phaseB_base/` (with `p1_detach_proj.patch`) | `analysis_main_v1/E_detach_history/eval_native_v1/`; logged curves in `training_code/*/runs/` |
| Detach replication, LERP K = 32, seeds 7/42 (App. F) | 2 | `analysis_main_v1/D_robustness_124m_100m_5seed/manifests/detach_replication_runs.tsv`, `detach_replication_checkpoints.tsv` | `training_code/phaseB_base/` | `analysis_main_v1/D_robustness_124m_100m_5seed/eval_native_v1/LERP-Detach_K32_s{7,42}/` |
| Single-channel and strict-forward runs (App. F, C.2) | 8 | `analysis_main_v1/supplement_four_runs_20260923/manifests/supplement_four_runs_manifest.tsv` | `analysis_main_v1/supplement_four_runs_20260923/code/e1_lerp_range_grad_single/`, `e2_fwm_probste_strict/`, `e2_minmax_probste_strict/` | `analysis_main_v1/supplement_four_runs_20260923/eval/rtx3090/`, `eval/rtx5090/` |
| Tail-policy runs (App. D.4) | 2 | `analysis_main_v1/tail_policy/tail_decomp/tail_decomp_seed1337.tsv` | `training_code/phaseB_base/code/rb_tail_prob/`, `rb_tail_weight/` | `analysis_main_v1/D_robustness_124m_100m_5seed/eval_native_v1/rb_s1337_tail_resolved_{prob,weight}_100m/` |
| Bin-mean control (App. J) | 5 | `analysis_main_v1/centroid_v1_20261013/manifests/protocol_centroid_v1.json` (four runs trained from scratch), `addendum_nearest_fwd_centroid_bwd.json` (the forward–backward decomposition run) | `analysis_main_v1/centroid_v1_20261013/code/centroid_root/`, `code/centroid_root_v2/` | `analysis_main_v1/centroid_v1_20261013/eval/rtx3090/<run>/<checkpoint>/` |
| Reduced-precision runs (App. D.5) | 2 | `analysis_main_v1/supplement_four_runs_20260923/manifests/round3_lowprec_softmax_addendum.json` | `analysis_main_v1/supplement_four_runs_20260923/code/r3_lowprec_softmax/` (derived from `training_code/phaseB_base/code/rb_lowprec/`) | `analysis_main_v1/supplement_four_runs_20260923/eval/rtx3090/sp_s7_softmax_{bf16exp,fp8e4m3exp}_100m/` |

Evaluation code: `analysis_main_v1/code/eval_worker.py` (suites A–C, jobs from `eval_jobs.tsv`), `eval_worker_5seed.py`
(suite D, detach replication, tail policy), `eval_detach_history.py` (App. F record),
`analysis_main_v1/supplement_four_runs_20260923/code/eval/sup_eval.py` (runs of App. C.2, D.5, F),
`analysis_main_v1/centroid_v1_20261013/code/eval/cv1_eval.py` and `code/eval_v2/cv1_eval.py` (runs of App. J). Analysis code:
`analyze_primary.py`, `analyze_5seed.py`, `analyze_detach_history.py` in `analysis_main_v1/code/`; `analyze_supplement.py`,
`analyze_e1_seed.py`, `analyze_r3_lowprec.py` in `analysis_main_v1/supplement_four_runs_20260923/code/`;
`analysis_main_v1/tail_policy/tail_decomp/analyze_tail_decomp.py`; `analyze_training.py` and `analyze_addendum.py` in
`analysis_main_v1/centroid_v1_20261013/code/analysis/` (section 2.3).
The evaluation and analysis code identify runs through the manifests above. `supplement_four_runs_manifest.tsv` lists the
8 runs together with the 7 runs they are compared with (status `historical, frozen`: runs of suite D and of the detach
replication). `tail_decomp_seed1337.tsv` lists the 2 tail-policy runs together with the 5 suite-D runs they are compared
with. `detach_history_endpoints.tsv` has one row per evaluation record (28 records of 16 runs).

### 2.2 Runs of Table S28 (Appendix I, 98 training runs)

Manifest: `analysis_main_v1/rb-rebuttal_v5/manifests/rb-runs_v5_manifest.json` (and `.tsv`): one row per run with its
code (`experiment`), configuration, exact command, the SHA-256 of `quant_attention.py`, `model.py` and `train_formal.py`,
and the fields `prediction` and `reading_rule`. Training code: `analysis_main_v1/rb-rebuttal_v5/code/rb-root_v1/` (97
runs; `base_snapshot/` holds the code it extends) and `code/c2_roots/e2_minmax_probste_strict/` (rb-E7-02). The
following paths are relative to `analysis_main_v1/rb-rebuttal_v5/`. Start and finish of every run:
`manifests/rb-run_log_machine-A.jsonl` and `rb-run_log_machine-B.jsonl`; size and SHA-256 of the checkpoint files:
`manifests/ckpt_sha256_machine-A.tsv` and `ckpt_sha256_machine-B.tsv` (on each machine the two files were named
`manifests/rb-run_log.jsonl` and `runs/_ckpt_sha256.tsv`; the evaluation records cite the second under that name).
`runs/` has the layout of the output directory
of the training and evaluation code (`RB_RUNS`), without the checkpoints: environment and code hashes as run in
`runs/<run>/env.json` (97 runs; the root of rb-E7-02 does not write this file); per evaluated checkpoint
`runs/_eval/nll/<run>/<checkpoint>/per_block_nll.npy` and `summary.json` (with the SHA-256 of the checkpoint read; the
field `host` names the machine of the evaluation); attention probes in `runs/_eval/probe/`; benchmarks (per
item) and held-out corpora (per block) in `runs/_eval/downstream/`. The result tables of Appendix I are in `reports/rb-dossier_20261114/csv/`
(`runs.csv`: one row per run; `comparisons.csv`: every paired difference with its 95% interval, per seed).

| Code (`rb-` + manifest field `experiment`) | Run ids | Runs | Protocol | Seeds | Trained on | Code root | Row of Table S28 | Appendix |
|---|---|---|---|---|---|---|---|---|
| rb-C0 | rb-C0-01 … rb-C0-03 | 3 | suiteD 124M@100M WikiText-103 | 7, 42, 1337 | B | `code/rb-root_v1` | same-stack softmax controls (100M, three seeds) | I.1 |
| rb-C0b | rb-C0b-01 | 1 | suiteB 124M@250M FineWebEdu | 1337 | B | `code/rb-root_v1` | same-stack softmax control (250M) | I.1 |
| rb-E1 | rb-E1-01 … rb-E1-08 | 8 | suiteB 124M@250M FineWebEdu; suiteD 124M@100M WikiText-103 | 7, 42, 1337 | B | `code/rb-root_v1` | FWM–LERP, 100M and 250M | I.2 |
| rb-E1b | rb-E1b-01 | 1 | suiteA 124M@2.5B FineWebEdu | 1337 | A | `code/rb-root_v1` | FWM–LERP K=4 at 2.5B | I.2 |
| rb-E11 | rb-E11-01 … rb-E11-09 | 9 | suiteB 124M@250M FineWebEdu; suiteD 124M@100M WikiText-103 | 7, 42, 1337 | B | `code/rb-root_v1` | row maximum detached under FWM | I.2 |
| rb-E2a/b | rb-E2ab-01 … rb-E2ab-10 | 10 | suiteB 124M@250M FineWebEdu | 1337 | B | `code/rb-root_v1` | surrogate switch, frozen step | I.3 |
| rb-E2c | rb-E2c-01 … rb-E2c-03 | 3 | suiteA 124M@2.5B FineWebEdu | 1337 | B | `code/rb-root_v1` | the same on the 2.5B schedule | I.3 |
| rb-E2d | rb-E2d-01 … rb-E2d-12 | 12 | suiteD 124M@100M WikiText-103 | 7, 42, 253 | B | `code/rb-root_v1` | surrogate switch on three seeds | I.3 |
| rb-E4 | rb-E4-01 … rb-E4-08 | 8 | suiteB 124M@250M FineWebEdu; suiteD 124M@100M WikiText-103 | 7, 42, 1337 | B | `code/rb-root_v1` | per-head exponential-moving-average (EMA) window | I.4 |
| rb-E5 | rb-E5-01 … rb-E5-18 | 18 | suiteB 124M@250M FineWebEdu; suiteD 124M@100M WikiText-103 | 7, 42, 1337 | B | `code/rb-root_v1` | rounding rules of five formats, incl. MXFP8, NVFP4 | I.5 |
| rb-E5n | rb-E5n-01 … rb-E5n-04 | 4 | suiteB 124M@250M FineWebEdu; suiteD 124M@100M WikiText-103 | 7, 42, 1337 | B | `code/rb-root_v1` | bf16 row sum and division | I.5 |
| rb-E3b | rb-E3b-01 … rb-E3b-08 | 8 | suiteB 124M@250M FineWebEdu; suiteD 124M@100M WikiText-103 | 7, 42, 1337 | B | `code/rb-root_v1` | EXAQ, statistics detached or in the graph | I.6 |
| rb-E3a | rb-E3a-01 … rb-E3a-08 | 8 | suiteB 124M@250M FineWebEdu; suiteD 124M@100M WikiText-103 | 7, 42, 1337 | B | `code/rb-root_v1` | IndexSoftmax, (i) float, (ii) UINT8 normalization | I.7 |
| rb-E3a-2.5B | rb-E3a-2.5B-01 | 1 | suiteA 124M@2.5B FineWebEdu | 1337 | A | `code/rb-root_v1` | IndexSoftmax at 2.5B | I.7 |
| rb-E8a | rb-E8a-01 | 1 | suiteA 124M@2.5B FineWebEdu | 7 | A | `code/rb-root_v1` | second softmax seed at 2.5B | I.9 |
| rb-E8c | rb-E8c-01 | 1 | suiteA 124M@2.5B FineWebEdu | 7 | A | `code/rb-root_v1` | second LERP K=16 seed at 2.5B | I.9 |
| rb-E7 | rb-E7-01 … rb-E7-02 | 2 | suiteD 124M@100M WikiText-103 | 7 | B | `code/c2_roots/e2_minmax_probste_strict`, `code/rb-root_v1` | TF32 off in training | I.11 |

Parts of Appendix I without training runs: rb-E0b (App. I.8; `csv/e0b_trained_endpoints.csv`, from the attention probes
`runs/_eval/probe/`, code `code/eval/rb_probe.py`), rb-E0a, rb-E0c and rb-E0d (App. I.10; `csv/e0a_accuracy.csv`,
`csv/e0c_bitwidth.csv`, `csv/e0d_roofline.csv`), rb-E6 (App. I.11; `csv/e6.csv`, `runs/_eval/e6/` (RTX 5090),
`runs/_eval/e6_rtx3090/` (RTX 3090), code `code/eval/rb_e6.py`).

### 2.3 Runs of Appendix J (bin-mean control, 5 training runs)

The following paths are relative to `analysis_main_v1/centroid_v1_20261013/`. All five runs: 124M, WikiText-103, 100M tokens, seed 7, K = 4, five
checkpoints each, trained and evaluated on the local RTX 3090 (stack S3 of Table S4).

| Run | Operator | Code root | Paper |
|---|---|---|---|
| `cv1_s7_fw_centroid_k4_tau6_100m` | FWM bin-mean (`--approx centroid --range_mode relevant_zero --range_tau 6`) | `code/centroid_root` | Table S54, App. J.2 |
| `cv1_s7_minmax_centroid_k4_100m` | MinMax bin-mean (`--approx centroid --range_mode minmax`) | `code/centroid_root` | Table S54, App. J.2 |
| `cv1_s7_fw_centroid2_k4_tau6_100m` | FWM, second variant (`--approx centroid2`) | `code/centroid_root` | App. J.1 (endpoint difference to FWM bin-mean) |
| `cv1_s7_minmax_centroid2_k4_100m` | MinMax, second variant (`--approx centroid2`) | `code/centroid_root` | counted in Table S26; not reported further (App. J.1) |
| `cv1b_s7_fw_nearestfwd_centroidbwd_k4_tau6_100m` | FWM, hard Nearest forward with the bin-mean backward (`--approx nearest_centroid_bwd`) | `code/centroid_root_v2` | Table S55, App. J.3 |

Commands of the runs: `code/jobs/run_training_queue.sh` (four runs) and `code/jobs/run_addendum_nfcb.sh` (the fifth run and
its evaluation); `code/jobs/eval_after_run.sh` evaluates a finished run. Training logs:
`training_code/phaseB_base/centroid_runs_20261013/<run>/` (`log.csv`, `attention_stats.csv`, `occupancy.csv`) and
`logs/train_<run>.log`. Evaluation results per checkpoint: `eval/rtx3090/<run>/<checkpoint>/` (`per_block_nll.npy`,
`summary.json`, `geometry_layers.csv`). The control rows of Tables S54 and S55 are runs of suite D and of Appendix C.2
(seed 7), read from `analysis_main_v1/D_robustness_124m_100m_5seed/eval_native_v1/` and
`analysis_main_v1/supplement_four_runs_20260923/eval/rtx3090/`.

Records written before training: `manifests/protocol_centroid_v1.json` (protocol, statistics and decision labels of the
four runs and of the frozen-model analyses), `manifests/PRETRAINING_RECORD.txt` (SHA-256 of the protocol, code and check
files before the first run), `manifests/addendum_nearest_fwd_centroid_bwd.json` and
`manifests/PRETRAINING_RECORD_addendum.txt` (the same for the fifth run, with the rule by which Appendix J.3 reads the
backward share). `manifests/method_description_input.md` is the method description that the protocol refers to (in
Chinese). The files of the two code roots, `code/eval*/common.py`, `code/eval*/geometry_probe_v1.py`,
`checks/check_gpu_step0.json`, `checks/check_v2_nearest_centroid_bwd.json`, the protocol and the method description
have the SHA-256 listed in the two records. The other listed files differ from the recorded ones only by the paths
of this repository: `code/eval*/cv1_eval.py`, `code/checks/*.py` and `code/jobs/*.sh` (root of the repository from
`KI_REPO`, `<LOCAL_HOME>`, the directory name `eval/rtx3090`), `checks/check_centroid_ops.json` (field `new_root`) and
the addendum manifest (one machine label). In the records and logs, `eval/local3090` is the directory `eval/rtx3090` of
this repository.
The protocol also registers a frozen-model evaluation on Qwen2.5-1.5B (`part_Q_qwen`); the paper does not report it
and its code and results are not included.

## 3. Evaluations and checks without training

| Item (paper) | Code | Result files |
|---|---|---|
| Downstream evaluation of the 2.5B suite (App. C.3, E): pre-declaration | — | `analysis_main_v1/downstream_v1/REPORT.md`, section 1 |
| Downstream: held-out corpora (layer 1), probes (layer 2), benchmarks (layer 3) | `analysis_main_v1/downstream_v1/code/` (`layer1_*.py`, `layer2_*.py`, `layer3_*.py`, `stats.py`) | `analysis_main_v1/downstream_v1/results/` (summary tables of the benchmarks: `results/layer3/stageC/tables/`), probe definitions `analysis_main_v1/downstream_v1/data/` |
| Selection of τ (App. C.3) | `analysis_main_v1/provenance/tau_selection/code/` | `analysis_main_v1/provenance/tau_selection/results/`, `baseline_code_sha256.txt` |
| Rounding audit (App. C.2) | `analysis_main_v1/rounding_audit/measure_rounding.py` | `analysis_main_v1/rounding_audit/rounding_full_split.csv`, `per_block/` |
| Frozen-endpoint audit and gradient decomposition (App. C.2, F) | `analysis_main_v1/supplement_four_runs_20260923/round2/code/` | `analysis_main_v1/supplement_four_runs_20260923/round2/task1/`, `task2/`, `task2_s42/` |
| Step-0 check of the frozen-step parents (App. I.3) | `analysis_main_v1/rb-rebuttal_v5/reports/e2c_step0_check/code/` | `analysis_main_v1/rb-rebuttal_v5/reports/e2c_step0_check/e2c_step0_check.csv` (the I.3 numbers), `eval_out/` |
| Re-evaluation of the suite-A softmax endpoint on the 8-GPU machine (App. I.1) | `analysis_main_v1/rb-rebuttal_v5/code/eval/rb_eval_gate.py` | `analysis_main_v1/rb-rebuttal_v5/runs/_eval/gates/`, `analysis_main_v1/rb-rebuttal_v5/checks/13_2_reeval_machine-B.json` |
| Score-gradient alignment on frozen models (App. J.4) | `analysis_main_v1/centroid_v1_20261013/code/analysis/o2_rows_grads.py` (needs the checkpoints), `analyze_o2.py`, operator library `ops_ref.py` | `analysis_main_v1/centroid_v1_20261013/results/o2_rows_grads/` (`table_exact100.csv`: Table S56) |
| Frozen-model substitution (App. J.5) | `analysis_main_v1/centroid_v1_20261013/code/analysis/o1_substitution.py` (needs the checkpoints), `analyze_o1.py` | `analysis_main_v1/centroid_v1_20261013/results/o1_substitution/` (per seed and configuration: per-block NLL; `summary.csv`: Table S57) |

The per-item outputs of the downstream evaluation of the 2.5B suite (`results/layer3/**/*.csv.gz`) are not included
because of their size; the summary tables are included, and the commands that regenerate the per-item outputs are in
section 6.1. The only per-item files of the 2.5B suite included are the 42 that the Appendix I scripts read
(`results/layer3/stageC/<condition>/<benchmark>.csv.gz` for softmax, LERP K = 4 and 16, FWM–Weight K = 4 and 16 and
FWM–Prob K = 4, the conditions that Appendix I compares with). The per-item outputs of the Appendix I runs are included
(`analysis_main_v1/rb-rebuttal_v5/runs/_eval/downstream/stageC/<run>/<benchmark>.csv.gz`; the directory name is the run id
without hyphens, for example `rbE1b01`, and `Exact_s1337_check` is the suite-A softmax endpoint evaluated again on the
8-GPU machine).
In these 77 per-item files only the columns that the statistics read are kept; the other columns are omitted because
of their size. Kept, in file order: multiple-choice benchmarks (ARC-Easy, BLiMP, HellaSwag, PIQA, SciQ with and without the support passage): `subtask`, `doc_id`, `doc_hash`, `choice_idx`, `gold_idx`, `acc_correct`, `accnorm_correct`, `margin_token_norm`; LAMBADA: `subtask`, `doc_id`, `doc_hash`, `n_scored_tokens`, `target_nll`, `token_norm_target_nll`, `greedy_correct`. Rows and their order are unchanged and the kept fields are
verbatim. Omitted: the labels `condition`, `cfg`, `task`, `split`, the hashes `input_hash`, `cont_tok_hash`, the
lengths `n_choices`, `cont_chars`, `choice_chars`, `target_delimiter_chars`, `target_chars`, `n_scored_tokens` (multiple
choice), the raw scores `raw_total_logprob`, `token_norm_logprob`, the harness fields `pred_acc_idx`, `pred_accnorm_idx`,
`harness_acc`, `harness_acc_norm`, the gold flag `is_gold` (multiple choice) and the run fields `truncated`, `status`, `retries`.

## 4. Tables and figures: generator and source files

Generated tables and figures are written by five scripts (commands in section 6.3). The source files below are the
files whose data enter that table or figure, including files that the script loads once at start. Hand-written tables have no generator; their content comes from the files named.

The paper has 82 tables and figures: Figures 1–5 and Tables 1–5 in the main text, Tables S1–S57 and Figures S1–S15 in the appendices.

| Item | Label | Output | Generator | Source files |
|---|---|---|---|---|
| Figure 1 | fig:schematic | `fig1_operator_schematic.pdf` | `paper/figures/make_paper_figures.py` | (constants in the generator) |
| Figure 2 | fig:detach | `fig2_detach_vs_full.pdf` | `paper/figures/make_paper_figures.py` | `analysis_main_v1/E_detach_history/dense_logged_gap_detach_vs_full.csv`; `analysis_main_v1/E_detach_history/endpoint_detach_vs_full_s1337.csv`; `analysis_main_v1/detach_replication/detach_replication_paired.csv`; `analysis_main_v1/supplement_four_runs_20260923/reports/supplement_e1_seed42.json`; `analysis_main_v1/supplement_four_runs_20260923/reports/supplement_endpoint_contrasts.csv`; `training_code/legacy_exact_fp32/runs/exact_fp32_softmax_100m/log.csv`; `training_code/legacy_exact_fp32/runs/q32_lerp_100m/attention_stats.csv`; `training_code/phaseB_base/runs/q32_lerp_detachproj_100m/attention_stats.csv`; `training_code/phaseB_base/runs/q32_lerp_detachproj_100m/log.csv`; `training_code/phaseB_base/runs/q32_lerp_fullgrad_100m/attention_stats.csv` |
| Figure 3 | fig:2x2 | `fig3_factorial_2x2.pdf` | `paper/figures/make_paper_figures.py` | `analysis_main_v1/B_factorial_124m_250m/endpoint_fullval/endpoint_table.csv`; `analysis_main_v1/D_robustness_124m_100m_5seed/analysis_v1/delta_by_seed.csv`; `analysis_main_v1/D_robustness_124m_100m_5seed/analysis_v1/endpoint_by_condition.csv` |
| Figure 4 | fig:k | `fig4_k_scaling_horizon.pdf` | `paper/figures/make_paper_figures.py` | `analysis_main_v1/A_longhorizon_124m_2p5b/endpoint_fullval/endpoint_table.csv`; `analysis_main_v1/A_longhorizon_124m_2p5b/trajectory/contrast_trajectories.csv`; `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/trajectories.csv` |
| Figure 5 | fig:heat | `fig9_downstream_heatmap.pdf` | `paper/figures/make_paper_figures.py` | `analysis_main_v1/A_longhorizon_124m_2p5b/endpoint_fullval/endpoint_table.csv`; `analysis_main_v1/downstream_v1/results/layer3/stageC/tables/paired_comparisons.csv`; `analysis_main_v1/downstream_v1/results/layer3/stageC/tables/point_estimates.csv` |
| Table 1 | tab:labels | — (hand-written in the LaTeX source) | — | definitions (§3); no numbers from result files |
| Table 2 | tab:notation | — (hand-written in the LaTeX source) | — | notation (§3); no numbers from result files |
| Table 3 | tab:suites | — (hand-written in the LaTeX source) | — | suite definitions; run counts from the suite manifests (`analysis_main_v1/manifests/primary_37_runs.tsv`, `analysis_main_v1/D_robustness_124m_100m_5seed/manifests/five_seed_wikitext_130.tsv`) |
| Table 4 | tab:calgrad | `tab_calgrad.tex` | `paper/tables/make_rb_tables.py` | `analysis_main_v1/E_detach_history/endpoint_detach_vs_full_s1337.csv`; `analysis_main_v1/E_detach_history/geometry_detach_vs_full_s1337.csv`; `analysis_main_v1/detach_replication/detach_replication_paired.csv`; `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/comparisons.csv`; `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/probes_layer_mean.csv`; `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/seed_means.csv`; `analysis_main_v1/supplement_four_runs_20260923/reports/supplement_e1_seed42.json`; `analysis_main_v1/supplement_four_runs_20260923/reports/supplement_endpoint_contrasts.csv`; `analysis_main_v1/supplement_four_runs_20260923/reports/supplement_trajectories.csv` |
| Table 5 | tab:cross | `tab_cross_suite_short.tex` | `paper/tables/make_paper_tables.py` | `analysis_main_v1/A_longhorizon_124m_2p5b/endpoint_fullval/endpoint_contrasts.csv`; `analysis_main_v1/B_factorial_124m_250m/endpoint_fullval/endpoint_contrasts.csv`; `analysis_main_v1/C_scale_1b_100m/endpoint_fullval/endpoint_contrasts.csv`; `analysis_main_v1/D_robustness_124m_100m_5seed/analysis_v1/delta_by_seed.csv`; `analysis_main_v1/D_robustness_124m_100m_5seed/analysis_v1/factorial_calibration_backward.csv` |
| Table S1 | tab:edge | — (hand-written in the LaTeX source) | — | conventions of the attention code; checked by `paper/check_degenerate_rows.py` |
| Table S2 | tab:dstack | `tab_dstack.tex` | `paper/tables/make_paper_tables.py` | `analysis_main_v1/D_robustness_124m_100m_5seed/manifests/five_seed_wikitext_130.tsv` |
| Table S3 | tab:rounding | `tab_rounding_full_split.tex` | `paper/tables/make_paper_tables.py` | `analysis_main_v1/rounding_audit/rounding_full_split.csv` |
| Table S4 | tab:hardware | `tab_hardware.tex` | `paper/tables/make_paper_tables.py` | `analysis_main_v1/D_robustness_124m_100m_5seed/manifests/five_seed_wikitext_130.tsv`; `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/compute_per_experiment.csv`; `analysis_main_v1/training_logs/A_longhorizon_124m_2p5b/*/` (15 files: training_health.csv); `analysis_main_v1/training_logs/B_factorial_124m_250m/*/` (10 files: training_health.csv); `analysis_main_v1/training_logs/C_scale_1b_100m/*/` (12 files: training_health.csv); `training_code/legacy_exact_fp32/runs/exact_fp32_softmax_100m/log.csv`; `training_code/phaseB_base/formal_runs_probste_5seeds_100m_v1/*/` (10 files: log.csv); `training_code/phaseB_base/formal_runs_probste_k64_5seeds_100m_v1/*/` (10 files: training_health.csv); `training_code/phaseB_base/centroid_runs_20261013/*/` (5 files: log.csv); `training_code/phaseB_base/formal_runs_rerun_wikitext_124m_5seed_20260917/*/` (40 files: log.csv); `training_code/phaseB_base/qrm_runs_v1/*/` (15 files: log.csv); `training_code/phaseB_base/runs/*/` (54 files: log.csv); `training_code/phaseB_base/runs_20260921/rb_s1337_tail_resolved_prob_100m/log.csv`; `training_code/phaseB_base/runs_20260921/rb_s1337_tail_resolved_weight_100m/log.csv`; `training_code/phaseB_base/runs_20260921/rb_s42_lerp_k32_detach_100m/log.csv`; `training_code/phaseB_base/runs_20260921/rb_s7_lerp_k32_detach_100m/log.csv`; `training_code/phaseB_base/supplement_runs_20260923/sp_s42_lerp_k32_maxonly_100m/log.csv`; `training_code/phaseB_base/supplement_runs_20260923/sp_s42_lerp_k32_minonly_100m/log.csv`; `training_code/phaseB_base/supplement_runs_20260923/sp_s7_lerp_k32_maxonly_100m/log.csv`; `training_code/phaseB_base/supplement_runs_20260923/sp_s7_lerp_k32_minonly_100m/log.csv` |
| Table S5 | tab:strictaudit | `tab_strict_audit.tex` | `paper/tables/make_paper_tables.py` | `analysis_main_v1/supplement_four_runs_20260923/round2/task1/prob_native_vs_strict.json`; `analysis_main_v1/supplement_four_runs_20260923/round2/task1/weight_native_vs_strict.json` |
| Table S6 | tab:strict | `tab_strict_forward.tex` | `paper/tables/make_paper_tables.py` | `analysis_main_v1/supplement_four_runs_20260923/reports/supplement_endpoint_contrasts.csv` |
| Table S7 | tab:crossfull | `tab_cross_suite.tex` | `paper/tables/make_paper_tables.py` | `analysis_main_v1/A_longhorizon_124m_2p5b/endpoint_fullval/endpoint_contrasts.csv`; `analysis_main_v1/B_factorial_124m_250m/endpoint_fullval/endpoint_contrasts.csv`; `analysis_main_v1/C_scale_1b_100m/endpoint_fullval/endpoint_contrasts.csv`; `analysis_main_v1/D_robustness_124m_100m_5seed/analysis_v1/delta_by_seed.csv`; `analysis_main_v1/D_robustness_124m_100m_5seed/analysis_v1/endpoint_by_condition.csv`; `analysis_main_v1/D_robustness_124m_100m_5seed/analysis_v1/factorial_calibration_backward.csv` |
| Table S8 | tab:seedfact | `tab_five_seed_factorial.tex` | `paper/tables/make_paper_tables.py` | `analysis_main_v1/D_robustness_124m_100m_5seed/analysis_v1/factorial_calibration_backward.csv` |
| Table S9 | tab:ktrend | `tab_five_seed_ktrend.tex` | `paper/tables/make_paper_tables.py` | `analysis_main_v1/D_robustness_124m_100m_5seed/analysis_v1/k_trend.csv` |
| Table S10 | tab:taildecomp | `tab_tail_decomp.tex` | `paper/tables/make_paper_tables.py` | `analysis_main_v1/tail_policy/tail_decomp/tail_decomp_contrasts.csv`; `analysis_main_v1/tail_policy/tail_decomp/tail_decomp_ratio.csv` |
| Table S11 | tab:lowprec | `tab_lowprec.tex` | `paper/tables/make_paper_tables.py` | `analysis_main_v1/supplement_four_runs_20260923/round3/reports/r3_lowprec_contrasts.csv` |
| Table S12 | tab:endpoints | `tab_endpoints_all.tex` | `paper/tables/make_paper_tables.py` | `analysis_main_v1/A_longhorizon_124m_2p5b/endpoint_fullval/endpoint_table.csv`; `analysis_main_v1/B_factorial_124m_250m/endpoint_fullval/endpoint_table.csv`; `analysis_main_v1/C_scale_1b_100m/endpoint_fullval/endpoint_table.csv` |
| Table S13 | tab:seedend | `tab_five_seed_endpoints.tex` | `paper/tables/make_paper_tables.py` | `analysis_main_v1/D_robustness_124m_100m_5seed/analysis_v1/endpoint_by_condition.csv`; `analysis_main_v1/D_robustness_124m_100m_5seed/analysis_v1/seed_noise.csv` |
| Table S14 | tab:nearci | `tab_layer1_nearbaseline_ci.tex` | `paper/tables/make_paper_tables.py` | `analysis_main_v1/downstream_v1/results/layer1/ppl_transfer.csv` |
| Table S15 | tab:subst | `tab_downstream_substitution.tex` | `paper/tables/make_paper_tables.py` | `analysis_main_v1/A_longhorizon_124m_2p5b/endpoint_fullval/endpoint_table.csv`; `analysis_main_v1/downstream_v1/results/layer1/ppl_transfer.csv` |
| Table S16 | tab:stagec | `tab_downstream_stageC.tex` | `paper/tables/make_paper_tables.py` | `analysis_main_v1/A_longhorizon_124m_2p5b/endpoint_fullval/endpoint_table.csv`; `analysis_main_v1/downstream_v1/results/layer3/stageC/tables/paired_comparisons.csv`; `analysis_main_v1/downstream_v1/results/layer3/stageC/tables/point_estimates.csv` |
| Table S17 | tab:probeab | `tab_downstream_probeAB.tex` | `paper/tables/make_paper_tables.py` | `analysis_main_v1/A_longhorizon_124m_2p5b/endpoint_fullval/endpoint_table.csv`; `analysis_main_v1/downstream_v1/results/layer2/probe_scores.csv`; `analysis_main_v1/downstream_v1/results/layer2/probe_scores_A_all_familyboot_20260925.csv` |
| Table S18 | tab:probec | `tab_downstream_probeC.tex` | `paper/tables/make_paper_tables.py` | `analysis_main_v1/A_longhorizon_124m_2p5b/endpoint_fullval/endpoint_table.csv`; `analysis_main_v1/downstream_v1/results/layer2/diagnostics/probeC_length_trend.csv`; `analysis_main_v1/downstream_v1/results/layer2/probe_scores.csv` |
| Table S19 | tab:detach | `tab_detach.tex` | `paper/tables/make_paper_tables.py` | `analysis_main_v1/E_detach_history/endpoint_detach_vs_full_s1337.csv` |
| Table S20 | tab:graddiag | — (hand-written in the LaTeX source) | — | outputs of the diagnostic tests in the training code roots: `training_code/legacy_exact_fp32/test_A1_forward_shift_invariance.py`, `test_A2_gradient_shift_residual.py`, `test_A3_directional_finite_difference.py`, `test_A4_local_finite_difference.py`, and `training_code/phaseB_base/test_P1_detach_proj.py` (the tests print their results; no stored output) |
| Table S21 | — | — (hand-written in the LaTeX source from the csv file in the last column) | `analysis_main_v1/code/analyze_detach_history.py` (writes the csv file) | `analysis_main_v1/E_detach_history/geometry_detach_vs_full_s1337.csv`; its inputs: the `geometry_layers.csv` files under `analysis_main_v1/E_detach_history/eval_native_v1/` (detached runs) and `analysis_main_v1/D_robustness_124m_100m_5seed/eval_native_v1/` (full-gradient runs, seed 1337) |
| Table S22 | tab:early5m | `tab_early5m.tex` | `paper/tables/make_paper_tables.py` | `analysis_main_v1/E_detach_history/early_5m_seeds_detach.csv` |
| Table S23 | tab:detachrep | `tab_detach_replication.tex` | `paper/tables/make_paper_tables.py` | `analysis_main_v1/detach_replication/detach_replication_paired.csv` |
| Table S24 | tab:graddecomp | `tab_grad_decomp.tex` | `paper/tables/make_paper_tables.py` | `analysis_main_v1/supplement_four_runs_20260923/round2/task2/grad_decomp_local_layers.csv`; `analysis_main_v1/supplement_four_runs_20260923/round2/task2_s42/grad_decomp_local_layers.csv` |
| Table S25 | tab:e1 | `tab_e1_channels.tex` | `paper/tables/make_paper_tables.py` | `analysis_main_v1/supplement_four_runs_20260923/reports/supplement_e1_seed42.json`; `analysis_main_v1/supplement_four_runs_20260923/reports/supplement_endpoint_contrasts.csv`; `analysis_main_v1/supplement_four_runs_20260923/reports/supplement_trajectories.csv` |
| Table S26 | tab:assets | — (hand-written in the LaTeX source) | — | run and checkpoint counts of the manifests listed in section 2 of this index |
| Table S27 | tab:threeops | — (hand-written in the LaTeX source) | — | constants read from the code: `analysis_main_v1/rb-rebuttal_v5/code/rb-root_v1/quant_attention.py` (FWM, IndexSoftmax, EXAQ as trained) and the IntAttention release (not included) |
| Table S28 | tab:rb-overview | `tab_rb_overview.tex` | `paper/tables/make_rb_tables.py` | `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/compute_per_experiment.csv`; `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/runs.csv` |
| Table S29 | tab:rb-e1 | `tab_rb_e1.tex` | `paper/tables/make_rb_tables.py` | `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/comparisons.csv`; `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/seed_means.csv` |
| Table S30 | tab:rb-e1b | `tab_rb_e1b.tex` | `paper/tables/make_rb_tables.py` | `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/comparisons.csv` |
| Table S31 | tab:rb-e1bdown | `tab_rb_e1b_down.tex` | `paper/tables/make_rb_tables.py` | `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/downstream_benchmarks.csv`; `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/downstream_heldout.csv` |
| Table S32 | tab:rb-e11 | `tab_rb_e11.tex` | `paper/tables/make_rb_tables.py` | `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/comparisons.csv`; `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/probes_layer_mean.csv` |
| Table S33 | tab:rb-e2ab | `tab_rb_e2ab.tex` | `paper/tables/make_rb_tables.py` | `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/comparisons.csv`; `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/probes_layer_mean.csv` |
| Table S34 | tab:rb-e2chain | `tab_rb_e2chain.tex` | `paper/tables/make_rb_tables.py` | `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/e2_analysis.csv` |
| Table S35 | tab:rb-e2d | `tab_rb_e2d.tex` | `paper/tables/make_rb_tables.py` | `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/comparisons.csv`; `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/e2d_endpoints.csv`; `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/seed_means.csv` |
| Table S36 | tab:rb-e2c | `tab_rb_e2c.tex` | `paper/tables/make_rb_tables.py` | `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/e2_analysis.csv` |
| Table S37 | tab:rb-e4 | `tab_rb_e4.tex` | `paper/tables/make_rb_tables.py` | `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/comparisons.csv`; `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/e4_historical_gap.csv`; `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/seed_means.csv` |
| Table S38 | tab:rb-e4probe | `tab_rb_e4probe.tex` | `paper/tables/make_rb_tables.py` | `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/probes_layer_mean.csv`; `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/runs.csv` |
| Table S39 | tab:rb-formats | `tab_rb_formats.tex` | `paper/tables/make_rb_tables.py` | `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/e0c_bitwidth.csv` |
| Table S40 | tab:rb-e5 | `tab_rb_e5.tex` | `paper/tables/make_rb_tables.py` | `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/comparisons.csv`; `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/runs.csv`; `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/seed_means.csv` |
| Table S41 | tab:rb-e3b | `tab_rb_e3b.tex` | `paper/tables/make_rb_tables.py` | `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/comparisons.csv`; `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/seed_means.csv` |
| Table S42 | tab:rb-e3bprobe | `tab_rb_e3bprobe.tex` | `paper/tables/make_rb_tables.py` | `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/probes_layer_mean.csv`; `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/runs.csv` |
| Table S43 | tab:rb-e3a | `tab_rb_e3a.tex` | `paper/tables/make_rb_tables.py` | `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/comparisons.csv`; `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/seed_means.csv` |
| Table S44 | tab:rb-e3a25 | `tab_rb_e3a25.tex` | `paper/tables/make_rb_tables.py` | `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/comparisons.csv`; `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/e3a_2p5b_forwards.csv` |
| Table S45 | tab:rb-e3adown | `tab_rb_e3a_down.tex` | `paper/tables/make_rb_tables.py` | `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/downstream_benchmarks.csv`; `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/downstream_heldout.csv` |
| Table S46 | tab:rb-e0b | `tab_rb_e0b.tex` | `paper/tables/make_rb_tables.py` | `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/e0b_trained_endpoints.csv` |
| Table S47 | tab:rb-e0blen | `tab_rb_e0b_len.tex` | `paper/tables/make_rb_tables.py` | `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/probes_layer_mean.csv` |
| Table S48 | tab:rb-e8 | `tab_rb_e8_down.tex` | `paper/tables/make_rb_tables.py` | `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/downstream_benchmarks.csv`; `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/downstream_heldout.csv` |
| Table S49 | tab:rb-e0a | — (hand-written in the LaTeX source) | — | operation counts read from `analysis_main_v1/rb-rebuttal_v5/code/rb-root_v1/quant_attention.py` |
| Table S50 | tab:rb-e0d | `tab_rb_e0d.tex` | `paper/tables/make_rb_tables.py` | `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/e0d_roofline.csv` |
| Table S51 | tab:rb-e0c | `tab_rb_e0c.tex` | `paper/tables/make_rb_tables.py` | `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/e0c_bitwidth.csv` |
| Table S52 | tab:rb-e0aacc | `tab_rb_e0a_acc.tex` | `paper/tables/make_rb_tables.py` | `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/e0a_accuracy.csv` |
| Table S53 | tab:rb-e6 | `tab_rb_e6.tex` | `paper/tables/make_rb_tables.py` | `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/e6.csv` |
| Table S54 | tab:binmean-train | `tab_binmean_train.tex` | `paper/tables/make_binmean_tables.py` | `analysis_main_v1/centroid_v1_20261013/results/training/contrasts.csv` |
| Table S55 | tab:binmean-2x2 | `tab_binmean_2x2.tex` | `paper/tables/make_binmean_tables.py` | `analysis_main_v1/centroid_v1_20261013/results/addendum/addendum.json`; `analysis_main_v1/centroid_v1_20261013/results/training/contrasts.csv` |
| Table S56 | tab:binmean-grad | `tab_binmean_grad.tex` | `paper/tables/make_binmean_tables.py` | `analysis_main_v1/centroid_v1_20261013/results/o2_rows_grads/table_exact100.csv` |
| Table S57 | tab:binmean-fwd | `tab_binmean_fwd.tex` | `paper/tables/make_binmean_tables.py` | `analysis_main_v1/centroid_v1_20261013/results/o1_substitution/summary.csv` |
| Figure S1 | fig:seeds | `figS_seed_noise.pdf` | `paper/figures/make_paper_figures.py` | `analysis_main_v1/D_robustness_124m_100m_5seed/analysis_v1/factorial_calibration_backward.csv`; `analysis_main_v1/D_robustness_124m_100m_5seed/analysis_v1/seed_noise.csv` |
| Figure S2 | fig:ksweep | `figS_minmaxw_ksweep.pdf` | `paper/figures/make_paper_figures.py` | `analysis_main_v1/A_longhorizon_124m_2p5b/trajectory/contrast_trajectories.csv` |
| Figure S3 | fig:lowprec | `figS_lowprec.pdf` | `paper/figures/make_paper_figures.py` | `analysis_main_v1/supplement_four_runs_20260923/round3/reports/r3_lowprec_results.json` |
| Figure S4 | fig:geom | `fig5_geometry_main.pdf` | `paper/figures/make_paper_figures.py` | `analysis_main_v1/A_longhorizon_124m_2p5b/geometry/geometry_with_nll_gap.csv` |
| Figure S5 | fig:geomfull | `fig5_geometry_2p5b.pdf` | `paper/figures/make_paper_figures.py` | `analysis_main_v1/A_longhorizon_124m_2p5b/geometry/geometry_with_nll_gap.csv` |
| Figure S6 | fig:onset | `fig8_250m_onset_geometry.pdf` | `paper/figures/make_paper_figures.py` | `analysis_main_v1/B_factorial_124m_250m/geometry/geometry_with_nll_gap.csv`; `analysis_main_v1/B_factorial_124m_250m/trajectory/contrast_trajectories.csv` |
| Figure S7 | fig:widthgap | `fig10_width_vs_gap_5seed.pdf` | `paper/figures/make_paper_figures.py` | `analysis_main_v1/D_robustness_124m_100m_5seed/analysis_v1/endpoint_by_condition.csv`; `analysis_main_v1/D_robustness_124m_100m_5seed/analysis_v1/geometry_association.csv`; `analysis_main_v1/D_robustness_124m_100m_5seed/analysis_v1/geometry_by_condition.csv` |
| Figure S8 | fig:downstream | `fig7_downstream.pdf` | `paper/figures/make_paper_figures.py` | `analysis_main_v1/A_longhorizon_124m_2p5b/endpoint_fullval/endpoint_table.csv`; `analysis_main_v1/downstream_v1/results/layer3/stageC/tables/paired_comparisons.csv` |
| Figure S9 | fig:detachextra | `figS_detach_extra.pdf` | `paper/figures/make_paper_figures.py` | `analysis_main_v1/E_detach_history/dense_logged_gap_detach_vs_full.csv`; `analysis_main_v1/E_detach_history/endpoint_detach_vs_full_s1337.csv`; `training_code/legacy_exact_fp32/runs/q16_nearest_100m/occupancy.csv`; `training_code/legacy_exact_fp32/runs/q4_nearest_100m/occupancy.csv`; `training_code/phaseB_base/runs/q16_nearest_fullgrad_100m/attention_stats.csv`; `training_code/phaseB_base/runs/q4_nearest_fullgrad_100m/attention_stats.csv` |
| Figure S10 | fig:e1 | `figS_e1_channels.pdf` | `paper/figures/make_paper_figures.py` | `analysis_main_v1/D_robustness_124m_100m_5seed/eval_native_v1/LERP-Detach_K32_s42/*/` (10 files: geometry_layers.csv, per_block_nll.npy); `analysis_main_v1/D_robustness_124m_100m_5seed/eval_native_v1/LERP-Detach_K32_s7/*/` (10 files: geometry_layers.csv, per_block_nll.npy); `analysis_main_v1/D_robustness_124m_100m_5seed/eval_native_v1/LERP_K32_s42/*/` (10 files: geometry_layers.csv, per_block_nll.npy); `analysis_main_v1/D_robustness_124m_100m_5seed/eval_native_v1/LERP_K32_s7/*/` (10 files: geometry_layers.csv, per_block_nll.npy); `analysis_main_v1/supplement_four_runs_20260923/eval/rtx3090/sp_s42_lerp_k32_maxonly_100m/*/` (10 files: geometry_layers.csv, per_block_nll.npy); `analysis_main_v1/supplement_four_runs_20260923/eval/rtx3090/sp_s42_lerp_k32_minonly_100m/*/` (10 files: geometry_layers.csv, per_block_nll.npy); `analysis_main_v1/supplement_four_runs_20260923/eval/rtx3090/sp_s7_lerp_k32_maxonly_100m/*/` (10 files: geometry_layers.csv, per_block_nll.npy); `analysis_main_v1/supplement_four_runs_20260923/eval/rtx3090/sp_s7_lerp_k32_minonly_100m/*/` (10 files: geometry_layers.csv, per_block_nll.npy); `analysis_main_v1/supplement_four_runs_20260923/reports/supplement_trajectories.csv` |
| Figure S11 | fig:rb-e11 | `figI_e11_detach_minus_full.pdf` | `paper/figures/make_rb_figures.py` | `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/trajectories.csv` |
| Figure S12 | fig:rb-e2ab | `figI_e2ab_branches.pdf` | `paper/figures/make_rb_figures.py` | `analysis_main_v1/B_factorial_124m_250m/trajectory/contrast_trajectories.csv`; `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/trajectories.csv` |
| Figure S13 | fig:rb-e2c | `figI_e2c_branches.pdf` | `paper/figures/make_rb_figures.py` | `analysis_main_v1/A_longhorizon_124m_2p5b/trajectory/checkpoint_nll.csv`; `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/trajectories.csv` |
| Figure S14 | fig:rb-e3a | `figI_e3a_2p5b.pdf` | `paper/figures/make_rb_figures.py` | `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/probes_layer_mean.csv`; `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/trajectories.csv` |
| Figure S15 | fig:rb-e8 | `figI_e8_seeds.pdf` | `paper/figures/make_rb_figures.py` | `analysis_main_v1/A_longhorizon_124m_2p5b/trajectory/contrast_trajectories.csv`; `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/trajectories.csv` |

Numbers in the text of Appendix I and of the main-text sentences that cite it are LaTeX macros written by
`paper/tables/make_rb_tables.py` to `rb_numbers.tex` from the files of `analysis_main_v1/rb-rebuttal_v5/reports/rb-dossier_20261114/csv/`
and `analysis_main_v1/rb-rebuttal_v5/reports/e2c_step0_check/e2c_step0_check.csv`; each macro carries its source row as a comment.

Tables S54–S57 are inline tabulars in the paper's source; `make_binmean_tables.py` writes each as a file. The numbers
in the text of Appendix J are read from the following files of `analysis_main_v1/centroid_v1_20261013/`: endpoint differences, paired
differences and their intervals from `results/training/contrasts.csv`; trajectories from `results/training/trajectory.csv`;
the softmax NLL and the 100M geometry (row span, entropy) from `results/training/training_results.json`; the backward and
forward effects, the backward share, and the trajectory and geometry of the decomposition run from
`results/addendum/addendum.json`; the verification bounds of Appendix J.1 from `checks/check_centroid_ops.json`
(maxima over the Nearest bins); the layer-0 Q/K gradient norms at initialization from
`checks/check_v2_nearest_centroid_bwd.json` (`V5_step0`, one backward on the first micro-batch). The field
`layer0_qk_grad_l2` of `checks/check_gpu_step0.json` is accumulated over the nine backward passes of that check (the
step-0 pass and the eight micro-steps of its throughput measurement) and is not the single-step value. The bin-mean rows of Tables S4 and S26 are a generated row of `tab_hardware.tex` (GPU
hours from the five training logs) and a hand-written row (5 runs, 25 checkpoints).

## 5. Appendix I: tables and figures to runs

The paper prints no run codes. This table links every table and figure of Appendix I to the runs
behind it (run ids of section 2.2; rows of `csv/comparisons.csv` give each paired difference with its runs).

| Item | Label | Appendix I runs | Runs of the earlier suites | Note |
|---|---|---|---|---|
| Table S28 | tab:rb-overview | rb-C0-01,02,03; rb-C0b-01; rb-E1-01,02,03,04,05,06,07,08; rb-E11-01,02,03,04,05,06,07,08,09; rb-E1b-01; rb-E2ab-01,02,03,04,05,06,07,08,09,10; rb-E2c-01,02,03; rb-E2d-01,02,03,04,05,06,07,08,09,10,11,12; rb-E3a-01,02,03,04,05,06,07,08; rb-E3a-2.5B-01; rb-E3b-01,02,03,04,05,06,07,08; rb-E4-01,02,03,04,05,06,07,08; rb-E5-01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18; rb-E5n-01,02,03,04; rb-E7-01,02; rb-E8a-01; rb-E8c-01 | — |  |
| Table S29 | tab:rb-e1 | rb-C0-01,02,03; rb-C0b-01; rb-E1-01,02,03,04,05,06,07,08 | suite B FWM-Prob K4 s1337 250M; suite B FWM-Weight K4 s1337 250M |  |
| Table S30 | tab:rb-e1b | rb-C0-01,02,03; rb-C0b-01; rb-E1-01,02,03,04,05,06,07,08; rb-E1b-01 | suite A FWM-Prob K4 s1337 2.5B; suite A FWM-Weight K4 s1337 2.5B; suite A LERP K4 s1337 2.5B; suite A softmax s1337 2.5B; suite B FWM-Prob K4 s1337 250M; suite B FWM-Weight K4 s1337 250M |  |
| Table S31 | tab:rb-e1bdown | rb-E1b-01 | suite A softmax s1337 (2.5B) |  |
| Table S32 | tab:rb-e11 | rb-C0-01,02; rb-E1-01,02,04,05,06; rb-E11-01,02,03,04,05,06,07,08,09 | App. C.2 run; App. F maxonly s42; App. F maxonly s7; App. F minonly s42; App. F minonly s7; suite D LERP K32 s42; suite D LERP K32 s7 |  |
| Figure S11 | fig:rb-e11 | rb-C0-01,02; rb-E1-01,02,04,05,06; rb-E11-01,02,03,04,05,06,07,08,09 | App. C.2 run; App. F maxonly s42; App. F maxonly s7; App. F minonly s42; App. F minonly s7; suite D LERP K32 s42; suite D LERP K32 s7 |  |
| Table S33 | tab:rb-e2ab | rb-C0b-01; rb-E2ab-01,02,03,04,05,06,07,08,09,10 | — |  |
| Table S34 | tab:rb-e2chain | rb-C0b-01; rb-E2ab-01,02,03,04,05,06,07,08,09,10 | — |  |
| Figure S12 | fig:rb-e2ab | rb-C0b-01; rb-E2ab-01,02,03,04,05,06,07,08,09,10 | — |  |
| Table S35 | tab:rb-e2d | rb-E2d-01,02,03,04,05,06,07,08,09,10,11,12 | — |  |
| Table S36 | tab:rb-e2c | rb-E2c-01,02,03 | suite A softmax s1337 at 625M |  |
| Figure S13 | fig:rb-e2c | rb-E2c-01,02,03 | suite A softmax s1337 at 625M |  |
| Table S37 | tab:rb-e4 | rb-C0-01,02,03; rb-C0b-01; rb-E4-01,02,03,04,05,06,07,08 | — |  |
| Table S38 | tab:rb-e4probe | rb-C0-01,02,03; rb-C0b-01; rb-E4-01,02,03,04,05,06,07,08 | — |  |
| Table S39 | tab:rb-formats | rb-E5-01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18; rb-E5n-01,02,03,04 | — | format rules and level counts: `analysis_main_v1/rb-rebuttal_v5/code/checks/check_formats.py` and its records `checks/13_3_formats_*.json` |
| Table S40 | tab:rb-e5 | rb-C0-01,02,03; rb-C0b-01; rb-E5-01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18; rb-E5n-01,02,03,04 | D.5 BF16-exp s7; D.5 FP8-E4M3 exp s7; suite D softmax s7 |  |
| Table S41 | tab:rb-e3b | rb-C0-01,02,03; rb-C0b-01; rb-E3a-01,02,03,04; rb-E3b-01,02,03,04,05,06,07,08 | — |  |
| Table S42 | tab:rb-e3bprobe | rb-C0-01,02,03; rb-C0b-01; rb-E3a-01,02,03,04; rb-E3b-01,02,03,04,05,06,07,08 | — |  |
| Table S43 | tab:rb-e3a | rb-C0-01,02,03; rb-C0b-01; rb-E3a-01,02,03,04,05,06,07,08 | — |  |
| Table S44 | tab:rb-e3a25 | rb-E3a-2.5B-01 | suite A FWM-Prob K4 s1337 2.5B; suite A FWM-Weight K16 s1337 2.5B; suite A LERP K4 s1337 2.5B; suite A softmax s1337 2.5B |  |
| Table S45 | tab:rb-e3adown | rb-E3a-2.5B-01 | suite A softmax s1337 (2.5B) |  |
| Figure S14 | fig:rb-e3a | rb-E3a-2.5B-01 | suite A FWM-Prob K4 s1337 2.5B; suite A FWM-Weight K16 s1337 2.5B; suite A LERP K4 s1337 2.5B; suite A softmax s1337 2.5B |  |
| Table S46 | tab:rb-e0b | rb-E1-01,02,03,04,05,06,07,08; rb-E11-01,02,03,04,05,06,07,08,09; rb-E2ab-01,02,03,04,05,06,07,08,09,10; rb-E2c-01,02,03; rb-E3a-01,02,03,04,05,06,07,08; rb-E3a-2.5B-01; rb-E3b-01,02,03,04,05,06,07,08; rb-E4-01,02,03,04,05,06,07,08; rb-E5-01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18; rb-E5n-01,02,03,04 | — |  |
| Table S47 | tab:rb-e0blen | rb-E1-01,02,03,04,05,06,07,08; rb-E11-01,02,03,04,05,06,07,08,09; rb-E2ab-01,02,03,04,05,06,07,08,09,10; rb-E2c-01,02,03; rb-E3a-01,02,03,04,05,06,07,08; rb-E3a-2.5B-01; rb-E3b-01,02,03,04,05,06,07,08; rb-E4-01,02,03,04,05,06,07,08; rb-E5-01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18; rb-E5n-01,02,03,04 | — |  |
| Table S48 | tab:rb-e8 | rb-E8a-01; rb-E8c-01 | suite A LERP K16 s1337; suite A softmax s1337; suite A softmax s1337 2.5B |  |
| Figure S15 | fig:rb-e8 | rb-E8a-01; rb-E8c-01 | suite A LERP K16 s1337; suite A softmax s1337; suite A softmax s1337 2.5B |  |
| Table S49 | tab:rb-e0a | — | — | no run: operation counts read from the code |
| Table S50 | tab:rb-e0d | — | — | no run: roofline constants (`e0d()` in `code/eval/rb_dossier.py`) |
| Table S51 | tab:rb-e0c | — | — | no run: levels and bit widths (`e0c_bitwidth.csv`, from the format check records) |
| Table S52 | tab:rb-e0aacc | rb-E1b-01; rb-E3a-2.5B-01 | suite A runs (seed 1337, 2.5B) |  |
| Table S53 | tab:rb-e6 | rb-E11-06; rb-E7-01,02 | App. C.2 strict MinMax-Weight K4 s7 | E6: 50 evaluations of six endpoints, four of suite A at 2.5B and the FWM–Weight and MinMax–Weight K = 4 runs of suite D, seed 7, at 100M (`runs/_eval/e6/`, RTX 3090 repeat in `runs/_eval/e6_rtx3090/`); E7: the TF32-off runs |

## 6. Commands

### 6.1 Evaluation (needs the checkpoints, which are not included)
```bash
# suites A-C: one job per checkpoint, the run's own frozen code root, batch 1, full validation split
python analysis_main_v1/code/eval_worker.py --worker main
# suite D, detach replication, tail policy (resumable)
python analysis_main_v1/code/eval_worker_5seed.py
# Appendix I: validation NLL, probes, held-out corpora and benchmarks
python analysis_main_v1/rb-rebuttal_v5/code/eval/rb_eval.py --help
python analysis_main_v1/rb-rebuttal_v5/code/eval/rb_probe.py --help
HF_HUB_OFFLINE=1 python analysis_main_v1/rb-rebuttal_v5/code/eval/rb_downstream.py --help
# downstream battery of the 2.5B suite (datasets cached first; then offline)
HF_HUB_OFFLINE=1 python analysis_main_v1/downstream_v1/code/layer3_stageC.py --all
```
Every evaluation runs a checkpoint through the code root it was trained with, with TF32 disabled, at batch size 1
on the full validation split; the downstream harness (`lm_eval==0.4.13`) runs offline (`HF_HUB_OFFLINE=1`) after the
datasets are cached.

### 6.2 Analysis (from the stored result files; CPU)
```bash
cd analysis_main_v1
python code/analyze_primary.py --suites B_factorial_124m_250m,A_longhorizon_124m_2p5b   # suites B and A
python code/analyze_primary.py --suites C_scale_1b_100m                                 # suite C
python code/analyze_5seed.py                                                            # suite D
python code/analyze_detach_history.py                                                   # Appendix F record
python supplement_four_runs_20260923/code/analyze_supplement.py                         # Appendices C.2 and F
python supplement_four_runs_20260923/code/analyze_e1_seed.py                            # Appendix F, seed 42
python supplement_four_runs_20260923/code/analyze_r3_lowprec.py                         # Appendix D.5
python tail_policy/tail_decomp/analyze_tail_decomp.py                                   # Appendix D.4
cd rb-rebuttal_v5/code/eval
python rb_dossier.py --out rb_dossier_regen   # Appendix I result tables (csv) from the evaluation results
python rb_dossier_check.py                   # re-derives every number of csv/ independently and compares
cd ../../../centroid_v1_20261013/code/analysis
python analyze_training.py                   # Appendix J.2: results/training/
python analyze_addendum.py                   # Appendix J.3: results/addendum/
python analyze_o1.py                         # Appendix J.5: results/o1_substitution/summary.*
python analyze_o2.py                         # Appendix J.4: results/o2_rows_grads/summary.json, table_exact100.csv
```
`analyze_primary.py` draws its bootstrap resamples from one random stream per call, so the suites are analysed in the
two calls above. With the versions of `requirements.txt` every command rewrites the result files of this repository
identically, and `rb_dossier.py --out DIR` writes the 20 files of `reports/rb-dossier_20261114/csv/` again under
`DIR/csv/`. With another NumPy version or on another processor, single cells can differ in the last floating-point
digits; `rb_dossier_check.py` accepts differences up to 10⁻¹² × max(1, |value|).

### 6.3 Tables, figures and numbers of the paper (CPU, a few minutes)
```bash
(cd paper/tables  && python make_paper_tables.py && python make_rb_tables.py && python make_binmean_tables.py)   # -> paper/tables/*.tex, rb_numbers.tex
(cd paper/figures && python make_paper_figures.py && python make_rb_figures.py)  # -> paper/figures/*.pdf, *.png
python paper/check_tables.py                                                     # compares paper/tables/ with paper/reference/tables/
```
`bash paper/make_all.sh` runs these commands. `paper/reference/tables/` holds the table files of the paper's source: the
50 files that it includes and the tabulars of Tables S54–S57, which it holds inline. Run with the versions of
`requirements.txt`, the five scripts write these 54 files and the 20 generated figures of the paper (PDF and PNG). The
table files are byte-identical to those of the paper's source; `rb_numbers.tex` is identical apart from its TeX
comments (the generator writes the source row of each number as a comment; the paper's source carries no comments).
The page content of the figure PDFs is identical to that of the paper; the compressed bytes of the figure files depend
on the zlib library of the Python installation.

### 6.4 Checks, by position in the paper
| Paper | Script | Records |
|---|---|---|
| Appendix A, edge cases (Table S1) | `python paper/check_degenerate_rows.py` (reads the attention code of every training root) | printed |
| Appendix B, verification | `python paper/check_minmax_hard_jacobian.py`; `python analysis_main_v1/report/verify_gradient_formulas.py` (closed forms vs autograd of the training code, float64) | printed |
| Appendix E.1, pre-declared condition groups and predictions | — | `analysis_main_v1/downstream_v1/REPORT.md`, section 1 |
| Appendix I.1, "Stack and code" (CPU equivalence of the merged root) | `analysis_main_v1/rb-rebuttal_v5/code/checks/check_root_equivalence.py` | `checks/13_3_root_equivalence_*.json` |
| Appendix I.7, agreement with the IntAttention release | `code/e3_ref/test_align.py`, `test_align_gpu.py`, `make_e3_reference.py` (each takes the path to IntAttention's `ArtifactEvaluation/pysimulation` as first argument; `make_e3_reference.py` takes the output file as second argument), `code/e3_ref/compare_e3_reference.py checks/e3_reference_tensors.pt` (compares with the saved reference outputs, after `gunzip -k checks/e3_reference_tensors.pt.gz`, section 7; loads no IntAttention code), `code/checks/check_external_ops.py PATH` or `--no-thirdparty` | `checks/13_3_e3_reference_*.json`, `checks/13_3_external_ops_*.json` |
| Appendix I.12, tests of the training mechanics and of the operators | `code/checks/run_all_checks.sh PYTHON [PATH|--no-thirdparty]` runs `smoke_all_modes.py`, `check_root_equivalence.py`, `check_fwm_ops.py`, `check_formats.py`, `check_external_ops.py`, `check_ema.py`, `check_train_mechanics.py` (shared settings `rb_common.py`, single-threaded CPU) | stack S3: `checks/13_3_*_torch2.5.1*.json` (also `torch2.14.1`); RTX 5090 machines: `checks/13_3_*_machine-A.json`, `checks/13_3_*_machine-B.json` |
| Appendix I.12, torch.clamp boundary gradient | `code/tools/clamp_boundary_check.py`; run before every run of that experiment by `code/tools/rb_queue.py` | `checks/13_2_clamp_machine-{A,B}.json`, `checks/13_2_env_machine-{A,B}.json` |
| Appendix J.1, verification of the bin-mean operators | `analysis_main_v1/centroid_v1_20261013/code/checks/check_centroid_ops.py` (CPU, float64; rewrites its record, with the local path in the field `new_root`) | `analysis_main_v1/centroid_v1_20261013/checks/check_centroid_ops.json` |
| Appendix J.3, forward equal to the strict FWM–Prob root, backward equal to the vector–Jacobian product; gradient norms at initialization | `analysis_main_v1/centroid_v1_20261013/code/checks/check_v2_nearest_centroid_bwd.py` (CUDA GPU and the tokenized WikiText-103 in `training_code/phaseB_base/data/wikitext103/`) | `analysis_main_v1/centroid_v1_20261013/checks/check_v2_nearest_centroid_bwd.json` |
| Appendix J, GPU checks before training | `analysis_main_v1/centroid_v1_20261013/code/checks/check_gpu_step0.py` (CUDA GPU and the tokenized data) | `analysis_main_v1/centroid_v1_20261013/checks/check_gpu_step0.json` |

Paths of the four Appendix I rows are relative to `analysis_main_v1/rb-rebuttal_v5/`. The real-score cases of
`check_external_ops.py` (run when a CUDA GPU is present) and of `make_e3_reference.py` read two checkpoints and the
WikiText-103 validation split, which are not included (set `RB_CKPT_SOFTMAX_S7_100M`, `RB_CKPT_FWM_WEIGHT_K4_S7_100M` and
`RB_DATA_WIKITEXT103`); without them these two scripts, and `run_all_checks.sh` on a machine with a GPU, stop with an
error. Bitwise agreement depends on the stack (Appendix I.7): with another PyTorch version the EXAQ outputs on the GPU can
differ from the reference in the last bits, and `compare_e3_reference.py` then exits with status 1.
`smoke_all_modes.py`, `code/e3_ref/test_align_gpu.py`, `make_e3_reference.py` and `compare_e3_reference.py` need a CUDA
GPU; the other checks run on the CPU. On a machine without a GPU,
`RB_SKIP_CHECKS=smoke_all_modes.py code/checks/run_all_checks.sh PYTHON --no-thirdparty` runs the six CPU checks. With
PyTorch 2.14.1 or 2.15.0 the six detached-row-maximum cases of `check_fwm_ops.py` fail, for the reason given in Appendix
I.12; they pass with PyTorch 2.13.0.

## 7. IndexSoftmax and EXAQ (Appendix I.7)

IndexSoftmax: our implementation follows the PyTorch implementation released with IntAttention (Zhong et al., 2026).
EXAQ: the original EXAQ source code is no longer publicly available; our implementation follows the EXAQ paper
(Shkolnik et al., 2024) and the reimplementation released in the IntAttention artifact repository.

The operators as trained are functions of `analysis_main_v1/rb-rebuttal_v5/code/rb-root_v1/quant_attention.py`
(`_index_softmax_hard`, `_exaq_window`, `_exaq_hard`, `_exaq_surrogate`, `external_operator_probs`); the reference
forward used by the tests is `analysis_main_v1/rb-rebuttal_v5/code/e3_ref/ref_ops.py`. No file or function of the
IntAttention release is included: the agreement tests load it from a path given on the command line.

`analysis_main_v1/rb-rebuttal_v5/checks/e3_reference_tensors.pt.gz` is the gzip-compressed file `e3_reference_tensors.pt`
(compressed because of its size; `gunzip -k analysis_main_v1/rb-rebuttal_v5/checks/e3_reference_tensors.pt.gz` restores
it, 130,975,491 bytes, SHA-256 `eabfd8212388f4e46def4731df1a7864f8dde8bd4f25ec7304bbbaa607833fe3`). It holds reference outputs of the IntAttention simulation
against which the RTX 5090 machines were compared ("within 2.0×10⁻⁶ of the reference outputs computed on S3"). The
complete file was written by `code/e3_ref/make_e3_reference.py` on stack S3 (local RTX 3090, PyTorch 2.5.1+cu121); it is
read by `code/e3_ref/compare_e3_reference.py`, which compares every case the file contains. The file is loaded with
`torch.load(..., weights_only=False)`: its `meta` block stores the PyTorch version as a `TorchVersion` object, which the
default loader of PyTorch 2.6 and later does not accept.

The file in this repository is a subset of whole cases of the complete file, kept for size. Its tensors are bit-for-bit
those of the complete file; no reference output was recomputed. Kept (16 cases): `random_n64_s0.3`, `random_n64_s3.0`, `random_n64_s8.0`, `random_n256_s0.3`, `random_n256_s3.0`, `random_n256_s8.0`, `random_n1024_s0.3`, `random_n1024_s3.0`, `random_n1024_s8.0`, `bin_boundaries`, `int_nbits6_n64`, `int_nbits6_n512`, `int_nbits10_n64`, `int_nbits10_n512`, `int_nbits12_n64`, `int_nbits12_n512`. Omitted
(4 cases): `real_softmax_s7_100m_L0_h0-3`, `real_softmax_s7_100m_L11_h0-3`, `real_fwm_weight_k4_s7_100m_L0_h0-3`, `real_fwm_weight_k4_s7_100m_L11_h0-3`. Each random case, the `bin_boundaries` case and each real case holds the scores, the
IndexSoftmax (float) and EXAQ 3-bit outputs on CPU and their GPU−CPU differences; each integer case holds the integer scores, scale, threshold and the
UINT8 codes of IndexSoftmax. The SHA-256 of the complete file is
`9c33874778d236ed9e86714b128460ce109a1fc4564c4e98e1391b6af192caae`; the file stores it (`meta.full_file_sha256`) together
with the list of omitted cases (`meta.omitted_cases`). The other fields of `meta` are unchanged: `source` (the IntAttention
simulation, loaded read-only and not copied), `torch` ("2.5.1+cu121"), `gpu` ("NVIDIA GeForce RTX 3090"). The comparisons
of the two RTX 5090 machines with the complete file, all 20 cases, are recorded in
`analysis_main_v1/rb-rebuttal_v5/checks/13_3_e3_reference_machine-A.json` and
`analysis_main_v1/rb-rebuttal_v5/checks/13_3_e3_reference_machine-B.json`.

## 8. Environment

`requirements.txt` pins PyTorch 2.13.0 (the stack of the Appendix I runs, stack S1 of Table S4). `env/pip_freeze_rb-s1.txt`
and `env/pip_freeze_rb-ds.txt` are the complete package lists of the training and the downstream environment on the
RTX 5090 machines. The earlier suites used the stacks of Table S4 recorded in the manifests and `summary.json` files;
the runs, evaluations and checks of Appendix J used stack S3 (RTX 3090, PyTorch 2.5.1+cu121).


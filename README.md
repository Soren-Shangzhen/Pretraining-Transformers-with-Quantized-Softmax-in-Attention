# Pretraining Transformers with Quantized Softmax in Attention

Code, run manifests and evaluation results for the paper *Pretraining Transformers with Quantized Softmax in Attention*
(Shangzhen Zhu, Muyan Hu, Tomasz Kozlowski). The paper studies K-interval attention, which approximates the exponential
of softmax attention with K+1 grid values, during pretraining: per-row grid calibration (MinMax or a fixed window at the
row maximum, FWM), interpolation (LERP) versus hard rounding (Nearest), and the placement of the straight-through
surrogate relative to normalization (Weight-STE, Prob-STE).

This repository holds the training code of every run the paper reports (the 202 runs of Table S26 and the 98 runs of
Table S28), the evaluations and checks whose results it prints, the stored per-checkpoint evaluation results, and the
scripts that regenerate every generated table, figure and number of the paper from those results.
`REPRODUCTION_INDEX.md` maps each run to its manifest, code and result files, each table and figure to its source
files, and gives the commands. Appendix letters and table and figure numbers are those of the arXiv version of the
paper.

## Repository layout

| Directory | Content |
|---|---|
| `training_code/` | frozen training code roots of suites A–D and of the runs of Appendices C.2, D.4, D.5 and F, and the per-run training logs read by the table and figure scripts |
| `analysis_main_v1/manifests/`, `analysis_main_v1/D_robustness_124m_100m_5seed/manifests/` | run and checkpoint manifests of suites A–C and D |
| `analysis_main_v1/eval_native_v1/`, `analysis_main_v1/D_robustness_124m_100m_5seed/eval_native_v1/` | per checkpoint: per-block validation NLL (`per_block_nll.npy`), `summary.json` (configuration, code SHA-256), `geometry_layers.csv` |
| `analysis_main_v1/code/` | evaluation and analysis code of suites A–D and of the detach record |
| `analysis_main_v1/E_detach_history/` | historical detach intervention (Appendix F): run list and evaluation results |
| `analysis_main_v1/supplement_four_runs_20260923/` | single-channel, strict-forward and reduced-precision runs, frozen-endpoint audit, gradient decomposition (Appendices C.2, D.5, F): code roots, manifests, evaluation results |
| `analysis_main_v1/rounding_audit/`, `detach_replication/`, `tail_policy/` | rounding audit (Appendix C.2), detach replication contrasts (Appendix F), tail-policy runs (Appendix D.4) |
| `analysis_main_v1/provenance/tau_selection/` | selection of τ (Appendix C.3) |
| `analysis_main_v1/downstream_v1/` | downstream evaluation of the 2.5B suite (Appendices C.3, E): pre-declaration (`REPORT.md`, section 1), code, probe definitions, results (summary tables; per-item outputs only for the six conditions that Appendix I compares with) |
| `analysis_main_v1/training_logs/` | training logs of suites A–C read by the table scripts |
| `analysis_main_v1/rb-rebuttal_v5/` | Appendix I: training code root, run manifest, test scripts and their records, evaluation results (`runs/_eval/`), result tables (`reports/rb-dossier_20261114/csv/`), the step-0 check |
| `analysis_main_v1/centroid_v1_20261013/` | Appendix J (bin-mean control): training code roots, protocol and pre-training records, operator checks, evaluation results, analysis code and results |
| `paper/` | the generators of the paper's tables, figures and in-text numbers, the table files of the paper's source for comparison (`reference/tables/`), and the checks of Appendices A and B |
| `env/` | package lists of the training and downstream environments of Appendix I |

## Where the operators are implemented

The attention operators are in `quant_attention.py` of each training code root; section 1 of `REPRODUCTION_INDEX.md`
lists the roots, their operators and the SHA-256 of each file. In
`analysis_main_v1/rb-rebuttal_v5/code/rb-root_v1/quant_attention.py`, the root of the Appendix I runs: MinMax calibration
is `_adaptive_interval_reconstruct`, FWM is `_relevant_zero_reconstruct`, the surrogates and the calibration-gradient
options are in `QuantCausalSelfAttention._quant_attention`; the operators added for Appendix I are `ema_window_probs`
(I.4), `_mxfp8_quantize`, `_nvfp4_quantize`, `_uint8_quantize` and `softmax_family_probs` (I.5), `_exaq_window`,
`_exaq_hard`, `_exaq_surrogate` (I.6) and `_index_softmax_hard` (I.7). The bin-mean reconstruction of Appendix J
("Centroid" in the files) is `_centroid_cells` and `_centroid_reconstruct` in
`analysis_main_v1/centroid_v1_20261013/code/centroid_root/quant_attention.py`.

## Installation

```bash
pip install -r requirements.txt
```

`requirements.txt` pins PyTorch 2.13.0 with Python 3.12, the stack of the Appendix I runs. Run every command from the
repository root unless it starts with `cd`.

## Quick start (CPU; no checkpoints needed)

```bash
bash paper/make_all.sh                                                            # all tables, figures and in-text numbers; compares the tables with the paper's
python paper/check_degenerate_rows.py                                             # Appendix A, edge cases
python paper/check_minmax_hard_jacobian.py                                        # Appendix B, verification
python analysis_main_v1/report/verify_gradient_formulas.py                        # Appendix B, verification
(cd analysis_main_v1/rb-rebuttal_v5/code/eval && python rb_dossier_check.py)      # Appendix I numbers from the evaluation results
(cd analysis_main_v1/centroid_v1_20261013/code/analysis && python analyze_training.py && python analyze_addendum.py)   # Appendix J contrasts
```

`paper/make_all.sh` runs the five generators of `paper/tables/` and `paper/figures/` and ends with
`54/54 table files identical to the paper's source`: the 50 table files that the paper's source includes and the four
tables of Appendix J (`paper/reference/tables/`; `rb_numbers.tex` holds the numbers printed in the text of Appendix I).
With the versions of `requirements.txt` the 20 generated figures have the same page content as those of the paper; the
compressed bytes of the figure files depend on the zlib library of the Python installation.

## Reproducing the training runs

The tokenized data and the model checkpoints are not included. WikiText-103 and FineWebEdu are public;
`training_code/phaseB_base/prepare_data.py` tokenizes them with the GPT-2 BPE:

```bash
python training_code/phaseB_base/prepare_data.py --dataset wikitext103 --out_dir data/wikitext103
python training_code/phaseB_base/prepare_data.py --dataset fineweb-edu --num_tokens 3000000000 --out_dir data/finewebedu_3b
```

Each run is retrained from its manifest row with the code root listed for it in `REPRODUCTION_INDEX.md` (section 2):

- Suites A–C: `train_formal.py` of the root in `training_code/formal_code_factorial250_v1/` (or `formal_lerp_k1_v1/`)
  with the configuration of the run's manifest row (operator recorded again in each `summary.json`, field
  `checkpoint_operator`) and the schedules of Appendix C.1.
- Suite D and the runs of Appendices D.4, D.5 and F: `train.py` / `train_formal.py` of the listed root, for example
  `cd training_code/phaseB_base && python train.py --run_name lerp_k4_s7 --data_dir ../../data/wikitext103 --out_dir runs --attn quant --approx lerp --K 4 --range_grad full --max_tokens 100000000 --warmup_tokens 10000000 --batch_tokens 131072 --micro_batch 1 --lr 6e-4 --seed 7`.
- Appendix I: the field `command` of `analysis_main_v1/rb-rebuttal_v5/manifests/rb-runs_v5_manifest.json` is the exact
  command of each run. Its variables: `RB_CODE` = `analysis_main_v1/rb-rebuttal_v5/code`, `RB_PY` = the Python
  interpreter, `RB_DATA` = the tokenized data, `RB_RUNS` = the output directory (`analysis_main_v1/rb-rebuttal_v5/runs/` has its layout, without the checkpoints); `RB_PARENTS`, `RB_CLOUD` and
  `RB_LOCAL_ENDPOINTS` = the checkpoints of the earlier suites that branch runs start from. rb-E3a-2.5B-01 was launched
  with `{E3A_2P5B_VARIANT}` = `u8`.
- Appendix J: the commands of the five runs are in `analysis_main_v1/centroid_v1_20261013/code/jobs/run_training_queue.sh`
  and `run_addendum_nfcb.sh` (`train.py` of `code/centroid_root/` with `--approx centroid`, and of `code/centroid_root_v2/`
  with `--approx nearest_centroid_bwd`).

The evaluation, analysis and check commands are in section 6 of `REPRODUCTION_INDEX.md`. Every evaluation runs a
checkpoint through the code root it was trained with, with TF32 disabled, at batch size 1 on the full validation split.

## Not included

- **Model checkpoints** (about 1.3 TB for suites A–D, and the checkpoints of Appendices I and J) and per-run training
  directories; manifests refer to them as `<CLOUD_DISK>/…`, `<REPO>/training_code/…` or `${RB_RUNS}`. Every
  evaluation result records the SHA-256 of the checkpoint it read (Appendix I) or its path and step.
- **Tokenized data** (see above).
- **Per-item outputs of the downstream evaluation of the 2.5B suite**, because of their size. The summary tables are
  included (`analysis_main_v1/downstream_v1/results/layer3/stageC/tables/`); the commands that regenerate the per-item
  outputs are in section 6.1 of `REPRODUCTION_INDEX.md`. Included are only the per-item files of the six conditions that
  the Appendix I comparisons read, and the per-item outputs of the Appendix I runs. In these per-item files only the
  columns that the statistics read are kept; the other columns are omitted because of their size. Kept: multiple-choice benchmarks (ARC-Easy, BLiMP, HellaSwag, PIQA, SciQ with and without the support passage): `subtask`, `doc_id`, `doc_hash`, `choice_idx`, `gold_idx`, `acc_correct`, `accnorm_correct`, `margin_token_norm`; LAMBADA: `subtask`, `doc_id`, `doc_hash`, `n_scored_tokens`, `target_nll`, `token_norm_target_nll`, `greedy_correct`.
  Rows, their order and the kept fields are unchanged (`REPRODUCTION_INDEX.md`, section 3).
- **Third-party code.** No file or function of the IntAttention release and no EXAQ source code is included. The
  agreement tests of Appendix I.7 load the IntAttention code from a path given on the command line.

`analysis_main_v1/rb-rebuttal_v5/checks/e3_reference_tensors.pt.gz` (reference outputs for the agreement tests of
Appendix I.7) is stored gzip-compressed because of its size; unpack it with `gunzip -k` before running
`compare_e3_reference.py` (`REPRODUCTION_INDEX.md`, section 7).

Paths of the machines that ran the experiments are replaced by placeholders (`<REPO>`, `<CLOUD_DISK>`, …) and machine
names by neutral labels; `REPRODUCTION_INDEX.md` lists them. Result files, run names, run directories and code fields
keep their original identifiers; section 1 of the index maps them to the names used in the paper (for example `QRM`,
`FW` and `range_mode=relevant_zero` denote FWM, and `Centroid` denotes the bin-mean reconstruction).

## IndexSoftmax and EXAQ

IndexSoftmax: our implementation follows the PyTorch implementation released with IntAttention (Zhong et al., 2026).
EXAQ: the original EXAQ source code is no longer publicly available; our implementation follows the EXAQ paper
(Shkolnik et al., 2024) and the reimplementation released in the IntAttention artifact repository.

## Environment

`requirements.txt` pins PyTorch 2.13.0, the stack of the Appendix I runs. Training enables TF32 matrix multiplications
(except the TF32-off runs of Appendix I.11); evaluation disables them; the attention operator runs in fp32 and the rest
of the network under bf16 autocast. The stacks of the earlier suites are given in Table S4 and recorded in the
manifests and `summary.json` files.

## Citation

```bibtex
@article{zhu2026pretraining,
  title   = {Pretraining Transformers with Quantized Softmax in Attention},
  author  = {Zhu, Shangzhen and Hu, Muyan and Kozlowski, Tomasz},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

MIT (`LICENSE`). The IntAttention release is not redistributed here.

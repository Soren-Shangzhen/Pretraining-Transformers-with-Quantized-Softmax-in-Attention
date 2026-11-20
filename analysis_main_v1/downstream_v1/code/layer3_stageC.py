#!/usr/bin/env python3
"""Stage C: seven evaluation configurations x 15 conditions, native forward only.

One process evaluates one condition over all seven configurations. Every scored request is written to disk
with enough detail to rebuild the standard task metrics offline (per-candidate raw total log-probability,
the number of tokens actually scored, the continuation character count, hashes of the original document and
of the exact scored input, the gold index, the harness's own acc / acc_norm decisions, and the truncation /
failure / retry status).

Nothing here compares conditions; §8 statistics are computed by layer3_stageC_stats.py from these files.

Resume: a configuration whose <task>.done file matches the current fingerprint (checkpoint sha256, code root,
harness version, task config hash, sample-order hash, scorer version) is reused; otherwise it is recomputed.
Writes are atomic (temp file + os.replace).

Usage:
  layer3_stageC.py --manifest                       # build the manifest only (no GPU)
  layer3_stageC.py --preflight                      # small end-to-end checks
  layer3_stageC.py --condition "LERP K4"            # one condition, all seven configurations
  layer3_stageC.py --all                            # every condition in manifest order
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict

import numpy as np
import tiktoken
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from common_loader import BLOCK, ROOT, load, runs, set_numerics, sha256  # noqa: E402
from common_loader import operator as loaded_op  # noqa: E402

import lm_eval  # noqa: E402
from lm_eval import simple_evaluate  # noqa: E402
from lm_eval.api.model import LM  # noqa: E402
from lm_eval.api.registry import register_model  # noqa: E402
from lm_eval.tasks import TaskManager  # noqa: E402

OUT = f"{BASE}/results/layer3/stageC"
TASKDIR = f"{HERE}/tasks"
SCORER_VERSION = "stageC-1.0"

# The seven evaluation configurations. BLiMP keeps limit=200 per paradigm; everything else is the full split.
CONFIGS = [
    dict(cfg="blimp", task="blimp", limit=200, kind="mc", split="validation"),
    dict(cfg="hellaswag", task="hellaswag", limit=None, kind="mc", split="validation"),
    dict(cfg="piqa", task="piqa", limit=None, kind="mc", split="validation"),
    dict(cfg="arc_easy", task="arc_easy", limit=None, kind="mc", split="test"),
    dict(cfg="lambada_openai", task="lambada_openai", limit=None, kind="lambada", split="test"),
    dict(cfg="sciq_default", task="sciq", limit=None, kind="mc", split="test"),
    dict(cfg="sciq_nosupport", task="sciq_nosupport", limit=None, kind="mc", split="test"),
]
SEEDS = dict(random_seed=20260920, numpy_random_seed=20260920, torch_random_seed=20260920,
             fewshot_random_seed=20260920)
MAX_RETRY = 2

MC_FIELDS = ["condition", "cfg", "task", "subtask", "split", "doc_id", "doc_hash", "n_choices", "choice_idx",
             "gold_idx", "is_gold", "input_hash", "cont_tok_hash", "cont_chars", "choice_chars",
             "target_delimiter_chars", "n_scored_tokens",
             "raw_total_logprob", "token_norm_logprob", "truncated", "status", "retries",
             "pred_acc_idx", "acc_correct", "pred_accnorm_idx", "accnorm_correct",
             "harness_acc", "harness_acc_norm", "margin_token_norm"]
LAM_FIELDS = ["condition", "cfg", "task", "subtask", "split", "doc_id", "doc_hash", "input_hash",
              "cont_tok_hash", "target_chars", "n_scored_tokens", "raw_total_logprob", "target_nll",
              "token_norm_target_nll", "greedy_correct", "harness_acc", "truncated", "status", "retries"]


def h(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]


def slug(cond):
    return cond.replace(" ", "_").replace("-", "")


def write_task_yaml():
    """SciQ without the support passage: identical template, question, choices and scoring; support removed."""
    os.makedirs(TASKDIR, exist_ok=True)
    p = f"{TASKDIR}/sciq_nosupport.yaml"
    body = """task: sciq_nosupport
dataset_path: allenai/sciq
output_type: multiple_choice
training_split: train
validation_split: validation
test_split: test
doc_to_text: "Question: {{question}}\\nAnswer:"
doc_to_target: 3
doc_to_choice: "{{[distractor1, distractor2, distractor3, correct_answer]}}"
should_decontaminate: true
doc_to_decontamination_query: "{{question}}"
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
  - metric: acc_norm
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
"""
    if not os.path.exists(p) or open(p).read() != body:
        with open(p, "w") as f:
            f.write(body)
    return p


@register_model("frozenop_stagec")
class StageCLM(LM):
    """lm-eval adapter: one loaded run, its own training operator, batch 1, eval mode, dropout off."""

    def __init__(self, run=None, forward_mode="native", **kwargs):
        super().__init__()
        self.enc = tiktoken.get_encoding("gpt2")
        self.loaded = load(run, forward_mode)
        self.model = self.loaded.model
        assert not self.model.training, "model must be in eval mode"
        for m in self.model.modules():
            if isinstance(m, torch.nn.Dropout):
                assert m.p == 0.0 or not m.training, "dropout must be inactive"
        self.reset()

    def reset(self):
        """Clear per-configuration accounting; the loaded weights and operator are reused."""
        self.records = {}
        self.order = []
        self.truncated = 0
        self.n_requests = 0
        self.failures = 0
        self.retries = 0

    @property
    def eot_token_id(self):
        return self.enc.eot_token

    @property
    def max_length(self):
        return BLOCK

    @property
    def max_gen_toks(self):
        return 64

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        return "cuda"

    def tok_encode(self, string):
        return self.enc.encode_ordinary(string)

    def tok_decode(self, tokens):
        return self.enc.decode(tokens)

    @torch.inference_mode()
    def _score(self, context, continuation):
        ctx = self.tok_encode(context) if context else [self.eot_token_id]
        cont = self.tok_encode(continuation)
        ids = ctx + cont
        truncated = len(ids) > BLOCK + 1
        if truncated:
            ids = ids[-(BLOCK + 1):]
        n_cont = min(len(cont), len(ids) - 1)
        x = torch.tensor(ids[:-1], dtype=torch.long, device="cuda").unsqueeze(0)
        y = torch.tensor(ids[1:], dtype=torch.long, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = self.model(x)
        lg = logits.float()[0]
        lp = torch.log_softmax(lg, dim=-1)
        tgt = y[-n_cont:]
        tok_lp = lp[-n_cont:].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        greedy = bool((lg[-n_cont:].argmax(-1) == tgt).all().item())
        scored = ids[-n_cont:]
        return dict(raw_total_logprob=float(tok_lp.sum().item()), n_scored_tokens=int(n_cont),
                    greedy=greedy, truncated=bool(truncated), cont_chars=len(continuation),
                    input_hash=h(context + "\x00" + continuation),
                    cont_tok_hash=h(",".join(map(str, scored))))

    def loglikelihood(self, requests, disable_tqdm=False):
        out = []
        for req in requests:
            context, continuation = req.args
            status, retries, rec = "ok", 0, None
            for attempt in range(MAX_RETRY + 1):
                try:
                    rec = self._score(context, continuation)
                    break
                except Exception as e:                       # noqa: BLE001 - recorded, never silently dropped
                    retries += 1
                    self.retries += 1
                    torch.cuda.empty_cache()
                    if attempt == MAX_RETRY:
                        status = f"failed:{type(e).__name__}"
                        self.failures += 1
                        rec = dict(raw_total_logprob=float("nan"), n_scored_tokens=0, greedy=False,
                                   truncated=False, cont_chars=len(continuation),
                                   input_hash=h(context + "\x00" + continuation), cont_tok_hash="")
            rec.update(status=status, retries=retries)
            key = (req.task_name, req.doc_id, req.idx)
            self.records[key] = rec
            self.order.append(rec["input_hash"])
            self.n_requests += 1
            self.truncated += int(rec["truncated"])
            out.append((rec["raw_total_logprob"], rec["greedy"]))
        return out

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        raise NotImplementedError("stage C uses only loglikelihood tasks")

    def generate_until(self, requests, disable_tqdm=False):
        raise NotImplementedError("stage C uses only loglikelihood tasks")


def target_delimiters(task_name, tm):
    """subtask name -> target delimiter string, read from the locked harness task configs (cached on disk)."""
    from lm_eval.tasks import get_task_dict
    cache = f"{OUT}/target_delimiters_{task_name}.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    out = {}

    def walk(d):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v)
            elif v is not None:
                cfg = getattr(v, "config", None)
                out[k] = getattr(cfg, "target_delimiter", " ") if cfg is not None else " "
    walk(get_task_dict([task_name], task_manager=tm))
    atomic_write_json(cache, out)
    return out


def doc_hash(doc):
    return h(json.dumps(doc, sort_keys=True, default=str))


def build_rows(cond, cfgd, samples, lm, delims):
    """Turn harness samples + adapter records into per-candidate rows.

    Character conventions. `cont_chars` is len() in Python characters of the continuation string actually
    scored, which includes the task's target delimiter. `choice_chars` is len() of the raw choice string
    without that delimiter: this is the denominator the locked harness uses for acc_norm
    (lm_eval Task.process_results: argmax(lls / [len(c) for c in doc_to_choice(doc)])).
    """
    rows, docs = [], 0
    kind = cfgd["kind"]
    for subtask, slist in samples.items():
        for d in slist:
            docs += 1
            did = d["doc_id"]
            dh = doc_hash(d.get("doc", {}))
            args = d.get("arguments", [])
            n = len(args)
            if kind == "lambada":
                rec = lm.records[(subtask, did, 0)]
                ll = rec["raw_total_logprob"]
                nt = rec["n_scored_tokens"]
                rows.append(dict(condition=cond, cfg=cfgd["cfg"], task=cfgd["task"], subtask=subtask,
                                 split=cfgd["split"], doc_id=did, doc_hash=dh,
                                 input_hash=rec["input_hash"], cont_tok_hash=rec["cont_tok_hash"],
                                 target_chars=rec["cont_chars"], n_scored_tokens=nt,
                                 raw_total_logprob=ll, target_nll=-ll,
                                 token_norm_target_nll=(-ll / nt) if nt else float("nan"),
                                 greedy_correct=int(bool(rec["greedy"])),
                                 harness_acc=float(d.get("acc", float("nan"))),
                                 truncated=int(rec["truncated"]), status=rec["status"],
                                 retries=rec["retries"]))
                continue
            gold = d["target"] if isinstance(d["target"], int) else int(d.get("gold", 0))
            cand = []
            for j in range(n):
                rec = lm.records[(subtask, did, j)]
                nt = rec["n_scored_tokens"]
                cand.append((rec, rec["raw_total_logprob"], nt,
                             rec["raw_total_logprob"] / nt if nt else float("nan")))
            dl = len(delims.get(subtask, " "))
            raw = np.array([c[1] for c in cand], dtype=float)
            chars = np.array([max(c[0]["cont_chars"] - dl, 1) for c in cand], dtype=float)
            tnorm = np.array([c[3] for c in cand], dtype=float)
            pred_acc = int(np.argmax(raw))
            pred_accnorm = int(np.argmax(raw / chars))    # harness acc_norm: per character of the RAW choice
            wrong = np.delete(tnorm, gold)
            margin = float(tnorm[gold] - wrong.max()) if len(wrong) else float("nan")
            for j, (rec, ll, nt, tn) in enumerate(cand):
                rows.append(dict(condition=cond, cfg=cfgd["cfg"], task=cfgd["task"], subtask=subtask,
                                 split=cfgd["split"], doc_id=did, doc_hash=dh, n_choices=n, choice_idx=j,
                                 gold_idx=gold, is_gold=int(j == gold), input_hash=rec["input_hash"],
                                 cont_tok_hash=rec["cont_tok_hash"], cont_chars=rec["cont_chars"],
                                 choice_chars=int(rec["cont_chars"] - dl), target_delimiter_chars=dl,
                                 n_scored_tokens=nt, raw_total_logprob=ll, token_norm_logprob=tn,
                                 truncated=int(rec["truncated"]), status=rec["status"],
                                 retries=rec["retries"], pred_acc_idx=pred_acc,
                                 acc_correct=int(pred_acc == gold), pred_accnorm_idx=pred_accnorm,
                                 accnorm_correct=int(pred_accnorm == gold),
                                 harness_acc=float(d.get("acc", float("nan"))),
                                 harness_acc_norm=float(d.get("acc_norm", float("nan"))),
                                 margin_token_norm=margin))
    return rows, docs


def atomic_write_csv(path, fields, rows):
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, path)


def atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def fingerprint(run, ck_sha, cfgd):
    return dict(checkpoint_sha256=ck_sha, train_code_root=run["train_code_root"],
                lm_eval_version=lm_eval.__version__, task=cfgd["task"], limit=cfgd["limit"],
                scorer_version=SCORER_VERSION, seeds=SEEDS,
                task_yaml_sha=h(open(f"{TASKDIR}/sciq_nosupport.yaml").read())
                if cfgd["task"] == "sciq_nosupport" else "")


def run_condition(cond, only=None, force=False):
    write_task_yaml()
    tm = TaskManager(include_path=TASKDIR)
    R = {r["cond"]: r for r in runs()}
    run = R[cond]
    d = f"{OUT}/{slug(cond)}"
    os.makedirs(d, exist_ok=True)
    ck_sha = sha256(run["final_path"])
    todo = [c for c in CONFIGS if only is None or c["cfg"] in only]
    pending = []
    for cfgd in todo:
        done = f'{d}/{cfgd["cfg"]}.done'
        if not force and os.path.exists(done) and json.load(open(done)).get("fingerprint") == fingerprint(run, ck_sha, cfgd):
            print(f'  [reuse] {cond} {cfgd["cfg"]}', flush=True)
            continue
        pending.append(cfgd)
    if not pending:
        return
    set_numerics()
    lm = StageCLM(run=run, forward_mode="native")        # one checkpoint load per condition
    for cfgd in pending:
        t0 = time.time()
        lm.reset()
        kwargs = dict(model=lm, tasks=[cfgd["task"]], batch_size=1, verbosity="ERROR", log_samples=True,
                      task_manager=tm, **{k: v for k, v in SEEDS.items() if k != "torch_random_seed"})
        if cfgd["limit"]:
            kwargs["limit"] = cfgd["limit"]
        res = simple_evaluate(**kwargs)
        delims = target_delimiters(cfgd["task"], tm)
        rows, docs = build_rows(cond, cfgd, res["samples"], lm, delims)
        fields = LAM_FIELDS if cfgd["kind"] == "lambada" else MC_FIELDS
        atomic_write_csv(f'{d}/{cfgd["cfg"]}.csv.gz', fields, rows)
        summary = dict(condition=cond, run_name=run["run_name"], cfg=cfgd["cfg"], task=cfgd["task"],
                       kind=cfgd["kind"], limit=cfgd["limit"], forward_mode="native",
                       operator=lm.loaded.op, n_docs=docs, n_rows=len(rows), n_requests=lm.n_requests,
                       n_subtasks=len(res["samples"]), truncated=lm.truncated,
                       truncation_rate=lm.truncated / max(lm.n_requests, 1),
                       failures=lm.failures, retries=lm.retries,
                       order_hash=h("".join(lm.order)),
                       target_delimiters=sorted(set(delims.values())),
                       acc_norm_rule="argmax(total_logprob / len(raw choice string in characters)), "
                                     "delimiter excluded; verbatim lm-eval Task.process_results",
                       harness_results=res["results"], seconds=round(time.time() - t0, 1),
                       fingerprint=fingerprint(run, ck_sha, cfgd))
        atomic_write_json(f'{d}/{cfgd["cfg"]}_summary.json', summary)
        atomic_write_json(f'{d}/{cfgd["cfg"]}.done',
                          dict(fingerprint=summary["fingerprint"], order_hash=summary["order_hash"],
                               n_rows=len(rows), finished=time.strftime("%Y-%m-%dT%H:%M:%S")))
        main = list(res["results"].values())[0] if res["results"] else {}
        print(f'  {cond:18s} {cfgd["cfg"]:16s} docs={docs:6d} rows={len(rows):6d} '
              f'trunc={summary["truncation_rate"]:.4f} fail={lm.failures} '
              f'{ {k: round(v, 4) for k, v in main.items() if isinstance(v, float)} } '
              f'({summary["seconds"]:.0f}s)', flush=True)
    del lm
    torch.cuda.empty_cache()


def build_manifest():
    write_task_yaml()
    os.makedirs(OUT, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    out = dict(built=time.strftime("%Y-%m-%dT%H:%M:%S"), host=platform.node(),
               python=platform.python_version(), torch=torch.__version__,
               lm_eval_version=lm_eval.__version__, tiktoken_encoding="gpt2",
               tokenizer_n_vocab=enc.n_vocab, context_length=BLOCK,
               inference_dtype="bf16 autocast outside fp32 attention; fp32 logits and log_softmax",
               tf32="disabled", batch_size=1, forward_mode="native", seeds=SEEDS,
               scorer_version=SCORER_VERSION, configs=CONFIGS,
               sciq_nosupport_yaml_sha=h(open(f"{TASKDIR}/sciq_nosupport.yaml").read()),
               output_root=OUT, conditions=[])
    for r in runs():
        ck = torch.load(r["final_path"], map_location="cpu", weights_only=False, mmap=True)
        c = ck["config"]
        step = int(ck["step"])
        blk = int(c["block_size"])
        tok_per_step = int(c["batch_tokens"])          # optimiser-step token budget recorded by the trainer
        tokens_done = int(ck.get("tokens_done", step * tok_per_step))
        out["conditions"].append(dict(
            condition=r["cond"], run_name=r["run_name"], label=r["label"], K=r["K"],
            calibration=r["calibration"], reconstruction=r["reconstruction"], backward=r["backward"],
            range_tau=r["range_tau"], range_grad=r["range_grad"], seed=int(r["seed"]),
            dataset=r["dataset"], model_scale=r["model_scale"],
            nominal_token_horizon_m=r["token_horizon_m"],
            checkpoint_path=r["final_path"], checkpoint_sha256=sha256(r["final_path"]),
            checkpoint_bytes=os.path.getsize(r["final_path"]),
            train_code_root=r["train_code_root"],
            train_code_sha256=h("".join(sorted(
                sha256(os.path.join(dp, fn)) for dp, _, fns in os.walk(r["train_code_root"])
                for fn in fns if fn.endswith(".py")))),
            n_layer=int(c["n_layer"]), n_head=int(c["n_head"]), n_embd=int(c["n_embd"]),
            block_size=blk, micro_batch=int(c.get("micro_batch", 0)), batch_tokens=tok_per_step,
            train_data_dir=c["data_dir"], max_tokens_configured=int(c.get("max_tokens", 0)),
            lr=float(c.get("lr", 0)), warmup_tokens=int(c.get("warmup_tokens", 0)),
            final_step=step, tokens_per_step=tok_per_step,
            tokens_trained_checkpoint_field=tokens_done,
            tokens_trained_step_times_batch=step * tok_per_step,
            operator_raw=dict(attn=c.get("attn"), K=int(c.get("K", 0)), approx=c.get("approx"),
                              range_mode=c.get("range_mode"), range_tau=c.get("range_tau"),
                              range_grad=c.get("range_grad")),
            operator_effective=loaded_op(c),
            n_params_state_dict=int(sum(v.numel() for v in ck["model"].values()))))
        del ck
    atomic_write_json(f"{OUT}/manifest_stageC.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "conditions"}, indent=2, default=str))
    for c in out["conditions"]:
        print(f'{c["condition"]:18s} step={c["final_step"]} tok/step={c["tokens_per_step"]} '
              f'tokens_done={c["tokens_trained_checkpoint_field"]/1e9:.6f}B '
              f'(step*batch={c["tokens_trained_step_times_batch"]/1e9:.6f}B) '
              f'sha={c["checkpoint_sha256"][:12]} op={c["operator_effective"]}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--only", help="comma-separated config ids")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    only = a.only.split(",") if a.only else None
    if a.manifest:
        build_manifest()
        return
    if a.preflight:
        sys.modules["layer3_stageC"] = sys.modules["__main__"]   # reuse this module, keep one registration
        import layer3_stageC_preflight as pf
        sys.exit(0 if pf.main() else 1)
    conds = [r["cond"] for r in runs()] if a.all else [a.condition]
    for c in conds:
        print(f"=== {c} ===", flush=True)
        run_condition(c, only=only, force=a.force)


if __name__ == "__main__":
    main()

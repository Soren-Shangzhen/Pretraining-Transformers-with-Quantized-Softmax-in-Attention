#!/usr/bin/env python3
"""13.3 checks of the rb-root_v1 training-script mechanics (CPU, tiny GPT, deterministic).

E2 resume / switch:
  - straight 6 steps == (3 steps, stop) + (--init_from, same operator, 3 steps): log rows of steps
    4-6 and final weights, optimizer state and data-RNG state bitwise identical (data cursor and
    learning-rate position are restored exactly);
  - switching the surrogate at the branch point leaves the first post-branch training loss
    identical (same forward), and is refused without --branch_override;
  - a protocol mismatch (learning rate) is refused; a historical parent without a recorded
    surrogate is refused unless --parent_surrogate is given; the branch provenance is recorded;
  - --stop_at_tokens keeps the learning-rate schedule of --max_tokens;
General:
  - --ckpt_weights_only stores weights only in intermediate checkpoints and full state at the end
    and at --ckpt_full_tokens;
  - --halt_on_nonfinite stops the run with exit code 3 and NONFINITE.json;
  - --resume of an rb run with the new operator fields works and refuses a changed operator.
"""
import csv, json, os, subprocess, sys, tempfile
import numpy as np
import torch
from rb_common import RB_ROOT, write_json, env_info

PY = sys.executable
RES = {"env": env_info(), "checks": []}
OK = True
STEP = 256
COMMON = ["--n_layer", "2", "--n_head", "2", "--n_embd", "64", "--block_size", "64", "--batch_tokens", str(STEP),
          "--micro_batch", "1", "--max_tokens", str(6 * STEP), "--warmup_tokens", str(2 * STEP),
          "--eval_every_tokens", str(STEP), "--eval_batches", "2", "--seed", "7", "--device", "cpu",
          "--health_every_steps", "1"]
NEAR_W = ["--attn", "quant", "--approx", "nearest", "--K", "4", "--range_grad", "full", "--nearest_surrogate", "weight_strict"]
NEAR_P = ["--attn", "quant", "--approx", "nearest", "--K", "4", "--range_grad", "full", "--nearest_surrogate", "prob"]


def record(name, passed, **info):
    global OK
    OK &= bool(passed)
    RES["checks"].append(dict(name=name, passed=bool(passed), **info))
    print(f"{'PASS' if passed else 'FAIL'}  {name}  " + " ".join(f"{k}={v}" for k, v in info.items()), flush=True)


def train(data, out, name, extra, expect_fail=False):
    cmd = [PY, "train_formal.py", "--run_name", name, "--data_dir", data, "--out_dir", out] + COMMON + extra
    r = subprocess.run(cmd, cwd=RB_ROOT, capture_output=True, text=True)
    if (r.returncode != 0) != expect_fail:
        raise RuntimeError(f"{name}: rc={r.returncode}\n{r.stdout[-1500:]}\n{r.stderr[-2500:]}")
    return r


def log_rows(rd):
    return [(a["step"], a["tokens"], a["train_loss"], a["val_loss"], a["lr"]) for a in csv.DictReader(open(os.path.join(rd, "log.csv")))]


def ck(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def same_state(a, b):
    w = all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    o = all(torch.equal(a["opt"]["state"][i][f], b["opt"]["state"][i][f]) for i in a["opt"]["state"] for f in a["opt"]["state"][i])
    return w, o, a["train_rng_state"] == b["train_rng_state"], (a["step"], a["tokens_done"]) == (b["step"], b["tokens_done"])


def main():
    with tempfile.TemporaryDirectory(dir=os.environ.get("RB_TMP")) as tmp:
        data = os.path.join(tmp, "data"); os.makedirs(data)
        rng = np.random.default_rng(0)
        rng.integers(0, 50000, 200_000, dtype=np.uint16).tofile(os.path.join(data, "train.bin"))
        rng.integers(0, 50000, 20_000, dtype=np.uint16).tofile(os.path.join(data, "val.bin"))
        out = os.path.join(tmp, "runs")

        # straight run A (6 steps) and stopped run B (3 steps of the 6-step schedule)
        train(data, out, "A", NEAR_W + ["--ckpt_every_tokens", str(6 * STEP)])
        train(data, out, "B", NEAR_W + ["--ckpt_every_tokens", str(6 * STEP), "--stop_at_tokens", str(3 * STEP)])
        a_rows, b_rows = log_rows(os.path.join(out, "A")), log_rows(os.path.join(out, "B"))
        record("--stop_at_tokens: first 3 steps identical to the full-schedule run (same LR schedule)",
               a_rows[:3] == b_rows and len(b_rows) == 3, b_rows=b_rows)
        bck = ck(os.path.join(out, "B", "ckpt.pt"))
        record("stopped run saves a full checkpoint (optimizer + data RNG) at the stop", "opt" in bck and "train_rng_state" in bck
               and bck["step"] == 3)
        # C: continue B in a new directory, same operator
        train(data, out, "C", NEAR_W + ["--ckpt_every_tokens", str(6 * STEP), "--init_from", os.path.join(out, "B", "ckpt.pt")])
        c_rows = log_rows(os.path.join(out, "C"))
        w, o, r, sc = same_state(ck(os.path.join(out, "A", "ckpt.pt")), ck(os.path.join(out, "C", "ckpt.pt")))
        record("--init_from continuation == straight run: logged rows of steps 4-6", c_rows == a_rows[3:], c_rows=c_rows)
        record("--init_from continuation == straight run: final weights, optimizer state, data RNG, step/tokens bitwise",
               w and o and r and sc, weights=w, opt=o, rng=r, step_tokens=sc)
        cc = ck(os.path.join(out, "C", "ckpt.pt"))["config"]
        record("branch provenance recorded in the checkpoint config",
               cc.get("branch_parent", "").endswith("B/ckpt.pt") and len(cc.get("branch_parent_sha256", "")) == 64
               and cc.get("branch_parent_step") == 3 and cc.get("branch_operator_changes") == {},
               parent_step=cc.get("branch_parent_step"), changes=cc.get("branch_operator_changes"))

        # surrogate switch at the branch point
        r_ref = train(data, out, "D_noover", NEAR_P + ["--init_from", os.path.join(out, "B", "ckpt.pt")], expect_fail=True)
        record("operator change without --branch_override is refused", "--branch_override" in (r_ref.stderr + r_ref.stdout))
        train(data, out, "D", NEAR_P + ["--ckpt_every_tokens", str(6 * STEP), "--init_from", os.path.join(out, "B", "ckpt.pt"),
                                         "--branch_override"])
        d_rows = log_rows(os.path.join(out, "D"))
        dc = ck(os.path.join(out, "D", "ckpt.pt"))["config"]
        record("W->P switch: train loss of the first post-branch step identical (same forward, same data, same LR)",
               d_rows[0][:3] == a_rows[3][:3] and d_rows[0][4] == a_rows[3][4], branch_row=d_rows[0], straight_row=a_rows[3])
        record("W->P switch: later steps differ (the backward changed)", d_rows[1:] != a_rows[3:])
        record("W->P switch recorded as an operator change", dc.get("branch_operator_changes") == {"nearest_surrogate": ["weight_strict", "prob"]})
        r_bad = train(data, out, "E_lr", NEAR_W + ["--lr", "1e-3", "--init_from", os.path.join(out, "B", "ckpt.pt")], expect_fail=True)
        record("protocol mismatch (lr) is refused", "protocol mismatch" in (r_bad.stderr + r_bad.stdout))

        # historical parent without a recorded surrogate
        hist = ck(os.path.join(out, "B", "ckpt.pt"))
        hist["config"].pop("nearest_surrogate")
        hp = os.path.join(tmp, "hist_parent.pt"); torch.save(hist, hp)
        r_h = train(data, out, "F_nohint", NEAR_W + ["--init_from", hp, "--branch_override"], expect_fail=True)
        record("historical parent without surrogate: refused unless --parent_surrogate is given", "--parent_surrogate" in (r_h.stderr + r_h.stdout))
        train(data, out, "F", NEAR_W + ["--init_from", hp, "--branch_override", "--parent_surrogate", "weight_hist",
                                         "--parent_code_root", RB_ROOT])
        fc = ck(os.path.join(out, "F", "ckpt.pt"))["config"]
        record("historical W -> strict W recorded as weight_hist -> weight_strict, parent code hashes recorded",
               fc.get("branch_operator_changes") == {"nearest_surrogate": ["weight_hist", "weight_strict"]}
               and len(fc.get("parent_code_sha256", {})) == 2)

        # weights-only intermediate checkpoints
        train(data, out, "G", NEAR_W + ["--ckpt_every_tokens", str(STEP), "--ckpt_weights_only",
                                         "--ckpt_full_tokens", str(2 * STEP)])
        files = sorted(f for f in os.listdir(os.path.join(out, "G")) if f.startswith("ckpt_step"))
        kinds = {f: ("opt" in ck(os.path.join(out, "G", f))) for f in files}
        expect = {f: (f.startswith("ckpt_step000002") or f.startswith("ckpt_step000006")) for f in files}
        record("--ckpt_weights_only: optimizer state only at --ckpt_full_tokens and at the end", kinds == expect and len(files) == 6,
               kinds=kinds)
        g_rows = log_rows(os.path.join(out, "G"))
        record("checkpoint format does not change training", g_rows == a_rows)

        # resume of an rb run, operator check covers the new fields
        train(data, out, "H", ["--attn", "exaq", "--attn_variant", "graph", "--range_grad", "full",
                               "--ckpt_every_tokens", str(6 * STEP), "--stop_at_tokens", str(3 * STEP)])
        r_res = train(data, out, "H", ["--attn", "exaq", "--attn_variant", "detach", "--range_grad", "full", "--resume"], expect_fail=True)
        record("--resume refuses a changed new-mode field (attn_variant)", "attn_variant" in (r_res.stderr + r_res.stdout))
        train(data, out, "H", ["--attn", "exaq", "--attn_variant", "graph", "--range_grad", "full", "--resume",
                               "--ckpt_every_tokens", str(6 * STEP)])
        record("--resume continues an rb run to the end of the schedule", log_rows(os.path.join(out, "H"))[-1][0] == "6")

        # non-finite halt
        r_nf = train(data, out, "N", NEAR_W + ["--lr", "1e30", "--halt_on_nonfinite", "--max_tokens", str(6 * STEP)],
                     expect_fail=True)
        nf = os.path.join(out, "N", "NONFINITE.json")
        record("--halt_on_nonfinite: exit code 3 and NONFINITE.json", r_nf.returncode == 3 and os.path.exists(nf),
               rc=r_nf.returncode, record=json.load(open(nf)) if os.path.exists(nf) else None)
        env = json.load(open(os.path.join(out, "A", "env.json")))
        record("env.json records torch, code-root SHA-256 and the TF32 probe",
               env["code_sha256"]["quant_attention.py"] == RES["env"]["rb_root_sha256"]["quant_attention.py"] and "tf32" in env)
    RES["all_pass"] = OK
    print("written", write_json(f"13_3_train_mechanics_torch{torch.__version__.split('+')[0]}.json", RES))
    print("ALL PASS" if OK else "SOME FAILED")
    sys.exit(0 if OK else 1)


if __name__ == "__main__":
    main()

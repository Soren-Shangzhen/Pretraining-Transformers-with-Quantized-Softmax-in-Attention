#!/usr/bin/env python3
"""Run queue for the rb- runs (plan 3 and 13.4).  One run per GPU (CUDA_VISIBLE_DEVICES), or
--per-gpu 2 if the packing test passed and the user approved it.

- Runs come from manifests/rb-runs_v5_manifest.json, in the order of a queue file (one run_id per line;
  '#' comments).  Placeholders ${RB_DATA} ${RB_RUNS} ${RB_CODE} ${RB_PARENTS} come from the environment;
  {E3A_2P5B_VARIANT} must be resolved by the user's decision before that run is queued.
- Before a run starts, a start record is appended to manifests/rb-run_log.jsonl: command, environment,
  code-root SHA-256 recomputed on this machine (must equal the manifest), seed, pairing, prediction and
  reading rule.  An existing run directory is never reused: the run is skipped and reported.
- A run whose log shows a non-finite value is stopped and reported (rb-root_v1 runs halt by themselves,
  exit code 3; other roots are watched for 'non-finite'/'nan' in their output).  Nothing is restarted.
- When the queue is empty, QUEUE_EMPTY is written and --notify-cmd (if given) is executed.
- A queue line '@wait PATH' is a hold point (2026-11-10, machine B: E3a starts once its gates pass, the other runs
  only after the remaining checks): no later run starts until PATH exists; running runs are still watched.

- Follow-on runner (2026-11-10, machine B: keep all GPUs busy across waves without stopping the running runner):
  --start-after Q waits until every run of queue file Q has been dispatched by another runner (an event in the run
  log), and --gpu-free-check uses a GPU only when nvidia-smi shows no compute process on it.

usage: rb_queue.py --queue QUEUE.txt[,QUEUE2.txt,...] --gpus 0,1,2,3,4,5,6,7 [--per-gpu 1] [--threads N] [--notify-cmd CMD]
                   [--start-after OTHER_QUEUE.txt --gpu-free-check]
       rb_queue.py --queue QUEUE.txt --dry-run
"""
import argparse, hashlib, json, os, re, shlex, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
RB = os.path.abspath(os.path.join(HERE, "..", ".."))
MAN = os.path.join(RB, "manifests", "rb-runs_v5_manifest.json")
LOG = os.path.join(RB, "manifests", "rb-run_log.jsonl")
NONFINITE = re.compile(r"non-finite|\bnan\b|\binf\b", re.I)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


RESOLVED = {}     # --resolve NAME=VALUE: user decisions filled into {NAME} placeholders (e.g. E3A_2P5B_VARIANT=u8)


def expand(s):
    for k, v in RESOLVED.items():
        s = s.replace("{" + k + "}", v)
    if "${RB_PY}" in s and not os.environ.get("RB_PY"):
        raise SystemExit("RB_PY (python of the S1 stack) must be set")
    out = os.path.expandvars(s)
    if "${" in out or "{E3A_2P5B_VARIANT}" in out:
        raise SystemExit(f"unresolved placeholder in: {out[:200]}")
    return out


DISPATCHED = {"start", "skipped_existing_dir", "code_hash_mismatch", "missing_inputs", "clamp_boundary_check_failed"}


def read_queue(paths):
    out = []
    for q in paths.split(","):
        out += [l.split("#")[0].strip() for l in open(q)]
    return [x for x in out if x]


def dispatched_ids():
    if not os.path.exists(LOG):
        return set()
    ids = set()
    for l in open(LOG):
        try:
            d = json.loads(l)
        except ValueError:
            continue
        if d.get("event") in DISPATCHED and d.get("run_id"):
            ids.add(d["run_id"])
    return ids


def gpus_with_compute():
    q = lambda f: subprocess.run(["nvidia-smi", f"--query-{f}", "--format=csv,noheader"], capture_output=True,
                                 text=True, check=True).stdout
    idx = {u.strip(): i.strip() for i, u in (l.split(",") for l in q("gpu=index,uuid").splitlines() if l.strip())}
    return {idx[l.split(",")[1].strip()] for l in q("compute-apps=pid,gpu_uuid").splitlines() if l.strip()}


def recently_started_gpus(window=float(os.environ.get("RB_START_GRACE_S", "180"))):
    """GPUs given to a run (by any runner on this machine) in the last `window` seconds: a new process needs a few
    seconds before its CUDA context shows in nvidia-smi."""
    if not os.path.exists(LOG):
        return set()
    now, out = time.time(), set()
    for l in open(LOG):
        try:
            d = json.loads(l)
        except ValueError:
            continue
        if d.get("event") == "start" and now - float(d.get("time", 0)) < window and d.get("host") == os.uname().nodename:
            out.add(str(d.get("gpu")))
    return out


def append_log(rec):
    with open(LOG, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--per-gpu", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0, help="OMP/MKL threads per run (default: cores / slots)")
    ap.add_argument("--notify-cmd", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--manifest", default=MAN)
    ap.add_argument("--poll", type=float, default=30.0)
    ap.add_argument("--start-after", default="", help="queue file whose runs must all be dispatched first")
    ap.add_argument("--gpu-free-check", action="store_true", help="use a GPU only if it has no compute process")
    ap.add_argument("--resolve", action="append", default=[], help="NAME=VALUE for a {NAME} placeholder (a user decision)")
    a = ap.parse_args()
    RESOLVED.update(dict(x.split("=", 1) for x in a.resolve))
    global LOG
    if a.manifest != MAN:
        LOG = os.path.join(os.path.dirname(os.path.abspath(a.manifest)), "rb-run_log.jsonl")
    man = {r["run_id"]: r for r in json.load(open(a.manifest))["runs"]}
    queue = read_queue(a.queue)
    unknown = [q for q in queue if q not in man and not q.startswith("@wait ")]
    if unknown:
        raise SystemExit(f"not in the manifest: {unknown}")
    slots = [g for g in a.gpus.split(",") for _ in range(a.per_gpu)]
    threads = a.threads or max(1, (os.cpu_count() or 8) // len(slots))
    runs_dir = os.environ.get("RB_RUNS", "")
    if a.dry_run:
        for q in queue:
            if q.startswith("@wait "):
                print(q.split(None, 1)[0], expand(q.split(None, 1)[1]))
                continue
            r = man[q]
            print(q, r["env"], expand(r["command"]))
        return
    if not runs_dir or not os.path.isdir(runs_dir):
        raise SystemExit("RB_RUNS must point to an existing directory")
    if a.start_after:
        before = [q for q in read_queue(a.start_after) if not q.startswith("@")]
        print(f"WAIT: starts after all {len(before)} runs of {a.start_after} have been dispatched", flush=True)
        append_log(dict(event="follow_on_wait", queue=a.queue, after=a.start_after, time=time.time(), host=os.uname().nodename))
        while not set(before) <= dispatched_ids():
            time.sleep(a.poll)
        print(f"GO: all runs of {a.start_after} dispatched", flush=True)
    running = {}          # slot index -> (run_id, Popen, logfile)
    pending = list(queue)
    held = None
    while pending or running:
        busy = (gpus_with_compute() | recently_started_gpus()) if a.gpu_free_check and pending else set()
        for i, g in enumerate(slots):
            if i in running or not pending:
                continue
            if g in busy:
                continue
            if pending[0].startswith("@wait "):
                hold = expand(pending[0].split(None, 1)[1])
                if not os.path.exists(hold):
                    if held != hold:
                        held = hold
                        print(f"HOLD: no further run starts until {hold} exists", flush=True)
                        append_log(dict(event="hold", path=hold, time=time.time(), host=os.uname().nodename))
                    break
                print(f"RELEASED by {hold}", flush=True)
                append_log(dict(event="released", path=hold, time=time.time(), host=os.uname().nodename,
                                release_note=open(hold, errors="ignore").read()[:2000]))
                pending.pop(0)
                held = None
                if not pending:
                    break
            rid = pending.pop(0)
            r = man[rid]
            rd = os.path.join(runs_dir, rid)
            if os.path.exists(rd):
                print(f"SKIP {rid}: run directory exists ({rd}); not overwritten", flush=True)
                append_log(dict(run_id=rid, event="skipped_existing_dir", time=time.time()))
                continue
            cmd = expand(r["command"])
            root = expand(r["code_root"])
            # branch runs: the parent checkpoint and the parent's code root must exist here (train_formal.py skips a
            # missing --parent_code_root silently, which would lose the recorded parent code hashes)
            miss = [m for m in re.findall(r"--init_from\s+(\S+)", cmd) if not os.path.isfile(m)]
            miss += [m for m in re.findall(r"--parent_code_root\s+(\S+)", cmd)
                     if not all(os.path.isfile(os.path.join(m, f)) for f in ("model.py", "quant_attention.py"))]
            if miss:
                print(f"STOP {rid}: missing on this machine: {miss}", flush=True)
                append_log(dict(run_id=rid, event="missing_inputs", missing=miss, time=time.time()))
                continue
            got = {f: sha(os.path.join(root, f)) for f in ("quant_attention.py", "model.py")}
            if got["quant_attention.py"] != r["code_sha256_quant_attention"] or got["model.py"] != r["code_sha256_model"]:
                print(f"STOP {rid}: code root hash differs from the manifest {got}", flush=True)
                append_log(dict(run_id=rid, event="code_hash_mismatch", got=got, time=time.time()))
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
            if r["experiment"] == "E11":
                # user rule 2026-11-10: E11 only where clamp passes gradient 1 at its bounds (CPU and this GPU)
                cc = subprocess.run([os.environ["RB_PY"], os.path.join(HERE, "clamp_boundary_check.py")], env=env,
                                    capture_output=True, text=True)
                if cc.returncode != 0:
                    print(f"STOP {rid}: clamp boundary gradient is not 1 here: {cc.stdout.strip()}", flush=True)
                    append_log(dict(run_id=rid, event="clamp_boundary_check_failed", detail=cc.stdout.strip(), time=time.time()))
                    continue
            env.update(json.loads(r["env"]))
            os.makedirs(os.path.join(runs_dir, "_logs"), exist_ok=True)
            lf = os.path.join(runs_dir, "_logs", f"{rid}.log")
            append_log(dict(run_id=rid, event="start", time=time.time(), host=os.uname().nodename, gpu=g,
                            command=cmd, env=json.loads(r["env"]), code_sha256=got, seed=r["seed"],
                            pair_with=r["pair_with"], prediction=r["prediction"], reading_rule=r["reading_rule"],
                            pending_decisions=r["pending_decisions"], resolved=dict(RESOLVED)))
            fh = open(lf, "w")
            p = subprocess.Popen(cmd, shell=True, stdout=fh, stderr=subprocess.STDOUT, env=env, preexec_fn=os.setsid)
            running[i] = (rid, p, lf, fh)
            print(f"START {rid} on GPU {g} (pid {p.pid})", flush=True)
        time.sleep(a.poll)
        for i in list(running):
            rid, p, lf, fh = running[i]
            rc = p.poll()
            tail = open(lf, errors="ignore").read()[-4000:]
            bad = [l for l in tail.splitlines() if NONFINITE.search(l)
                   and any(k in l for k in ("grad", "gnorm", "train", "loss", "HALT"))]
            if rc is None and bad:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait()
                rc = "killed_nonfinite"
            if rc is None:
                continue
            fh.close()
            ev = {0: "finished", 3: "halted_nonfinite"}.get(rc, f"failed_rc_{rc}") if isinstance(rc, int) else rc
            append_log(dict(run_id=rid, event=ev, time=time.time(), last_lines=tail.splitlines()[-5:]))
            print(f"{ev.upper()} {rid}", flush=True)
            del running[i]
    tag = "" if "," not in a.queue and not a.start_after else "_" + "+".join(
        os.path.splitext(os.path.basename(q))[0] for q in a.queue.split(","))
    open(os.path.join(runs_dir, "QUEUE_EMPTY" + tag), "w").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print("QUEUE EMPTY", flush=True)
    if a.notify_cmd:
        subprocess.run(a.notify_cmd, shell=True)


if __name__ == "__main__":
    main()

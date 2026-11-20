#!/usr/bin/env python3
"""Evaluation queue for machine B (user decision 2026-11-12: a GPU that becomes free starts the next evaluation).

- Starts only after every training run of the given queue files has been dispatched (the training runners never
  start anything afterwards, so the two cannot race for a GPU).
- A GPU is used only when nvidia-smi shows no compute process on it and nothing was started on it in the last
  3 minutes (rb_queue.py helpers); one unit per GPU.
- Units come from $RB_RUNS/_eval/units.jsonl (rb_eval_units.py), re-read every cycle so that later phases can be
  appended while the queue runs.  The next unit is the one with the lowest phase (then file order) whose
  `requires` paths all exist; a unit whose inputs are not there yet waits and the next runnable one starts.
- Each unit's exit is logged in _eval/queue_log.jsonl (done / failed with the last lines); nothing is retried.
- Stops when _eval/STOP exists (running units finish first).  Status: _eval/queue_status.json.
usage (on B): rb_eval_queue.py --gpus 0,1,2,3,4,5,6,7 --start-after Q1.txt,Q2.txt,... [--threads 24]
"""
import argparse, json, os, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import rb_queue  # noqa: E402  (gpus_with_compute, recently_started_gpus, read_queue, dispatched_ids)


def jl(path):
    if not os.path.exists(path):
        return []
    out = []
    for l in open(path):
        try:
            out.append(json.loads(l))
        except ValueError:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--start-after", default="")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--poll", type=float, default=30.0)
    a = ap.parse_args()
    ev = os.path.join(os.environ["RB_RUNS"], "_eval")
    os.makedirs(os.path.join(ev, "logs"), exist_ok=True)
    unf, logf, stopf = (os.path.join(ev, x) for x in ("units.jsonl", "queue_log.jsonl", "STOP"))
    gpus = a.gpus.split(",")
    threads = a.threads or max(1, (os.cpu_count() or 8) // len(gpus))
    host = os.uname().nodename

    def log(rec):
        rec.update(time=time.time(), host=host)
        with open(logf, "a") as f:
            f.write(json.dumps(rec) + "\n")

    if a.start_after:
        before = [q for q in rb_queue.read_queue(a.start_after) if not q.startswith("@")]
        print(f"WAIT: evaluation starts after all {len(before)} training runs have been dispatched", flush=True)
        while not set(before) <= rb_queue.dispatched_ids():
            time.sleep(a.poll)
        print("GO: all training runs dispatched", flush=True)
    ended = {e["unit_id"]: e["event"] for e in jl(logf) if e.get("event") in ("done", "failed")}
    running = {}                     # gpu -> (unit, Popen, fh)
    while not (os.path.exists(stopf) and not running):
        units = jl(unf)
        if not os.path.exists(stopf):
            busy = rb_queue.gpus_with_compute() | rb_queue.recently_started_gpus()
            taken = {u["unit_id"] for u, _, _ in running.values()}
            pending = sorted((u for u in units if u["unit_id"] not in ended and u["unit_id"] not in taken),
                             key=lambda u: u["phase"])          # stable: file order inside a phase
            for g in gpus:
                if g in running or g in busy:
                    continue
                nxt = next((u for u in pending if all(os.path.exists(p) for p in u.get("requires", []))), None)
                if nxt is None:
                    break
                pending.remove(nxt)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, OMP_NUM_THREADS=str(threads),
                           MKL_NUM_THREADS=str(threads))
                fh = open(os.path.join(ev, "logs", f"{nxt['unit_id']}.log"), "a")
                p = subprocess.Popen(nxt["cmd"], shell=True, stdout=fh, stderr=subprocess.STDOUT, env=env,
                                     preexec_fn=os.setsid)
                running[g] = (nxt, p, fh)
                log(dict(event="start", unit_id=nxt["unit_id"], phase=nxt["phase"], gpu=g, pid=p.pid))
                print(f"START {nxt['unit_id']} on GPU {g}", flush=True)
        time.sleep(a.poll)
        for g in list(running):
            u, p, fh = running[g]
            rc = p.poll()
            if rc is None:
                continue
            fh.close()
            tail = open(os.path.join(ev, "logs", f"{u['unit_id']}.log"), errors="ignore").read()[-3000:].splitlines()[-5:]
            ev_name = "done" if rc == 0 else "failed"
            ended[u["unit_id"]] = ev_name
            log(dict(event=ev_name, unit_id=u["unit_id"], phase=u["phase"], gpu=g, rc=rc, last_lines=tail))
            print(f"{ev_name.upper()} {u['unit_id']} (rc {rc})", flush=True)
            del running[g]
        units = jl(unf)
        st = dict(time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), units=len(units),
                  done=sum(1 for v in ended.values() if v == "done"), failed=sorted(k for k, v in ended.items() if v == "failed"),
                  running={g: u["unit_id"] for g, (u, _, _) in running.items()},
                  waiting_for_inputs=sorted(u["unit_id"] for u in units if u["unit_id"] not in ended
                                            and not all(os.path.exists(p) for p in u.get("requires", []))),
                  by_phase={str(ph): [sum(1 for u in units if u["phase"] == ph and ended.get(u["unit_id"]) == "done"),
                                      sum(1 for u in units if u["phase"] == ph)] for ph in sorted({u["phase"] for u in units})})
        json.dump(st, open(os.path.join(ev, "queue_status.json"), "w"), indent=1)
    print("STOPPED", flush=True)


if __name__ == "__main__":
    main()

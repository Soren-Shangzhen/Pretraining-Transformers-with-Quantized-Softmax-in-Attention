#!/usr/bin/env python3
"""Write manifests/PRETRAINING_RECORD.txt: sha256 of the protocol, the method description, the frozen code root,
the evaluator copy, the check scripts and their outputs.  Run once before the first training run."""
import hashlib, os, datetime as dt
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../..")))  # repository root
C = (KI_REPO + "/analysis_main_v1/centroid_v1_20261013")
FILES = ["manifests/protocol_centroid_v1.json", "manifests/method_description_input.md",
         "code/centroid_root/model.py", "code/centroid_root/quant_attention.py", "code/centroid_root/train.py",
         "code/eval/cv1_eval.py", "code/eval/common.py", "code/eval/geometry_probe_v1.py",
         "code/checks/check_centroid_ops.py", "code/checks/check_gpu_step0.py",
         "checks/check_centroid_ops.json", "checks/check_gpu_step0.json",
         "code/jobs/run_training_queue.sh", "code/jobs/eval_after_run.sh"]
def sha(p):
    h = hashlib.sha256(); h.update(open(os.path.join(C, p), "rb").read()); return h.hexdigest()
lines = [f"{sha(p)}  {p}" for p in FILES]
out = os.path.join(C, "manifests", "PRETRAINING_RECORD.txt")
with open(out, "w") as f:
    f.write("# centroid_v1 pre-training record, written " + dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
            + " before the first training run\n")
    f.write("\n".join(lines) + "\n")
print(open(out).read())

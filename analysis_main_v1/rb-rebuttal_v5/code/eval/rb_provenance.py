"""Provenance block written into every evaluation result (user decision 2026-11-12: intermediate checkpoints are
gone once machine B is released, so each result must name the exact file it came from).

  checkpoint   path, size, SHA-256 computed from the file as it is read for this evaluation, and the value the
               checkpoint lists record for that file: B's manifest (runs/_ckpt_sha256.tsv, b_ckpt_manifest.py),
               the SHA256SUMS of the uploaded copies (A_eval_model_only, endpoints_from_local, parents_from_local),
               or the transfer record of the historical checkpoints (manifests/rb-recorded_sha256.tsv);
               matches_manifest is None when the file is not listed yet (B's manifest is filled while runs finish;
               rb_eval_report.py repeats the comparison against the final manifest).
  code_root    SHA-256 of every top-level .py file of the code root used for the forward, and one hash over them.
  scripts      SHA-256 of the evaluation script and its helper modules.
"""
import csv, hashlib, os

HERE = os.path.dirname(os.path.abspath(__file__))
RB = os.path.abspath(os.path.join(HERE, "..", ".."))
_MAN, _ROOTS = None, {}


def file_sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def _manifests():
    global _MAN
    if _MAN is not None:
        return _MAN
    m = {}
    runs = os.environ.get("RB_RUNS", "")
    p = os.path.join(runs, "_ckpt_sha256.tsv")
    if runs and os.path.exists(p):
        for r in csv.DictReader(open(p), delimiter="\t"):
            if r.get("sha256"):
                m[os.path.realpath(os.path.join(runs, r["run"], r["file"]))] = (r["sha256"], "B checkpoint manifest runs/_ckpt_sha256.tsv")
    for d in (os.environ.get("RB_A_EVAL", "<POD_WORKSPACE>/A_eval_model_only"),
              os.environ.get("RB_LOCAL_ENDPOINTS", "<POD_WORKSPACE>/endpoints_from_local"),
              os.environ.get("RB_PARENTS", "<POD_WORKSPACE>/parents_from_local")):
        s = os.path.join(d, "SHA256SUMS")
        if os.path.exists(s):
            for l in open(s):
                f = l.split()
                if len(f) >= 2:
                    m[os.path.realpath(os.path.join(d, f[1]))] = (f[0], s)
    rec = os.path.join(RB, "manifests", "rb-recorded_sha256.tsv")
    cloud = os.environ.get("RB_CLOUD", "<CLOUD_DISK>")
    if os.path.exists(rec):
        for r in csv.DictReader(open(rec), delimiter="\t"):
            rel = r["transfer_path"][2:] if r["transfer_path"].startswith("./") else r["transfer_path"]
            m[os.path.realpath(os.path.join(cloud, rel))] = (r["recorded_sha256"], "manifests/rb-recorded_sha256.tsv (private transfer record)")
    _MAN = m
    return m


def code_root(root):
    root = os.path.abspath(root)
    if root not in _ROOTS:
        files = sorted(f for f in os.listdir(root) if f.endswith(".py"))
        per = {f: file_sha256(os.path.join(root, f)) for f in files}
        _ROOTS[root] = dict(path=root, files=per,
                            sha256_of_file_hashes=hashlib.sha256("".join(f"{per[f]}  {f}\n" for f in files).encode()).hexdigest())
    return _ROOTS[root]


def provenance(ckpt, root, scripts, extra_roots=()):
    s = file_sha256(ckpt)
    man = _manifests().get(os.path.realpath(ckpt))
    return dict(checkpoint=dict(path=ckpt, realpath=os.path.realpath(ckpt), bytes=os.path.getsize(ckpt), sha256=s,
                                manifest_sha256=man[0] if man else None, manifest_source=man[1] if man else None,
                                matches_manifest=(s == man[0]) if man else None),
                code_root=code_root(root), other_code_roots=[code_root(r) for r in extra_roots],
                scripts={os.path.basename(p): file_sha256(p) for p in scripts})

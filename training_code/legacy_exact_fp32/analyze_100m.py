import csv
import matplotlib.pyplot as plt

RUNS = {
    "Exact": "runs/exact_fp32_softmax_100m/log.csv",
    "Q16": "runs/q16_nearest_100m/log.csv",
    "Q8": "runs/q8_nearest_100m/log.csv",
    "Q4": "runs/q4_nearest_100m/log.csv",
    "Q2": "runs/q2_nearest_100m/log.csv",
}

KS = {"Q16": 16, "Q8": 8, "Q4": 4, "Q2": 2}


def read_log(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {
        "tokens": [int(r["tokens"]) / 1e6 for r in rows],
        "val": [float(r["val_loss"]) for r in rows],
        "gnorm": [float(r["grad_norm"]) for r in rows],
    }


def read_occ(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {
        "tokens": [int(r["tokens"]) / 1e6 for r in rows],
        "span": [float(r["row_span_mean"]) for r in rows],
    }


logs = {name: read_log(path) for name, path in RUNS.items()}
exact = logs["Exact"]

# 1. Paired delta NLL
plt.figure(figsize=(8, 5))
for name in ["Q16", "Q8", "Q4", "Q2"]:
    n = min(len(logs[name]["val"]), len(exact["val"]))
    delta = [
        logs[name]["val"][i] - exact["val"][i]
        for i in range(n)
    ]
    plt.plot(exact["tokens"][:n], delta, marker="o", label=name)

plt.axhline(0, linewidth=1)
plt.xlabel("Training tokens (M)")
plt.ylabel("Val NLL - Exact FP32 Softmax")
plt.legend()
plt.tight_layout()
plt.savefig("delta_nll_100m.png", dpi=200)
plt.close()

# 2. Effective interval width = row_span / K
plt.figure(figsize=(8, 5))
for name, K in KS.items():
    occ = read_occ(f"runs/{name.lower()}_nearest_100m/occupancy.csv")
    width = [x / K for x in occ["span"]]
    plt.plot(occ["tokens"], width, marker="o", label=name)

plt.axhline(20, linestyle="--", linewidth=1, label="width=20")
plt.axhline(100, linestyle="--", linewidth=1, label="width=100")
plt.yscale("log")
plt.xlabel("Training tokens (M)")
plt.ylabel("Mean interval width = row span / K")
plt.legend()
plt.tight_layout()
plt.savefig("interval_width_100m.png", dpi=200)
plt.close()

print("saved:")
print("  delta_nll_100m.png")
print("  interval_width_100m.png")

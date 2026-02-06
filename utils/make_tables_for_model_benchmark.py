import json
from collections import defaultdict

PATH = "results/nsys/model_benchmark_mix_precision_compile.jsonl"

# 指定 model_tag 顺序
MODEL_ORDER = ["small", "medium", "large", "xl", "2.7B"]
model_rank = {m: i for i, m in enumerate(MODEL_ORDER)}

pairs = defaultdict(lambda: {"compiled": None, "eager": None})

with open(PATH, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        key = (r["mode"], r["model_tag"], r["context_len"])
        slot = "compiled" if r.get("use_torch_compile", False) else "eager"
        pairs[key][slot] = r

def fmt_time(r):
    if r is None:
        return "NA"
    if r.get("status") != "OK":
        return "OOM"
    return f"{r['mean_s'] * 1e3:.2f}"

def fmt_mem(r):
    if r is None:
        return "NA"
    if r.get("status") != "OK":
        return "OOM"
    return f"{r['mean_peak_mem'] / 2**20:.1f}"

def fmt_speedup(re, rc):
    if re is None or rc is None:
        return "-"
    if re.get("status") != "OK" or rc.get("status") != "OK":
        return "-"
    if rc["mean_s"] == 0:
        return "-"
    return f"{re['mean_s'] / rc['mean_s']:.2f}x"

def fmt_mem_saving(re, rc):
    # mem saving = 1 - compiled_peak / eager_peak
    if re is None or rc is None:
        return "-"
    if re.get("status") != "OK" or rc.get("status") != "OK":
        return "-"
    if rc["mean_peak_mem"] == 0:
        return "-"
    return f"{100*(1 - rc['mean_peak_mem'] / re['mean_peak_mem']):.2f}%"

# 按 mode 分组
by_mode = defaultdict(list)
for (mode, model_tag, context_len), v in pairs.items():
    by_mode[mode].append((model_tag, context_len, v))

for mode in sorted(by_mode.keys()):
    print(f"\n- mode = {mode}\n")
    print("| model_tag | context_len | eager mean (ms) | compiled mean (ms) | speedup | eager peak (MiB) | compiled peak (MiB) | mem saving |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")

    rows = sorted(
        by_mode[mode],
        key=lambda x: (model_rank.get(x[0], 1e9), x[1])
    )

    for model_tag, context_len, recs in rows:
        re = recs["eager"]
        rc = recs["compiled"]

        print(
            f"| {model_tag} | {context_len} | "
            f"{fmt_time(re)} | {fmt_time(rc)} | {fmt_speedup(re, rc)} | "
            f"{fmt_mem(re)} | {fmt_mem(rc)} | {fmt_mem_saving(re, rc)} |"
        )
import json

INPUT = "results/attn/attn_benchmark_compile.jsonl"
D_MODELS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]

data = {}
with open(INPUT) as f:
    for line in f:
        r = json.loads(line)
        data[(r["d_model"], r["context_len"])] = r

for d in D_MODELS:
    print(f"\n- d_model = {d}\n")
    print("| T | forward (ms) | backward (ms) | mem before (MiB) | peak (MiB) |")
    print("|---|--------------|---------------|------------------|------------|")
    for t in SEQ_LENS:
        r = data.get((d, t))
        if r is None or r["status"] == "OOM":
            print(f"| {t} | OOM | OOM | OOM | OOM |")
        else:
            print(
                f"| {t} | "
                f"{r['forward_time(s)']*1e3:.2f} | "
                f"{r['backward_time(s)']*1e3:.2f} | "
                f"{r['before_backward_mem(M)']:.2f} | "
                f"{r['peak_mem(M)']:.2f} |"
            )
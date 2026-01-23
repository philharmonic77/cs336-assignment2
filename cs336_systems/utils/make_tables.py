from pathlib import Path
import pandas as pd


def fmt_s(mean_s: float, std_s: float) -> str:
    return f"{mean_s:.4f} ± {std_s:.4f}"


def main():
    ROOT = Path(__file__).resolve().parents[1]  
    in_path = ROOT / "results" / "first_benchmark.jsonl"
    out_md = ROOT / "results" / "first_benchmark.md"

    df = pd.read_json(in_path, lines=True)

    # 只保留我们关心的列
    df = df[["model_tag", "mode", "mean_s", "std_s"]]

    # 透视：行 = model_tag，列 = mode，值 = mean/std
    mean_p = df.pivot(index="model_tag", columns="mode", values="mean_s")
    std_p  = df.pivot(index="model_tag", columns="mode", values="std_s")

    # 组装输出表
    rows = []
    for tag in mean_p.index:
        f_mean = float(mean_p.loc[tag, "forward"]) # type: ignore
        f_std  = float(std_p.loc[tag,  "forward"]) # type: ignore

        fb_mean = float(mean_p.loc[tag, "forward_backward"]) # type: ignore
        fb_std  = float(std_p.loc[tag,  "forward_backward"]) # type: ignore

        backward_s = fb_mean - f_mean

        rows.append({
            "model_tag": tag,
            "forward (s)": fmt_s(f_mean, f_std),
            "forward+backward (s)": fmt_s(fb_mean, fb_std),
            "backward est. (s)": f"{backward_s:.4f}",
        })

    out_df = pd.DataFrame(rows).sort_values("model_tag")

    # 输出 Markdown / LaTeX
    out_md.write_text(out_df.to_markdown(index=False), encoding="utf-8")

    print("Wrote:")
    print(f"  {out_md}")


if __name__ == "__main__":
    main()
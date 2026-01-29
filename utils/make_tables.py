from pathlib import Path
import pandas as pd


def fmt_s(mean_s: float, std_s: float) -> str:
    if pd.isna(mean_s) or pd.isna(std_s):
        return ""
    return f"{mean_s:.4f} ± {std_s:.4f}"


def main():
    ROOT = Path(__file__).resolve().parents[1]  

    in_path = ROOT / "results" / "nsys" / "times_bf16.jsonl"
    out_md = ROOT / "results" / "nsys" / "times_bf16.md"

    df = pd.read_json(in_path, lines=True)

    # 只保留我们关心的行和列
    df = df[["context_len", "model_tag", "mode", "mean_s", "std_s"]]

    # 透视：行 = (context_len, model_tag)，列 = mode，值 = mean/std
    mean_p = df.pivot(index=["context_len", "model_tag"], columns="mode", values="mean_s")
    std_p  = df.pivot(index=["context_len", "model_tag"], columns="mode", values="std_s")

    # 组装输出表
    rows = []
    for ctx, tag in mean_p.index:
        f_mean = float(mean_p.loc[(ctx, tag), "forward_only"]) # type: ignore
        f_std  = float(std_p.loc[(ctx, tag),  "forward_only"]) # type: ignore

        fb_mean = float(mean_p.loc[(ctx, tag), "train_step"]) # type: ignore
        fb_std  = float(std_p.loc[(ctx, tag),  "train_step"]) # type: ignore

        rows.append({
            "context_len": ctx,
            "model_tag": tag,
            "forward(infer) /s": fmt_s(f_mean, f_std),
            "train_step /s": fmt_s(fb_mean, fb_std),
        })

    out_df = pd.DataFrame(rows)
    order = ["small", "medium", "large", "xl", "2.7B"]
    out_df["model_tag"] = pd.Categorical(out_df["model_tag"], categories=order, ordered=True)
    out_df = out_df.sort_values(["context_len", "model_tag"])

    # Insert separator rows between different context_len groups
    sep_row = {
        "context_len": "",
        "model_tag": "",
        "forward(infer) /s": "",
        "train_step /s": "",
    }
    rows_with_sep = []
    last_ctx = None
    for _, row in out_df.iterrows():
        if last_ctx is not None and row["context_len"] != last_ctx:
            rows_with_sep.append(sep_row)
        rows_with_sep.append(row.to_dict())
        last_ctx = row["context_len"]
    out_df = pd.DataFrame(rows_with_sep)
    out_df = out_df.fillna("")

    # 输出 Markdown / LaTeX
    out_md.write_text(out_df.to_markdown(index=False), encoding="utf-8")

    print("Wrote:")
    print(f"  {out_md}")


if __name__ == "__main__":
    main()

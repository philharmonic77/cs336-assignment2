import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

FNAME_RE = re.compile(
    r"^(?P<model_tag>.+)_ctx(?P<context_len>\d+)_(?P<mode>forward_only|train_step)_nvtx_sum.*\.csv$"
)

R_FORWARD = ":forward"
R_BACKWARD = ":backward"
R_OPTSTEP = ":optimizer_step"
R_TRAINSTEP = ":measure"


def parse_fname(p: Path) -> Optional[dict]:
    m = FNAME_RE.match(p.name)
    if not m:
        return None
    return {
        "model_tag": m.group("model_tag"),
        "context_len": int(m.group("context_len")),
        "mode": m.group("mode"),
    }


def extract_times_one_csv(csv_path: Path) -> Dict[str, float]:
    """
    从单个 nvtx_sum CSV 中提取 :forward/:backward/:optimizer_step 的 Total Time (ns)，并转成秒。
    返回 dict 只包含找到的 key。
    """
    df = pd.read_csv(csv_path, usecols=["Range", "Total Time (ns)", "Instances"])

    # 聚合（同名 Range 可能出现多次，按 Total Time 求和）
    g = df.groupby("Range", as_index=True)["Total Time (ns)"].sum()
    cnt = df.groupby("Range", as_index=True)["Instances"].sum()

    out: Dict[str, float] = {}
    if R_FORWARD in g.index:
        out["forward_s"] = float(g.loc[R_FORWARD]) * 1e-9 / float(cnt.loc[R_FORWARD])
    if R_BACKWARD in g.index:
        out["backward_s"] = float(g.loc[R_BACKWARD]) * 1e-9 / float(cnt.loc[R_FORWARD])
    if R_OPTSTEP in g.index:
        out["optimizer_step_s"] = float(g.loc[R_OPTSTEP]) * 1e-9 / float(cnt.loc[R_FORWARD])
    if R_TRAINSTEP in g.index:
        out["train_step_s"] = float(g.loc[R_TRAINSTEP]) * 1e-9 / float(cnt.loc[R_FORWARD])
    return out


def fmt(x: Optional[float]) -> str:
    return "" if x is None else f"{x:.4f}"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    stats_dir = repo_root / "results" / "nsys" / "stats"
    out_md = stats_dir / "nvtx_summary.md"

    csvs = sorted(stats_dir.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"No CSVs found under {stats_dir}")

    # key = (model_tag, context_len)
    rows: Dict[tuple, dict] = {}

    for p in csvs:
        meta = parse_fname(p)
        if meta is None:
            continue

        key = (meta["model_tag"], meta["context_len"])
        row = rows.setdefault(
            key,
            {
                "model_tag": meta["model_tag"],
                "context_len": meta["context_len"],
                "forward_infer_s": None,
                "forward_train_s": None,
                "backward_s": None,
                "optimizer_step_s": None,
                "train_step_s": None
            },
        )

        t = extract_times_one_csv(p)

        if meta["mode"] == "forward_only":
            row["forward_infer_s"] = t.get("forward_s")
        else:  # train_step
            row["forward_train_s"] = t.get("forward_s")
            row["backward_s"] = t.get("backward_s")
            row["optimizer_step_s"] = t.get("optimizer_step_s")
            row["train_step_s"] = t.get("train_step_s")


    order = {"small": 0, "medium": 1, "large": 2, "xl": 3, "2.7B": 4}
    sorted_rows = sorted(
        rows.values(),
        key=lambda r: (r["context_len"], order.get(r["model_tag"], 999)),
    )

    md = []
    md.append(
        "| model_tag | context_len | forward(inference)/s | forward(train)/s | backward/s | optimizer_step/s | train_step/s |\n"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for r in sorted_rows:
        md.append(
            f"| {r['model_tag']} | {r['context_len']} | "
            f"{fmt(r['forward_infer_s'])} | {fmt(r['forward_train_s'])} | "
            f"{fmt(r['backward_s'])} | {fmt(r['optimizer_step_s'])} | "
            f"{fmt(r['train_step_s'])} |\n"
        )

    out_md.write_text("".join(md), encoding="utf-8")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
import os
import re
import pandas as pd
import plotly.express as px

# ============================================================
# Config
# ============================================================
BASE_DIR = r"g:\Workspace\Đatn\vidore\result_ColPali"
OURS_AGG = "max"      # "max" or "mean"
ATTN_AGG = "max"      # "max" or "mean"
EXPORT_PNG = os.path.join(BASE_DIR, "r10_by_ratio.png")

FILES = {
    "Traditional": "traditional_summary.csv",
    "Hierarchical": "hierarchical_summary.csv",
    "Attention": "attention_pruning_summary.csv",
    "KMeans": "spherical_kmeans_summary.csv",
    "Random": "random_pruning_summary.csv",
    "Ours": "ours_ablation_summary.csv",
}


def _extract_ratio(method_name: str):
    m = re.search(r"_r(\d+)", str(method_name))
    if not m:
        return None
    return int(m.group(1)) / 100.0


def _load_csv(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    if "method" not in df.columns or "r@10" not in df.columns:
        raise ValueError(f"CSV must contain 'method' and 'r@10': {path}")
    return df


def _prepare_method_series(label, filename):
    df = _load_csv(os.path.join(BASE_DIR, filename))

    if label == "Traditional":
        row = df[df["method"] == "traditional"]
        if row.empty:
            raise ValueError("traditional row not found in traditional_summary.csv")
        return pd.DataFrame(
            {
                "method_group": ["Traditional"],
                "ratio": [1.0],
                "r10": [float(row.iloc[0]["r@10"])],
            }
        )

    work = df.copy()
    work["ratio"] = work["method"].apply(_extract_ratio)
    work = work.dropna(subset=["ratio"]).copy()

    if label == "Ours":
        agg = OURS_AGG
    elif label == "Attention":
        agg = ATTN_AGG
    else:
        agg = "mean"

    if agg == "max":
        out = work.groupby("ratio", as_index=False)["r@10"].max()
    else:
        out = work.groupby("ratio", as_index=False)["r@10"].mean()

    out = out.sort_values("ratio").reset_index(drop=True)
    out = out.rename(columns={"r@10": "r10"})
    out["method_group"] = label
    return out[["method_group", "ratio", "r10"]]


def build_plot_dataframe():
    parts = []

    # Build non-traditional series first to get the ratio grid.
    for label, fn in FILES.items():
        if label == "Traditional":
            continue
        parts.append(_prepare_method_series(label, fn))

    non_trad_df = pd.concat(parts, ignore_index=True)
    ratio_grid = sorted(non_trad_df["ratio"].dropna().unique().tolist())
    if 1.0 not in ratio_grid:
        ratio_grid.append(1.0)

    # Expand Traditional as a horizontal line across all ratios.
    trad_df_raw = _prepare_method_series("Traditional", FILES["Traditional"])
    trad_r10 = float(trad_df_raw.iloc[0]["r10"])
    trad_df = pd.DataFrame(
        {
            "method_group": ["Traditional"] * len(ratio_grid),
            "ratio": ratio_grid,
            "r10": [trad_r10] * len(ratio_grid),
        }
    )

    final_df = pd.concat([non_trad_df, trad_df], ignore_index=True)
    final_df["ratio_pct"] = final_df["ratio"] * 100.0
    return final_df


def make_figure(df_plot, selected_methods):
    sub = df_plot[df_plot["method_group"].isin(selected_methods)].copy()
    sub = sub.sort_values(["method_group", "ratio"])

    fig = px.line(
        sub,
        x="ratio_pct",
        y="r10",
        color="method_group",
        markers=True,
        title="R@10 vs Retrieval Ratio",
        labels={
            "ratio_pct": "Retrieval ratio (%)",
            "r10": "R@10",
            "method_group": "Method",
        },
    )

    fig.update_layout(
        template="plotly_white",
        legend_title_text="Method",
        xaxis=dict(tickmode="array", tickvals=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
    )
    return fig


if __name__ == "__main__":
    df = build_plot_dataframe()

    print("Data points used for plotting:")
    print(df.sort_values(["method_group", "ratio"]).to_string(index=False))

    # Export static PNG only (no HTML).
    fig_all = make_figure(df, list(df["method_group"].drop_duplicates()))
    fig_all.write_image(EXPORT_PNG, width=1400, height=800, scale=2)
    print(f"Saved PNG: {EXPORT_PNG}")

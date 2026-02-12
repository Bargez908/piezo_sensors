#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec, colors


def sort_key(val: str) -> tuple:
    s = str(val).strip()
    if s.lower() == "scale":
        return (1, float("inf"))
    try:
        return (0, float(s))
    except ValueError:
        return (1, float("inf"))


def format_param(val: str) -> str:
    s = str(val).strip()
    if s.lower() == "scale":
        return "scale"
    try:
        f = float(s)
        if np.isclose(f, 0.125):
            return "1/8"
        if np.isclose(f, 0.25):
            return "1/4"
        if np.isclose(f, 0.5):
            return "1/2"
        if f.is_integer():
            return str(int(f))
        return f"{f:g}"
    except ValueError:
        return s


def format_number(val) -> str:
    try:
        f = float(val)
        if f.is_integer():
            return str(int(f))
        return f"{f:g}"
    except Exception:
        return str(val)


def build_alpha_cmap(base_cmap_name: str, base_level: float = 0.85) -> colors.Colormap:
    base = plt.get_cmap(base_cmap_name)
    r, g, b, _ = base(base_level)
    cmap = colors.LinearSegmentedColormap.from_list(
        f"{base_cmap_name}_alpha",
        [(r, g, b, 0.0), (r, g, b, 1.0)],
    )
    cmap.set_bad((r, g, b, 0.0))
    return cmap


def annotate_heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    norm: colors.Normalize,
    fmt: str = "{:.4f}",
    fontsize: int = 6,
) -> None:
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            val = data[y, x]
            if np.isnan(val):
                continue
            text_color = "#ffffff" if norm(val) >= 0.6 else "#111111"
            ax.text(
                x,
                y,
                fmt.format(val),
                ha="center",
                va="center",
                fontsize=fontsize,
                color=text_color,
            )


def default_csv_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "data"
        / "records_final"
        / "thumb_pressure"
        / "csv_results"
        / "AR_SVM.csv"
    )


def load_and_filter(csv_path: Path, fold_mask: str, fold_filter: str | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    fold_mask = str(fold_mask)
    if "fold_mask" in df.columns:
        df = df[df["fold_mask"].astype(str) == fold_mask]
    elif "fold" in df.columns:
        df = df[df["fold"].astype(str) == fold_mask]
    else:
        raise ValueError("CSV must contain 'fold_mask' or 'fold'.")

    if fold_filter is not None and "fold" in df.columns:
        df = df[df["fold"].astype(str) == str(fold_filter)]
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gamma"] = df["gamma"].astype(str)
    df["C"] = df["C"].astype(str)

    for col in ["order", "window_length", "overlap", "accuracy"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["gamma", "C", "order", "window_length", "overlap", "accuracy"])
    df = (
        df.groupby(["gamma", "C", "overlap", "order", "window_length"], as_index=False)[
            "accuracy"
        ]
        .mean()
    )
    return df


def plot_nested_heatmaps(
    df: pd.DataFrame,
    cmap_name: str,
    figscale: float,
    output_path: Path | None,
    show: bool,
) -> None:
    gamma_vals = sorted(df["gamma"].unique(), key=sort_key)
    c_vals = sorted(df["C"].unique(), key=sort_key)
    overlap_vals = sorted(df["overlap"].unique())
    order_vals = sorted(df["order"].unique())
    window_vals = sorted(df["window_length"].unique())

    if not gamma_vals or not c_vals or not overlap_vals:
        raise ValueError("No data after filtering. Check fold_mask and filters.")

    acc_min = float(np.nanmin(df["accuracy"].values))
    acc_max = float(np.nanmax(df["accuracy"].values))
    if np.isclose(acc_min, acc_max):
        acc_min -= 1e-6
        acc_max += 1e-6

    norm = colors.Normalize(vmin=acc_min, vmax=acc_max)
    cmap = build_alpha_cmap(cmap_name)

    sub_w = max(1.0, 0.35 * len(window_vals) + 0.6)
    sub_h = max(1.0, 0.35 * len(order_vals) + 0.6)

    fig_w = (len(c_vals) * len(overlap_vals) * sub_w + 2.0) * figscale
    fig_h = (len(gamma_vals) * sub_h + 2.0) * figscale

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=0.10, right=0.90, top=0.90, bottom=0.08)

    outer = gridspec.GridSpec(len(gamma_vals), len(c_vals), figure=fig, wspace=0.25, hspace=0.35)

    last_im = None

    for i, gamma in enumerate(gamma_vals):
        for j, c_val in enumerate(c_vals):
            cell_spec = outer[i, j]
            inner = gridspec.GridSpecFromSubplotSpec(
                1,
                len(overlap_vals),
                subplot_spec=cell_spec,
                wspace=0.15,
                hspace=0.1,
            )
            for k, overlap in enumerate(overlap_vals):
                ax = fig.add_subplot(inner[0, k])
                sub = df[
                    (df["gamma"] == gamma)
                    & (df["C"] == c_val)
                    & (df["overlap"] == overlap)
                ]
                if sub.empty:
                    data = np.full((len(order_vals), len(window_vals)), np.nan)
                else:
                    pivot = sub.pivot_table(
                        index="order",
                        columns="window_length",
                        values="accuracy",
                        aggfunc="mean",
                    )
                    pivot = pivot.reindex(index=order_vals, columns=window_vals)
                    data = pivot.values

                data_masked = np.ma.masked_invalid(data)
                last_im = ax.imshow(
                    data_masked,
                    cmap=cmap,
                    norm=norm,
                    origin="lower",
                    aspect="auto",
                    interpolation="nearest",
                )
                annotate_heatmap(ax, data, norm)

                ax.set_xticks(np.arange(len(window_vals)))
                ax.set_yticks(np.arange(len(order_vals)))
                ax.set_xticklabels(
                    [format_number(v) for v in window_vals],
                    fontsize=6,
                    rotation=45,
                    ha="right",
                )
                ax.set_yticklabels(
                    [format_number(v) for v in order_vals],
                    fontsize=6,
                )
                ax.tick_params(length=0, pad=1)
                ax.set_title(f"ov {format_number(overlap)}", fontsize=7)
                ax.set_xlabel("length", fontsize=7, labelpad=1)
                if k == 0:
                    ax.set_ylabel("order", fontsize=7, labelpad=1)
                else:
                    ax.set_ylabel("")

                ax.set_xticks(np.arange(-0.5, len(window_vals), 1), minor=True)
                ax.set_yticks(np.arange(-0.5, len(order_vals), 1), minor=True)
                ax.grid(which="minor", color="#4a4a4a", linewidth=0.4)
                ax.tick_params(which="minor", bottom=False, left=False)

    for j, c_val in enumerate(c_vals):
        pos = outer[0, j].get_position(fig)
        x = (pos.x0 + pos.x1) / 2
        fig.text(x, pos.y1 + 0.03, format_param(c_val), ha="center", va="bottom", fontsize=12)

    for i, gamma in enumerate(gamma_vals):
        pos = outer[i, 0].get_position(fig)
        y = (pos.y0 + pos.y1) / 2
        fig.text(pos.x0 - 0.04, y, format_param(gamma), ha="right", va="center", fontsize=12)

    fig.text(0.5, 0.98, "C", ha="center", va="top", fontsize=14)
    fig.text(0.015, 0.5, "Gamma", ha="left", va="center", rotation=90, fontsize=14)

    if last_im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label("Accuracy", rotation=90)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot nested heatmaps for SVM results (gamma rows, C columns)."
    )
    parser.add_argument("--csv", type=Path, default=default_csv_path())
    parser.add_argument(
        "--fold-mask",
        default="11111",
        help="Filter on fold_mask (or fold if fold_mask does not exist).",
    )
    parser.add_argument(
        "--fold",
        default=None,
        help="Optional filter on fold column (e.g. mean).",
    )
    parser.add_argument(
        "--cmap",
        default="Blues",
        help="Base colormap used to pick the alpha-colored hue.",
    )
    parser.add_argument("--figscale", type=float, default=1.0, help="Figure size scale.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="If set, save the plot to this file.",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not show the figure.")
    args = parser.parse_args()

    df = load_and_filter(args.csv, args.fold_mask, args.fold)
    df = prepare_data(df)
    show = not args.no_show
    plot_nested_heatmaps(df, args.cmap, args.figscale, args.output, show)


if __name__ == "__main__":
    main()

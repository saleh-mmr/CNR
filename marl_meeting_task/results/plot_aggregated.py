#!/usr/bin/env python3
"""
Plot aggregated results from an `aggregated.json` file.

Usage:
    python3 plot_aggregated.py /path/to/aggregated.json

This script will:
- Load the JSON file.
- Print a small summary table to the console (including `final_success_rate_mean` if present).
- Produce a PNG with a simple table and plots saved next to the JSON file as `aggregated_plot.png`.

The script handles a few possible JSON shapes (common patterns):
- Top-level key `final_success_rate_mean`: a scalar to display.
- Top-level `seed_results`: list of {"seed": N, "final_success_rate": val} entries.
- Top-level `metrics_over_time`: dict with e.g. `episodes` and `success_rate_mean` lists for plotting.

Assumptions:
- The aggregated JSON contains at least one of the above keys. If not, the script will show available keys and attempt a best-effort table/plot.

"""

import argparse
import json
import os
import sys
from typing import Any, Dict

try:
    import matplotlib.pyplot as plt
    from matplotlib.table import Table
    import numpy as np
except Exception as e:
    print("This script requires matplotlib and numpy. Install them with: pip install matplotlib numpy")
    raise


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"aggregated.json not found: {path}")
    with open(path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"File exists but is not valid JSON: {path}")
    return data


def summarize(data: Dict[str, Any]) -> Dict[str, Any]:
    summary = {}
    # direct top-level final_success_rate_mean
    if "final_success_rate_mean" in data:
        summary["final_success_rate_mean"] = data["final_success_rate_mean"]

    # Handle nested `aggregated` structure (common in run outputs)
    agg = data.get("aggregated") if isinstance(data.get("aggregated"), dict) else None
    if agg is not None:
        # aggregated.final_metrics.final_success_rate_mean
        fm = agg.get("final_metrics") if isinstance(agg.get("final_metrics"), dict) else None
        if fm is not None and "final_success_rate_mean" in fm:
            summary["final_success_rate_mean"] = fm["final_success_rate_mean"]
        # store seeds list if present
        if "seeds" in agg and isinstance(agg["seeds"], list):
            summary["aggregated_seeds"] = agg["seeds"]
        # metrics over time may be nested under aggregated (e.g., episode_rewards)
        mot = {}
        if "episode_rewards" in agg and isinstance(agg["episode_rewards"], dict):
            if "mean" in agg["episode_rewards"]:
                mot["episodes"] = list(range(1, len(agg["episode_rewards"]["mean"]) + 1))
                mot["success_rate_mean"] = agg["episode_rewards"]["mean"]
        if mot:
            summary.setdefault("metrics_over_time", {}).update(mot)

    # seed_results: explicit list under 'seed_results'
    if "seed_results" in data and isinstance(data["seed_results"], list):
        seeds = []
        vals = []
        for item in data["seed_results"]:
            s = item.get("seed") if isinstance(item, dict) else None
            v = None
            if isinstance(item, dict):
                v = item.get("final_success_rate") or item.get("final_success") or item.get("success_rate")
            seeds.append(s)
            vals.append(v)
        summary["seed_results"] = {"seeds": seeds, "values": vals}
        numeric_vals = [v for v in vals if isinstance(v, (int, float))]
        if numeric_vals:
            summary["seed_mean"] = float(np.mean(numeric_vals))

    # per_seed: many aggregated files include a top-level `per_seed` array with per-seed objects
    if "per_seed" in data and isinstance(data["per_seed"], list):
        seeds = []
        vals = []
        for idx, item in enumerate(data["per_seed"]):
            s = item.get("seed") if isinstance(item, dict) and "seed" in item else None
            if s is None and "aggregated" in data and isinstance(data["aggregated"].get("seeds"), list):
                try:
                    s = data["aggregated"]["seeds"][idx]
                except Exception:
                    s = None
            v = None
            if isinstance(item, dict):
                fm = item.get("final_metrics") if isinstance(item.get("final_metrics"), dict) else None
                if fm is not None:
                    v = fm.get("final_success_rate") or fm.get("final_success_rate_mean") or fm.get("final_return_mean")
                if v is None:
                    for cand in ("final_success_rate", "final_success_rate_mean", "final_success", "success_rate"):
                        if cand in item:
                            v = item[cand]
                            break
            seeds.append(s)
            vals.append(v)
        numeric_vals = [x for x in vals if isinstance(x, (int, float))]
        if numeric_vals:
            summary["seed_results"] = {"seeds": seeds, "values": vals}
            summary["seed_mean"] = float(np.mean(numeric_vals))

    # metrics_over_time: top-level key as before
    if "metrics_over_time" in data and isinstance(data["metrics_over_time"], dict):
        summary.setdefault("metrics_over_time", {}).update(data["metrics_over_time"])

    return summary


def print_table(summary: Dict[str, Any]):
    print("\nAggregated Summary:\n")
    if "final_success_rate_mean" in summary:
        print(f"final_success_rate_mean: {summary['final_success_rate_mean']}")
    if "seed_results" in summary:
        seeds = summary["seed_results"]["seeds"]
        vals = summary["seed_results"]["values"]
        print("\nSeed results:")
        print("seed\tfinal_success_rate")
        for s, v in zip(seeds, vals):
            print(f"{s}\t{v}")
        if "seed_mean" in summary:
            print(f"\nseed_mean: {summary['seed_mean']}")
    if "metrics_over_time" in summary:
        mot = summary["metrics_over_time"]
        print("\nMetrics over time keys:", list(mot.keys()))
    if not summary:
        print("No recognized aggregated keys found. Available top-level keys in JSON may be:\n - ")


def plot_and_save(data: Dict[str, Any], summary: Dict[str, Any], json_path: str):
    out_dir = os.path.dirname(json_path)
    out_png = os.path.join(out_dir, "aggregated_plot.png")

    # Create a figure with a table at top and plots below
    fig = plt.figure(constrained_layout=True, figsize=(10, 6))
    gs = fig.add_gridspec(3, 2)

    # Top-left: simple textual table of key metrics
    ax_table = fig.add_subplot(gs[0, :])
    ax_table.axis("off")

    table_data = []
    table_cols = ["metric", "value"]

    # Ensure we always show final_success_rate_mean if present
    if "final_success_rate_mean" in summary:
        table_data.append(["final_success_rate_mean", summary["final_success_rate_mean"]])
    # show seed mean if available
    if "seed_mean" in summary:
        table_data.append(["seed_mean", summary["seed_mean"]])
    # show number of seeds
    if "seed_results" in summary:
        table_data.append(["num_seeds", len(summary["seed_results"]["seeds"])])

    if not table_data:
        # fallback: dump top-level keys and short repr
        for k, v in data.items():
            if isinstance(v, (int, float, str)):
                table_data.append([k, v])
            else:
                table_data.append([k, type(v).__name__])

    # Draw table
    table = ax_table.table(cellText=table_data, colLabels=table_cols, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax_table.set_title("Aggregated results summary", fontsize=12)

    # Bottom-left: bar chart of seed final success rates (if present)
    ax1 = fig.add_subplot(gs[1:, 0])
    if "seed_results" in summary:
        seeds = summary["seed_results"]["seeds"]
        vals = summary["seed_results"]["values"]
        # replace None with nan
        vals_num = [float(v) if isinstance(v, (int, float)) else np.nan for v in vals]
        x = np.arange(len(seeds))
        ax1.bar(x, vals_num, color="tab:blue")
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(s) for s in seeds], rotation=45)
        ax1.set_ylabel("final_success_rate")
        ax1.set_title("Per-seed final success rates")
    else:
        ax1.text(0.5, 0.5, "No per-seed results to show", ha="center", va="center")
        ax1.set_axis_off()

    # Bottom-right: metrics over time plot (if present)
    ax2 = fig.add_subplot(gs[1:, 1])
    if "metrics_over_time" in summary:
        mot = summary["metrics_over_time"]
        # Try to find episodes/x and success_rate_mean/y
        x = None
        if "episodes" in mot:
            x = mot["episodes"]
        elif "steps" in mot:
            x = mot["steps"]
        # y candidates
        y = None
        for candidate in ("success_rate_mean", "final_success_rate_mean", "success_rate", "mean_success_rate"):
            if candidate in mot:
                y = mot[candidate]
                break
        if x is not None and y is not None:
            ax2.plot(x, y, marker="o")
            ax2.set_xlabel("episode")
            ax2.set_ylabel("success_rate_mean")
            ax2.set_title("Success rate over time")
        else:
            ax2.text(0.5, 0.5, "metrics_over_time present but missing expected keys", ha="center", va="center")
            ax2.set_axis_off()
    else:
        ax2.text(0.5, 0.5, "No metrics-over-time to show", ha="center", va="center")
        ax2.set_axis_off()

    plt.savefig(out_png, bbox_inches="tight")
    print(f"Saved plot to: {out_png}")


def main():
    # Default path: the qmix/4 aggregated file in the repository (relative to this script)
    default_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "qmix", "4", "aggregated.json"))

    parser = argparse.ArgumentParser(description="Plot aggregated.json results")
    # Accept an optional positional argument; if not provided, use the default_path
    parser.add_argument("json_path", nargs="?", default=default_path, help="Path to aggregated.json (optional)")
    args = parser.parse_args()

    if not os.path.exists(args.json_path):
        print(f"Error: JSON file not found: {args.json_path}")
        sys.exit(1)

    try:
        data = load_json(args.json_path)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        sys.exit(1)

    summary = summarize(data)
    print_table(summary)

    # Ensure we at least print final_success_rate_mean to console if present
    if "final_success_rate_mean" in summary:
        print(f"\nfinal_success_rate_mean = {summary['final_success_rate_mean']}")

    try:
        plot_and_save(data, summary, args.json_path)
    except Exception as e:
        print(f"Error during plotting: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Weak-scaling analysis of the PM timers found in OpenGadget3 ``cpu.txt`` files.

``cpu.txt`` format (one block per step)::

    Step 2 Time: a=0.016141, MPI-Tasks: 4 Task:0
    Total wall clock time for Global = 116.831 sec
    * Timestep                : 115.3149 sec,  98.70%
    - * PM                    :   3.7211 sec,   3.23%
    - - * PM_2                :   3.4996 sec,  94.05%
    - - - * PM_SEND_PARTICLES :   0.5616 sec,  16.05%

Assumptions:
  * after step 0 (initialisation) the timers are *cumulative*;
  * a timer that is not executed in a step may be missing from that block.

Because the PM solver is not executed at every step, the cumulative value at the
chosen step is divided by the number of steps in which the parent timer
(default: ``PM``) actually increased. The result is the average time per
*PM-active* step, which is what is compared across node counts.

Example::

    python extract_timers.py 512/cpu.txt 1024/cpu.txt 2048/cpu.txt \\
        --title HEFFTE --prefix weak_scaling_heffte --heffte --efficiency
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (cluster login nodes)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PM_TIMERS = [
    "PM", "PM_SEND_PARTICLES", "PM_COMM_RESULTS", "PM_ADD_FORCE",
    "PM_INTERP_FORCE", "PM_FILL_RHO_GRID", "PM_FORCE_GRID", "PM_FFTW_FRWD",
    "PM_FFTW_INV", "PM_MULT_GREENS", "PM_TRANSPOSE_B", "PM_TRANSPOSE_A",
    "PM_PREP",
]
HEFFTE_TIMERS = ["PM_HEFFTE_D2H_FORCES", "PM_HEFFTE_DEVICE_P", "PM_HEFFTE_DEALLOC"]
BASE_SIZE = 512  # 512^3 particles on 1 node; nodes = (size / 512)^3

STEP_RE = re.compile(r"^Step\s+(\d+)\b")
TOTAL_RE = re.compile(r"^Total wall clock time .*=\s*([0-9.]+)")
TIMER_RE = re.compile(r"^\s*(?:-\s+)*\*\s+(\S+)\s*:\s*([0-9.]+)\s+sec")


# ---------------------------------------------------------------- parsing
@dataclass
class StepBlock:
    step: int
    total: float | None = None
    timers: dict[str, float] = field(default_factory=dict)


def parse_cpu_file(path: str | Path) -> list[StepBlock]:
    """Read every ``Step N`` block. For repeated names the first occurrence wins."""
    blocks: list[StepBlock] = []
    current: StepBlock | None = None
    with open(path) as f:
        for line in f:
            m = STEP_RE.match(line)
            if m:
                current = StepBlock(int(m.group(1)))
                blocks.append(current)
                continue
            if current is None:
                continue
            m = TOTAL_RE.match(line)
            if m:
                current.total = float(m.group(1))
                continue
            m = TIMER_RE.match(line)
            if m:
                current.timers.setdefault(m.group(1), float(m.group(2)))
    return blocks


def count_active_steps(blocks: list[StepBlock], parent: str, target_step: int) -> int:
    """Number of steps in 1..target_step where the cumulative ``parent`` timer grew.

    Step 0 is skipped: it holds the initialisation and the timers restart after it.
    """
    previous = 0.0
    active = 0
    for block in blocks:
        if block.step == 0 or block.step > target_step:
            continue
        value = block.timers.get(parent)
        if value is None:
            continue
        if value > previous:
            active += 1
        previous = value
    return active


def load_run(path: str | Path, names: list[str], parent: str, step: int | None = None) -> dict[str, float]:
    """Average time per parent-active step for each timer in ``names``."""
    blocks = parse_cpu_file(path)
    if not blocks:
        raise ValueError(f"{path}: no 'Step N' block found")
    target = step if step is not None else blocks[-1].step
    block = next((b for b in blocks if b.step == target), None)
    if block is None:
        raise ValueError(f"{path}: step {target} not found (last step: {blocks[-1].step})")
    active = count_active_steps(blocks, parent, target)
    if active == 0:
        raise ValueError(f"{path}: timer '{parent}' never increased in steps 1..{target}")
    missing = [k for k in names if k not in block.timers]
    if missing:
        print(f"warning: {path}: timers not found (set to 0): {', '.join(missing)}",
              file=sys.stderr)
    print(f"{path}: step {target}, '{parent}' active in {active}/{target} steps")
    return {k: block.timers.get(k, 0.0) / active for k in names}


# ---------------------------------------------------------------- tables
def build_frame(runs: list[dict[str, float]], columns: list[str],
                names: list[str]) -> pd.DataFrame:
    """Rows = timers, columns = runs (seconds per parent-active step)."""
    return pd.DataFrame({c: [r[k] for k in names] for c, r in zip(columns, runs)},
                        index=names)


def _fmt(x: float, spec: str = "{:.4f}") -> str:
    return spec.format(x) if np.isfinite(x) else "n/a"


def format_table(df: pd.DataFrame, parent: str,
                 ref: pd.DataFrame | None = None) -> pd.DataFrame:
    """Text table.

    Without ``ref``: values normalised to the parent time of the first run, plus
    the growth factor with respect to the first run, e.g. ``0.5200 (x1.30)``.
    With ``ref``: ratio to the same timer of the reference run (same column).
    """
    cells = pd.DataFrame(index=df.index, columns=df.columns, dtype=object)
    with np.errstate(divide="ignore", invalid="ignore"):
        if ref is not None:
            ratio = df.to_numpy() / ref.to_numpy()
            for j, col in enumerate(df.columns):
                cells[col] = [_fmt(ratio[i, j]) for i in range(len(df))]
        else:
            norm = df / df.loc[parent].iloc[0]
            growth = df.div(df.iloc[:, 0], axis=0)
            for j, col in enumerate(df.columns):
                col_cells = []
                for k in df.index:
                    s = _fmt(norm.loc[k, col])
                    if j > 0:
                        s += f" (x{_fmt(growth.loc[k, col], '{:.2f}')})"
                    col_cells.append(s)
                cells[col] = col_cells
    cells.index.name = "Block"
    return cells


def save_table_image(cells: pd.DataFrame, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 0.55 * len(cells) + 2))
    ax.axis("off")
    text = [[k, *row] for k, row in zip(cells.index, cells.to_numpy())]
    col_labels = ["Block", *cells.columns]
    table = ax.table(cellText=text, colLabels=col_labels, cellLoc="center",
                     loc="center", bbox=[0.05, 0.05, 0.9, 0.85])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for j in range(len(col_labels)):
        cell = table[(0, j)]
        cell.set_facecolor("#4CAF50")
        cell.set_text_props(weight="bold", color="white", fontsize=11)
    for i in range(1, len(text) + 1):
        for j in range(len(col_labels)):
            cell = table[(i, j)]
            cell.set_facecolor("#f0f0f0" if i % 2 == 0 else "#ffffff")
            if j == 0:
                cell.set_text_props(weight="bold", ha="left", fontsize=9)
    plt.title(title, fontsize=13, weight="bold", pad=15)
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


# ---------------------------------------------------------------- plots
def plot_breakdown(df: pd.DataFrame, parent: str, nodes: list[int],
                   title: str, out: Path) -> None:
    """Stacked bars of the sub-timers; the parent is a wider bar behind them."""
    norm = df / df.loc[parent].iloc[0]
    colors = {k: plt.cm.tab20(i % 20) for i, k in enumerate(df.index)}
    x = np.arange(len(df.columns))
    bottom = np.zeros(len(x))
    fig, ax = plt.subplots(figsize=(12, 8))
    for k in df.index:
        values = norm.loc[k].to_numpy()
        if k == parent:
            ax.bar(x, values, 0.7, label=k, color=colors[k])
        else:
            ax.bar(x, values, 0.6, bottom=bottom, label=k, color=colors[k])
            bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in nodes])
    ax.set_xlabel("Nodes", fontsize=18)
    ax.set_ylabel(f"T(N) / T(1)  [{parent}]", fontsize=18)
    ax.set_title(title, fontsize=18)
    ax.tick_params(axis="both", labelsize=18)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=14)
    ax.grid(True, linestyle="dotted")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_efficiency(series: dict[str, np.ndarray], nodes: list[int], out: Path) -> None:
    """Weak-scaling efficiency: 100 * T(first run) / T(N)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    x = np.arange(len(nodes))
    for label, times in series.items():
        ax.plot(x, 100 * times[0] / times, marker="o", label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in nodes])
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Parallel efficiency (%)")
    ax.set_title("Weak-scaling efficiency: T(1 node) / T(N nodes)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=14)
    ax.grid(True, linestyle="dotted")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- CLI
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PM timers weak-scaling plot/table from OpenGadget3 cpu.txt files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("files", nargs="+", help="cpu.txt files, in increasing size order")
    p.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048],
                   help="grid/particle size of each file")
    p.add_argument("--nodes", type=int, nargs="+",
                   help=f"nodes of each file (default: (size/{BASE_SIZE})^3)")
    p.add_argument("--step", type=int,
                   help="step to read (default: last step in each file)")
    p.add_argument("--timers", nargs="+",
                   help="timers to extract; the first is the parent used for "
                        "normalisation (default: PM breakdown)")
    p.add_argument("--heffte", action="store_true", help="also extract PM_HEFFTE_* timers")
    p.add_argument("--title", default="", help="plot title")
    p.add_argument("--prefix", default="weak_scaling", help="output file prefix")
    p.add_argument("--outdir", default=".", help="output directory")
    p.add_argument("--ratio-to", nargs="+", metavar="FILE",
                   help="reference cpu.txt files (same order); the table shows "
                        "ratios to them")
    p.add_argument("--ref-label", default="reference", help="label of the reference runs")
    p.add_argument("--efficiency", action="store_true",
                   help="also plot the parallel efficiency")
    p.add_argument("--no-table", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)

    if len(args.files) != len(args.sizes):
        p.error(f"{len(args.files)} files but {len(args.sizes)} sizes: use --sizes")
    if args.ratio_to and len(args.ratio_to) != len(args.files):
        p.error("--ratio-to needs one file per input file")
    if args.nodes is None:
        args.nodes = [round((s / BASE_SIZE) ** 3) for s in args.sizes]
        if min(args.nodes) < 1:
            p.error("cannot infer nodes from these sizes: use --nodes")
    elif len(args.nodes) != len(args.files):
        p.error("--nodes needs one value per input file")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    names = args.timers or (PM_TIMERS + (HEFFTE_TIMERS if args.heffte else []))
    parent = names[0]
    columns = [f"{s}^3 ({n} node{'s' if n > 1 else ''})"
               for s, n in zip(args.sizes, args.nodes)]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        runs = [load_run(f, names, parent, args.step) for f in args.files]
        refs = ([load_run(f, names, parent, args.step) for f in args.ratio_to]
                if args.ratio_to else None)
    except (ValueError, OSError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    df = build_frame(runs, columns, names)
    ref_df = build_frame(refs, columns, names) if refs else None

    if not args.no_table:
        cells = format_table(df, parent, ref_df)
        print("\n" + cells.to_string() + "\n")
        if ref_df is not None:
            subtitle = f"(ratio to the same timer of the {args.ref_label} run)"
        else:
            subtitle = (f"(normalised to the {parent} time of the first run = "
                        f"{df.loc[parent].iloc[0]:.4f} s per active step)")
        table_path = outdir / f"{args.prefix}_table.png"
        save_table_image(cells, f"{args.title} {parent} timers\n{subtitle}", table_path)
        print(f"Table saved to {table_path}")

    if not args.no_plot:
        plot_path = outdir / f"{args.prefix}.png"
        plot_breakdown(df, parent, args.nodes, args.title, plot_path)
        print(f"Plot saved to {plot_path}")

    if args.efficiency:
        series = {args.title or "run": df.loc[parent].to_numpy()}
        if ref_df is not None:
            series[args.ref_label] = ref_df.loc[parent].to_numpy()
        eff_path = outdir / f"{args.prefix}_efficiency.png"
        plot_efficiency(series, args.nodes, eff_path)
        print(f"Efficiency plot saved to {eff_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Turn mbench.py's jsonl into markdown tables (median over seeds)."""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

KMED_ORDER = ("pypi-fasterpam-1cpu", "pypi-fasterpam-allcpu", "local-fasterpam", "local-hybrid")
MEDOID_ORDER = ("sklearn-rowsum", "local-brute", "local-cpv")
Row = dict[str, Any]


def load(path: Path) -> list[Row]:
    with path.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return [r for r in rows if r.get("part") != "env"]


def median(rows: list[Row], field: str) -> float:
    return statistics.median(r[field] for r in rows)


def fmt(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    return f"{seconds:.3f}" if seconds < 1 else f"{seconds:.1f}"


def kmed_table(rows: list[Row]) -> list[str]:
    ok = [r for r in rows if r["part"] == "kmed" and "error" not in r]
    matrix = {(r["data"], r["d"], r["n"]): r for r in ok if r["method"] == "pypi-matrix"}
    groups: dict[tuple[Any, ...], dict[str, list[Row]]] = defaultdict(lambda: defaultdict(list))
    for r in ok:
        if r["method"] != "pypi-matrix":
            groups[(r["data"], r["d"], r["n"], r["k"])][r["method"]].append(r)
    head = ("| data | n | k | matrix s (GB) | pypi 1cpu s | pypi all s | local FasterPAM s | "
            "hybrid s | hybrid vs pypi total | FasterPAM loss diff | hybrid loss diff | seeds |")
    lines = [head, "|" + "---|" * 12]
    for key in sorted(groups, key=lambda g: (g[0], g[1], g[3], g[2])):
        data, d, n, k = key
        by = groups[key]
        wall = {m: median(by[m], "wall") if by[m] else None for m in KMED_ORDER}
        loss = {m: median(by[m], "loss") if by[m] else None for m in KMED_ORDER}
        mat = matrix.get((data, d, n))
        ref_name = "pypi-fasterpam-1cpu" if loss["pypi-fasterpam-1cpu"] else "local-fasterpam"
        ref = loss[ref_name]

        def diff(name: str, ref: float | None = ref, loss: dict = loss) -> str:
            if ref is None or loss[name] is None:
                return "-"
            return f"{100 * (loss[name] / ref - 1):+.2f}%"

        total = None
        if mat and wall["pypi-fasterpam-allcpu"] is not None:
            total = mat["wall"] + min(wall["pypi-fasterpam-allcpu"], wall["pypi-fasterpam-1cpu"])
        ratio = f"{total / wall['local-hybrid']:.0f}x" if total and wall["local-hybrid"] else "-"
        mat_txt = f"{fmt(mat['wall'])} ({mat['gb']:.1f})" if mat else "-"
        lines.append(
            f"| {data} d={d} | {n:,} | {k} | {mat_txt} | {fmt(wall['pypi-fasterpam-1cpu'])} | "
            f"{fmt(wall['pypi-fasterpam-allcpu'])} | {fmt(wall['local-fasterpam'])} | "
            f"{fmt(wall['local-hybrid'])} | {ratio} | {diff('local-fasterpam')} | "
            f"{diff('local-hybrid')} | {max(len(v) for v in by.values())} |")
    return lines


def clara_table(rows: list[Row]) -> list[str]:
    """Hybrid against CLARA built on the PyPI package, paired by (config, seed)."""
    ok = [r for r in rows if r["part"] == "kmed" and "error" not in r and "seed" in r]
    runs: dict[tuple[Any, ...], dict[str, Row]] = defaultdict(dict)
    for r in ok:
        runs[(r["data"], r["d"], r["n"], r["k"], r["seed"])][r["method"]] = r
    groups: dict[tuple[Any, ...], list[dict[str, float]]] = defaultdict(list)
    for (data, d, n, k, _seed), by in runs.items():
        if not {"pypi-clara-1cpu", "pypi-clara-allcpu", "local-hybrid"} <= set(by):
            continue
        base = min((by["pypi-clara-1cpu"], by["pypi-clara-allcpu"]), key=lambda r: r["wall"])
        hyb = by["local-hybrid"]
        groups[(data, d, n, k)].append({
            "base": base["wall"], "hybrid": hyb["wall"], "clara": hyb["wall_clara"],
            "gain": 100 * (hyb["loss"] / base["loss"] - 1),
            "same_stage": 100 * (hyb["loss_clara"] / base["loss"] - 1)})
    lines = ["| data | n | k | pypi CLARA s | local CLARA stage s | hybrid s | "
             "hybrid loss vs pypi CLARA (min / median / max) | local CLARA loss vs pypi CLARA "
             "| seeds |", "|" + "---|" * 9]
    for key in sorted(groups, key=lambda g: (g[0], g[1], g[3], g[2])):
        data, d, n, k = key
        g = groups[key]
        gains = sorted(x["gain"] for x in g)
        lines.append(
            f"| {data} d={d} | {n:,} | {k} | {fmt(statistics.median(x['base'] for x in g))} | "
            f"{fmt(statistics.median(x['clara'] for x in g))} | "
            f"{fmt(statistics.median(x['hybrid'] for x in g))} | "
            f"{gains[0]:+.2f}% / {statistics.median(gains):+.2f}% / {gains[-1]:+.2f}% | "
            f"{statistics.median(x['same_stage'] for x in g):+.2f}% | {len(g)} |")
    return lines


def medoid_table(rows: list[Row]) -> list[str]:
    ok = [r for r in rows if r["part"] == "medoid" and "error" not in r]
    groups: dict[tuple[Any, ...], dict[str, Row]] = defaultdict(dict)
    for r in ok:
        groups[(r["data"], r["d"], r["n"])][r["method"]] = r
    lines = ["| data | n | sklearn row-sum s | local brute s | local CPV s | CPV vs sklearn | "
             "same medoid | CPV cpu/wall |", "|" + "---|" * 8]
    for key in sorted(groups, key=lambda g: (g[0], g[1], g[2])):
        data, d, n = key
        by = groups[key]
        wall = {m: by[m]["wall"] if m in by else None for m in MEDOID_ORDER}
        found = {by[m]["medoid"] for m in by}
        cpv = by.get("local-cpv")
        ratio = (f"{wall['sklearn-rowsum'] / wall['local-cpv']:.0f}x"
                 if wall["sklearn-rowsum"] and wall["local-cpv"] else "-")
        same = "-" if len(by) < 2 else ("yes" if len(found) == 1 else f"NO {sorted(found)}")
        par = f"{cpv['cpu'] / cpv['wall']:.1f}" if cpv else "-"
        lines.append(f"| {data} d={d} | {n:,} | {fmt(wall['sklearn-rowsum'])} | "
                     f"{fmt(wall['local-brute'])} | {fmt(wall['local-cpv'])} | {ratio} | "
                     f"{same} | {par} |")
    return lines


def errors(rows: list[Row]) -> list[str]:
    return [f"- {r['key']}: {r['error']}" for r in rows if "error" in r]


def main() -> None:
    rows = load(Path(sys.argv[1]))
    out = ["## k-medoids", *kmed_table(rows), "", "## hybrid vs sampling with the PyPI package",
           *clara_table(rows), "", "## exact medoid", *medoid_table(rows), "",
           "## errors", *(errors(rows) or ["none"])]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

"""Snapshot selection for the HTML demo (no torch)."""

from __future__ import annotations

import re
from pathlib import Path

LEAD_RE = re.compile(r"^(?P<pid>\d+)_lead_(?P<lead>.+)$")


def parse_ludb_name(fname: str) -> tuple[int | None, str]:
    stem = Path(fname).stem
    match = LEAD_RE.match(stem)
    if not match:
        return None, stem
    return int(match.group("pid")), match.group("lead")


def select_records(records: list[dict], keep: int) -> list[dict]:
    if not records:
        return []
    chosen: list[int] = []

    def add(index: int) -> None:
        if index not in chosen:
            chosen.append(index)

    kept = [i for i, rec in enumerate(records) if rec["sample_keep"]]
    rejected = [i for i, rec in enumerate(records) if not rec["sample_keep"]]
    pool = kept or list(range(len(records)))

    t_onset = max(pool, key=lambda i: records[i]["t_score"])
    add(t_onset)
    records[t_onset]["role"] = "t_onset"

    if rejected:
        gate_i = min(rejected, key=lambda i: records[i]["mean_conf"])
        records[gate_i]["role"] = "rejected"
    else:
        gate_i = min(range(len(records)), key=lambda i: records[i]["mean_conf"])
        records[gate_i]["role"] = "closest_to_gate"
    add(gate_i)

    disagree = max(pool, key=lambda i: records[i]["mean_ce_wb"])
    if disagree not in chosen:
        records[disagree]["role"] = "boundary_disagreement"
        add(disagree)

    confs = sorted(pool, key=lambda i: records[i]["mean_conf"])
    typical = confs[len(confs) // 2]
    if typical not in chosen:
        records[typical]["role"] = "typical"
        add(typical)

    remaining = [i for i in pool if i not in chosen]
    remaining.sort(key=lambda i: records[i]["mean_conf"])
    stride = max(1, len(remaining) // max(1, keep - len(chosen)))
    for index in remaining[::stride]:
        if len(chosen) >= keep:
            break
        records[index]["role"] = records[index].get("role") or "diverse"
        add(index)

    ordered = [t_onset] + [i for i in chosen if i != t_onset]
    return [records[i] for i in ordered[:keep]]

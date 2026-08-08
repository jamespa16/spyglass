from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional

Status = Literal[
    "written",
    "planned",
    "skipped_unsupported_ndim",
    "skipped_filtered",
    "skipped_dequant_unsupported",
    "skipped_exists",
    "error",
]


@dataclass
class TensorOutcome:
    name: str
    ggml_type: str
    shape: Optional[tuple[int, ...]]
    status: Status
    reason: Optional[str] = None
    output_path: Optional[str] = None
    stats: Optional[dict] = None


@dataclass
class RunReport:
    input_path: str
    output_dir: str
    clip_value: float
    clip_source: Literal["manual", "auto_sampled"]
    outcomes: list[TensorOutcome] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for outcome in self.outcomes:
            counts[outcome.status] = counts.get(outcome.status, 0) + 1
        return counts


def write_manifest(report: RunReport, path: Path) -> None:
    payload = {
        "input_path": report.input_path,
        "output_dir": report.output_dir,
        "clip_value": report.clip_value,
        "clip_source": report.clip_source,
        "outcomes": [asdict(o) for o in report.outcomes],
    }
    path.write_text(json.dumps(payload, indent=2))


def print_summary(report: RunReport, *, print_fn=print) -> None:
    counts = report.counts()
    print_fn(f"clip value: {report.clip_value:.6g} (source: {report.clip_source})")
    print_fn(f"tensors written: {counts.get('written', 0)}")
    for status, n in sorted(counts.items()):
        if status != "written":
            print_fn(f"  {status}: {n}")

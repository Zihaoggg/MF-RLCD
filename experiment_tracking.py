from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

@dataclass
class MetricSnapshot:
    E: Optional[float]
    sigma_y: Optional[float]
    Kt: Optional[float]
    reward: float
    success: bool
    E_error: Optional[float]
    sigma_y_error: Optional[float]
    Kt_error: Optional[float]

def build_metric_snapshot(target, E, sigma_y, Kt, reward_fn) -> MetricSnapshot:
    if E is None or sigma_y is None or Kt is None:
        return MetricSnapshot(
            E=E,
            sigma_y=sigma_y,
            Kt=Kt,
            reward=-5.0,
            success=False,
            E_error=None,
            sigma_y_error=None,
            Kt_error=None,
        )

    E_error = abs(float(E) - float(target.E_target)) / float(target.E_target)
    sigma_y_error = abs(float(sigma_y) - float(target.sigma_y_target)) / float(target.sigma_y_target)
    denom = abs(float(target.Kt_target)) if abs(float(target.Kt_target)) > 1e-12 else 1.0
    Kt_error = abs(float(Kt) - float(target.Kt_target)) / denom
    success = (
        E_error < float(target.tolerance["E"])
        and sigma_y_error < float(target.tolerance["sigma_y"])
        and Kt_error < float(target.tolerance["Kt"])
    )
    reward = float(reward_fn(E, sigma_y, Kt))
    return MetricSnapshot(
        E=float(E),
        sigma_y=float(sigma_y),
        Kt=float(Kt),
        reward=reward,
        success=bool(success),
        E_error=float(E_error),
        sigma_y_error=float(sigma_y_error),
        Kt_error=float(Kt_error),
    )

def format_metric_delta(label: str, value: Optional[float], target: float, ratio: Optional[float], scale: float = 1.0) -> str:
    if value is None or ratio is None:
        return f"{label}=None"
    diff = (float(value) - float(target)) / float(scale)
    return f"{label}={value/scale:.4f} (target={target/scale:.4f}, diff={diff:+.4f}, err={ratio*100:.2f}%)"

class ExperimentTracker:
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.records_dir = self.root_dir / "records"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.records_dir / "events.jsonl"
        self.evaluations_path = self.records_dir / "evaluations.csv"
        self.episode_path = self.records_dir / "episode_summary.csv"
        self.diagnostics_path = self.records_dir / "mfrlcd_diagnostics.csv"
        self.updates_path = self.records_dir / "mfrlcd_updates.csv"
        self.run_summary_path = self.root_dir / "run_summary.json"
        self._csv_headers_written = set()

    def log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event_type": event_type,
            **payload,
        }
        with self.events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def append_csv(self, path: Path, row: Dict[str, Any]) -> None:
        fieldnames = list(row.keys())
        write_header = (path not in self._csv_headers_written) and (not path.exists() or path.stat().st_size == 0)
        with path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
                self._csv_headers_written.add(path)
            writer.writerow(row)

    def log_evaluation(self, row: Dict[str, Any]) -> None:
        self.append_csv(self.evaluations_path, row)
        self.log_event("evaluation", row)

    def log_episode(self, row: Dict[str, Any]) -> None:
        self.append_csv(self.episode_path, row)
        self.log_event("episode_summary", row)

    def log_diagnostic(self, row: Dict[str, Any]) -> None:
        self.append_csv(self.diagnostics_path, row)
        self.log_event("mfrlcd_diagnostic", row)

    def log_update(self, row: Dict[str, Any]) -> None:
        self.append_csv(self.updates_path, row)
        self.log_event("mfrlcd_update", row)

    def write_summary(self, summary: Dict[str, Any]) -> None:
        with self.run_summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)

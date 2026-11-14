from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np

from mfrlcd.parameters import decode_design_vector
from mfrlcd.records import write_json


SUMMARY_FIELDS = (
    "run", "status", "best_success", "target_E", "target_sigma_y", "target_Kt",
    "best_E", "best_sigma_y", "best_Kt", "best_reward", "num_evaluations",
    "num_backend_evaluations", "num_cache_hits",
)


def read_summary(run_dir):
    with (Path(run_dir) / "run_summary.json").open(encoding="utf-8") as source:
        summary = json.load(source)
    if not isinstance(summary, dict) or not isinstance(summary.get("target"), dict):
        raise ValueError(f"Invalid run summary in {run_dir}")
    return summary


def export_design(run_dir, output_dir):
    run_dir = Path(run_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    summary = read_summary(run_dir)
    if summary.get("status") != "completed":
        raise ValueError("Only completed designs can be exported")
    for name in ("best_E", "best_sigma_y", "best_Kt"):
        value = summary.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid predicted property: {name}")
    vector = np.load(run_dir / "best_params.npy", allow_pickle=False)
    params = decode_design_vector(vector)
    angles = params.angles.reshape(3, params.n_priorBeta).T
    seeds = params.seeds.reshape(3, params.n_priorBeta).T.astype(int)
    geometry = summary.get("best_geometry_dir")
    geometry_dir = (run_dir / geometry).resolve() if geometry else None
    if geometry_dir is not None and not geometry_dir.is_relative_to(run_dir):
        raise ValueError("Geometry directory must be inside the run directory")
    if geometry_dir is not None and not geometry_dir.is_dir():
        raise ValueError("Recorded geometry is missing; restore the run's evaluation directory")
    document = {
        "schema_version": 1,
        "target": summary["target"],
        "prediction": {"E_MPa": summary["best_E"], "sigma_y_MPa": summary["best_sigma_y"], "Kt": summary["best_Kt"]},
        "meets_tolerance": bool(summary.get("best_success", False)),
        "n_priorBeta": params.n_priorBeta,
        "lam_ratio": params.lam_ratio,
        "euler_angles_deg": angles.tolist(),
        "requested_seed_grid_coordinates": seeds.tolist(),
        "requested_seed_positions": (seeds * 0.03125).tolist(),
        "geometry_seed_positions": None,
        "geometry_files": [],
        "prediction_source": "pretrained_surrogates",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if geometry_dir is not None:
        files = sorted(path for path in geometry_dir.iterdir() if path.name == "seeds" or path.suffix in {".tess", ".tesr", ".stgroup"})
        for path in files:
            if not path.is_file() or not path.resolve().is_relative_to(geometry_dir):
                raise ValueError(f"Invalid geometry file: {path.name}")
        seed_file = geometry_dir / "seeds"
        if seed_file.is_file():
            positions = np.loadtxt(seed_file, ndmin=2)
            if positions.shape != (params.n_priorBeta, 3) or not np.isfinite(positions).all():
                raise ValueError("Invalid realized geometry seed coordinates")
            document["geometry_seed_positions"] = positions.tolist()
        for path in files:
            destination = output_dir / "geometry" / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.resolve() != path.resolve():
                shutil.copyfile(path, destination)
            document["geometry_files"].append(str(destination.relative_to(output_dir)))
    columns = ("grain", "phi1_deg", "Phi_deg", "phi2_deg", "requested_seed_x", "requested_seed_y", "requested_seed_z")
    with (output_dir / "grains.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(columns)
        for index, (orientation, seed) in enumerate(zip(angles, seeds), start=1):
            writer.writerow([index, *orientation.tolist(), *seed.tolist()])
    write_json(output_dir / "design.json", document)
    return document


def summarize_runs(input_root, output_path):
    input_root = Path(input_root).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not input_root.is_dir():
        raise ValueError(f"Run directory does not exist: {input_root}")
    rows = []
    for path in sorted(input_root.rglob("run_summary.json")):
        summary = read_summary(path.parent)
        target = summary["target"]
        row = {name: summary.get(name) for name in SUMMARY_FIELDS}
        row.update(run=str(path.parent.relative_to(input_root)), target_E=target.get("E"),
                   target_sigma_y=target.get("sigma_y"), target_Kt=target.get("Kt"))
        rows.append(row)
    if not rows:
        raise ValueError(f"No run summaries found under {input_root}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv=None, *, command="export"):
    parser = argparse.ArgumentParser(description="Export design parameters" if command == "export" else "Collect design run summaries")
    if command == "export":
        parser.add_argument("--run_dir", required=True)
        parser.add_argument("--output_dir", required=True)
    else:
        parser.add_argument("--input_root", required=True)
        parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if command == "export":
            result = export_design(args.run_dir, args.output_dir)
            print(f"Exported {result['n_priorBeta']} beta-grain parameters to {args.output_dir}")
        else:
            rows = summarize_runs(args.input_root, args.output)
            print(f"Saved {len(rows)} design summaries to {args.output}")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0

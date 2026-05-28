from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None

def _augment_target_summary(summary: pd.DataFrame, diagnostics: pd.DataFrame | None) -> pd.DataFrame:
    out = summary.copy()
    if diagnostics is None or diagnostics.empty:
        return out

    diag = diagnostics.copy()
    grouped = (
        diag.groupby(["difficulty", "target_name"])
        .agg(
            ppo_updates=("ppo_update_applied", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
            diffusion_updates=("diffusion_update_applied", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
            first_success_step_diag=("step_best_success", lambda s: int(diag.loc[s.index][pd.Series(s).fillna(False).astype(bool)]["step"].iloc[0]) if pd.Series(s).fillna(False).astype(bool).any() else pd.NA),
            global_best_step=("global_best_reward", lambda s: int(diag.loc[s.index, "step"].iloc[pd.Series(s).astype(float).argmax()])),
        )
        .reset_index()
    )
    out = out.merge(grouped, on=["difficulty", "target_name"], how="left")
    return out

def collect_variant(variant_dir: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    analysis_dir = variant_dir / "analysis"
    target_summary = _read_csv_if_exists(analysis_dir / "paper_target_summary.csv")
    if target_summary is None or target_summary.empty:
        return None, None

    diagnostics = _read_csv_if_exists(analysis_dir / "paper_diagnostics.csv")
    target_summary = _augment_target_summary(target_summary, diagnostics)
    target_summary.insert(0, "variant", variant_dir.name)

    diff_summary = (
        target_summary.groupby(["variant", "difficulty"])
        .agg(
            num_targets=("target_name", "count"),
            success_rate=("success", "mean"),
            avg_best_reward=("best_reward", "mean"),
            avg_best_error_sum=("best_error_sum", "mean"),
            avg_first_success_eval=("first_success_eval", "mean"),
            avg_n_evals=("n_evals", "mean"),
            avg_ppo_updates=("ppo_updates", "mean"),
            avg_diffusion_updates=("diffusion_updates", "mean"),
            avg_duration_sec=("duration_sec", "mean"),
        )
        .reset_index()
    )
    return target_summary, diff_summary

def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate MF-RLCD ablation outputs")
    parser.add_argument("--input_root", required=True, help="Root directory that contains ablation variants")
    parser.add_argument("--output_dir", required=True, help="Directory to write combined ablation summaries")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_frames: list[pd.DataFrame] = []
    diff_frames: list[pd.DataFrame] = []

    for variant_dir in sorted(p for p in input_root.iterdir() if p.is_dir() and (p / "analysis").exists()):
        target_summary, diff_summary = collect_variant(variant_dir)
        if target_summary is not None:
            target_frames.append(target_summary)
        if diff_summary is not None:
            diff_frames.append(diff_summary)

    if target_frames:
        combined_target = pd.concat(target_frames, ignore_index=True)
    else:
        combined_target = pd.DataFrame()

    if diff_frames:
        combined_diff = pd.concat(diff_frames, ignore_index=True)
    else:
        combined_diff = pd.DataFrame()

    if not combined_target.empty:
        variant_summary = (
            combined_target.groupby("variant")
            .agg(
                num_targets=("target_name", "count"),
                success_rate=("success", "mean"),
                avg_best_reward=("best_reward", "mean"),
                avg_best_error_sum=("best_error_sum", "mean"),
                avg_first_success_eval=("first_success_eval", "mean"),
                avg_n_evals=("n_evals", "mean"),
                avg_ppo_updates=("ppo_updates", "mean"),
                avg_diffusion_updates=("diffusion_updates", "mean"),
                avg_duration_sec=("duration_sec", "mean"),
            )
            .reset_index()
        )
    else:
        variant_summary = pd.DataFrame()

    target_path = output_dir / "ablation_target_summary.csv"
    diff_path = output_dir / "ablation_difficulty_summary.csv"
    variant_path = output_dir / "ablation_variant_summary.csv"

    combined_target.to_csv(target_path, index=False, encoding="utf-8-sig")
    combined_diff.to_csv(diff_path, index=False, encoding="utf-8-sig")
    variant_summary.to_csv(variant_path, index=False, encoding="utf-8-sig")

    summary = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "num_variants": int(combined_target["variant"].nunique()) if not combined_target.empty else 0,
        "num_targets": int(len(combined_target)),
        "generated_files": [
            str(target_path),
            str(diff_path),
            str(variant_path),
        ],
    }
    (output_dir / "ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[AblationAnalysis] variants={summary['num_variants']} targets={summary['num_targets']}")
    print(f"[Saved] {target_path}")
    print(f"[Saved] {diff_path}")
    print(f"[Saved] {variant_path}")

if __name__ == "__main__":
    main()

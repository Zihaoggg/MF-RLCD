from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


SECTIONS = {
    "target": {"E_target", "sigma_y_target", "Kt_target"},
    "search": {
        "num_episodes", "steps_per_episode", "update_every_episodes",
        "ppo_min_batch", "diffusion_train_every", "diffusion_steps", "diffusion_top_frac",
    },
    "sampling": {"p_current", "p_pool", "p_jitter", "jitter_E", "jitter_Sy", "jitter_Kt"},
    "history": {"history_root", "history_topk", "history_replay_runs", "warmstart_topn", "warmstart_verify_repeats"},
    "runtime": {"work_dir", "seed", "no_resume", "device", "mtex_path"},
}
PATH_OPTIONS = {"work_dir", "history_root", "mtex_path"}


def load_defaults(path, parser):
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as source:
        document = json.load(source)
    if not isinstance(document, dict):
        raise ValueError("Configuration must be a JSON object")
    unknown = document.keys() - SECTIONS.keys()
    if unknown:
        raise ValueError(f"Unknown configuration sections: {', '.join(sorted(unknown))}")
    actions = {action.dest: action for action in parser._actions}
    defaults = {}
    for section, values in document.items():
        if not isinstance(values, dict):
            raise ValueError(f"Configuration section '{section}' must be an object")
        unknown = values.keys() - SECTIONS[section]
        if unknown:
            raise ValueError(f"Unknown options in '{section}': {', '.join(sorted(unknown))}")
        for name, value in values.items():
            action = actions[name]
            if value is None and name in {"history_root", "mtex_path"}:
                defaults[name] = None
                continue
            expected = bool if name == "no_resume" else action.type
            if expected is float:
                valid = type(value) in (int, float) and math.isfinite(value)
            else:
                valid = type(value) is expected
            if not valid:
                raise ValueError(f"Invalid value type for '{section}.{name}'")
            if expected is float:
                value = float(value)
            if action.choices is not None and value not in action.choices:
                raise ValueError(f"Invalid value for '{section}.{name}': {value}")
            if name in PATH_OPTIONS:
                if not value.strip():
                    raise ValueError(f"Path '{section}.{name}' cannot be empty")
                destination = Path(value).expanduser()
                value = str((path.parent / destination).resolve())
            defaults[name] = value
    return defaults


def validate_options(args):
    for name in ("E_target", "sigma_y_target", "Kt_target"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be finite and positive")
    for name in ("num_episodes", "steps_per_episode", "update_every_episodes",
                 "diffusion_train_every", "diffusion_steps", "warmstart_verify_repeats"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name} must be positive")
    if args.ppo_min_batch < 2:
        raise ValueError("--ppo_min_batch must be at least 2")
    for name in ("history_topk", "history_replay_runs", "warmstart_topn"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name} cannot be negative")
    probabilities = [args.p_current, args.p_pool, args.p_jitter]
    if any(not math.isfinite(p) or p < 0 for p in probabilities) or not math.isfinite(sum(probabilities)) or sum(probabilities) <= 0:
        raise ValueError("Sampling weights must be finite, nonnegative, and have a positive sum")
    for name in ("jitter_E", "jitter_Sy", "jitter_Kt"):
        if not 0 <= getattr(args, name) < 1:
            raise ValueError(f"--{name} must be in [0, 1)")
    if not 0 < args.diffusion_top_frac <= 1:
        raise ValueError("--diffusion_top_frac must be in (0, 1]")
    if not 0 <= args.seed < 2 ** 32:
        raise ValueError("--seed must be in [0, 2**32)")
    for name in PATH_OPTIONS:
        value = getattr(args, name)
        if value is not None and not value.strip():
            raise ValueError(f"--{name} cannot be empty")


def parse_options(argv=None):
    from mfrlcd.cli import build_parser

    parser = build_parser()
    preview = argparse.ArgumentParser(add_help=False)
    preview.add_argument("--config")
    known, _ = preview.parse_known_args(argv)
    try:
        if known.config:
            parser.set_defaults(**load_defaults(known.config, parser))
        args = parser.parse_args(argv)
        validate_options(args)
        for name in PATH_OPTIONS:
            value = getattr(args, name)
            if value is not None:
                setattr(args, name, str(Path(value).expanduser().resolve()))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return args


def resolved_options(args):
    result = {}
    for name in sorted(set().union(*SECTIONS.values())):
        value = getattr(args, name)
        result[name] = str(Path(value).expanduser().resolve()) if name in PATH_OPTIONS and value is not None else value
    return result


def config_fingerprint(args):
    encoded = json.dumps(resolved_options(args), sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

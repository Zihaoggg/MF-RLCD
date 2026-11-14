from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = (
    "graph_sage/model_weights.pth", "graph_sage/label_normalization.npz",
    "cgan/ckpt-3.index", "cgan/ckpt-3.data-00000-of-00001", "stress_field_bounds.npy",
)


@dataclass(frozen=True)
class EnvironmentCheck:
    name: str
    status: str
    detail: str


def inspect_environment(mtex_path=None, deep=False, device="auto"):
    checks = [EnvironmentCheck("platform", "ok" if sys.platform.startswith("linux") else "error", platform.platform())]
    supported = (3, 9) <= sys.version_info[:2] <= (3, 10)
    checks.append(EnvironmentCheck("python", "ok" if supported else "warning", platform.python_version()))
    for module, package in (("numpy", "numpy"), ("torch", "torch"), ("tensorflow", "tensorflow"), ("matlab.engine", None)):
        try:
            found = importlib.util.find_spec(module) is not None
            version = metadata.version(package) if package and found else "installed" if found else "not installed"
        except (ImportError, ModuleNotFoundError, ValueError, metadata.PackageNotFoundError):
            found, version = False, "not installed"
        checks.append(EnvironmentCheck(module, "ok" if found else "error", version))
    for executable in ("matlab", "neper", "bash", "awk", "sed"):
        path = shutil.which(executable)
        checks.append(EnvironmentCheck(executable, "ok" if path else "error", path or "not on PATH"))
    for asset in ASSETS:
        path = ROOT / "model_assets" / asset
        checks.append(EnvironmentCheck(asset, "ok" if path.is_file() and path.stat().st_size else "error", "available" if path.is_file() else "missing"))
    mtex = mtex_path or os.environ.get("MFRLCD_MTEX_PATH")
    if mtex:
        path = Path(mtex).expanduser().resolve()
        checks.append(EnvironmentCheck("MTEX", "ok" if (path / "startup_mtex.m").is_file() else "error", str(path)))
    else:
        checks.append(EnvironmentCheck("MTEX", "warning", "MATLAB startup must initialize MTEX, or set MFRLCD_MTEX_PATH"))
    if deep:
        for backend in ("torch", "tensorflow"):
            try:
                if backend == "torch":
                    import torch
                    available = torch.cuda.is_available()
                    detail = torch.cuda.get_device_name(0) if available else "CPU only"
                    detail += f"; CUDA build {torch.version.cuda}"
                else:
                    import tensorflow as tf
                    available = bool(tf.config.list_physical_devices("GPU"))
                    detail = f"GPU available: {available}; CUDA build {tf.sysconfig.get_build_info().get('cuda_version', 'none')}"
                checks.append(EnvironmentCheck(f"{backend} runtime", "error" if device == "cuda" and not available else "ok", detail))
            except Exception as exc:
                checks.append(EnvironmentCheck(f"{backend} runtime", "error", str(exc)))
    return checks


def require_environment(mtex_path=None):
    failures = [item for item in inspect_environment(mtex_path) if item.status == "error"]
    if failures:
        raise RuntimeError("Runtime prerequisites are missing: " + "; ".join(f"{item.name}: {item.detail}" for item in failures))


def select_device(preference, torch_module, tensorflow_module):
    available = torch_module.cuda.is_available()
    if preference == "cuda" and not available:
        raise RuntimeError("CUDA was requested but is unavailable to PyTorch")
    selected = "cuda" if preference != "cpu" and available else "cpu"
    if selected == "cpu":
        tensorflow_module.config.set_visible_devices([], "GPU")
    else:
        devices = tensorflow_module.config.list_physical_devices("GPU")
        if preference == "cuda" and not devices:
            raise RuntimeError("CUDA was requested but is unavailable to TensorFlow")
        for device in devices:
            tensorflow_module.config.experimental.set_memory_growth(device, True)
    return torch_module.device(selected)


def start_matlab(resources, mtex_path=None):
    import matlab.engine

    engine = matlab.engine.start_matlab()
    resources.callback(engine.quit)
    mtex_path = mtex_path or os.environ.get("MFRLCD_MTEX_PATH")
    if mtex_path:
        root = Path(mtex_path).expanduser().resolve()
        if not (root / "startup_mtex.m").is_file():
            raise FileNotFoundError(f"MTEX startup script not found in {root}")
        engine.addpath(str(root), nargout=0)
        engine.eval("startup_mtex", nargout=0)
    if not engine.exist("crystalSymmetry", "class", nargout=1):
        raise RuntimeError("MTEX is not initialized; set MFRLCD_MTEX_PATH")
    return engine


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect MF-RLCD runtime prerequisites")
    parser.add_argument("--mtex_path")
    parser.add_argument("--deep", action="store_true", help="Load ML libraries and inspect GPU availability")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    checks = inspect_environment(args.mtex_path, args.deep, args.device)
    if args.as_json:
        print(json.dumps([asdict(item) for item in checks], indent=2))
    else:
        for item in checks:
            print(f"[{item.status.upper():7}] {item.name}: {item.detail}")
    return int(any(item.status == "error" for item in checks))

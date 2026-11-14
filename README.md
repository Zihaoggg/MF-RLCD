# MF-RLCD

Microstructure inverse design using reinforcement-learning-guided conditional
diffusion.

## Environment

Reference execution environment:

| Component | Version / hardware |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS, x86_64 |
| CPU | 2 x Intel Xeon Silver 4114, 20 physical cores / 40 threads |
| GPU | NVIDIA GeForce RTX 4090 D, 24 GB |
| NVIDIA driver | 550.163.01 |
| Python | 3.9.23 |
| NumPy | 1.23.5 |
| PyTorch | 2.5.1+cu121, CUDA build 12.1 |
| TensorFlow | 2.10.1, CUDA build 11.2, cuDNN 8.x (runtime 8.7) |
| MATLAB / Engine | R2023b |
| MTEX | 5.10.2 |
| Neper | 4.5.0 |

The geometry backend requires Linux, a licensed MATLAB installation, MTEX,
Neper, Bash, `awk`, `sed`, and standard Unix utilities. A Python-only install is
not sufficient. Native Windows execution of the geometry backend is not supported.
CPU-only execution is not performance-qualified.

## Setup

Use a dedicated Python environment. From the project directory:

```bash
conda create -n mfrlcd python=3.9 pip
conda activate mfrlcd
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt

export MATLAB_ROOT=/path/to/MATLAB/R2023b
python -m pip install "$MATLAB_ROOT/extern/engines/python"
export MFRLCD_MTEX_PATH=/path/to/mtex-5.10.2
export PATH="$MATLAB_ROOT/bin:/path/to/neper/bin:$PATH"
export LD_LIBRARY_PATH="$MATLAB_ROOT/bin/glnxa64:${LD_LIBRARY_PATH:-}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
```

Replace installation paths with local paths. MTEX is initialized when
`MFRLCD_MTEX_PATH` is set; an existing MATLAB startup configuration can also
initialize it. TensorFlow's CUDA 11.x and cuDNN 8.x libraries must be available
to the dynamic linker. They are separate from PyTorch's CUDA 12.1 wheel;
`requirements.txt` does not install native CUDA libraries or geometry software.
Do not substitute a different TensorFlow/Keras major version without checking
pretrained-checkpoint compatibility. See the official
[PyTorch installation archive](https://pytorch.org/get-started/previous-versions/),
[TensorFlow compatibility table](https://www.tensorflow.org/install/source#gpu),
and [MATLAB Engine setup](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).

Keep `model_assets/` intact. It contains the GraphSAGE weights and label
normalization, the two cGAN checkpoint shards, and stress-field bounds. No
external training dataset is needed for optimization.

## Run

```bash
python run.py \
  --E_target 110000 --sigma_y_target 910 --Kt_target 1.20 \
  --num_episodes 50 --steps_per_episode 10 \
  --work_dir outputs/design_001 --no_resume
```

`E_target` and `sigma_y_target` are in MPa; `Kt_target` is dimensionless.
Run `python run.py --help` for available options. Defaults are 50 episodes and
10 steps per episode. Model paths are resolved relative to the code; output
paths are relative to the current working directory.

### Configuration and Design Workflows

```bash
python run.py doctor --deep --device cuda
python run.py optimize --config /path/to/design.json --num_episodes 100
python run.py batch --config /path/to/design.json --targets /path/to/designs.csv \
  --output_root outputs/designs --skip_completed
python run.py export --run_dir outputs/designs/design_alpha --output_dir outputs/export/design_alpha
python run.py summarize --input_root outputs/designs --output outputs/design_summary.csv
```

JSON settings are grouped into `target`, `search`, `sampling`, `history`, and
`runtime`; command-line options override configured values. Paths inside a
configuration file are relative to that file. Use `--device auto`, `cpu`, or
`cuda` to select execution devices, and `--mtex_path` to override the MTEX path.
`doctor` checks dependencies and assets; `--deep` also loads the ML libraries.

Batch design uses one isolated process per CSV row, sequentially. The CSV
contains `name,E_target,sigma_y_target,Kt_target` and an optional `seed` column.
Seeds otherwise increment from the configured seed. Runs share a history
directory; `--skip_completed` skips only completed runs with matching effective
settings and required artifacts. `--continue_on_error` allows later designs to
run after a failure; the batch command still returns a nonzero status.

Exports contain `design.json`, a per-grain `grains.csv`, and available geometry
files. Requested seed coordinates are distinguished from realized coordinates
after geometry repair. Older runs without geometry metadata export parameters
only. Run summaries can be collected into a single CSV without loading models.

Rerunning with the same `--work_dir` restores available model/optimizer state
and continues searching. Use a new output directory for an independent run.
`--no_resume` skips local checkpoint restoration but does not delete or disable
existing replay/history. Use `--history_root /path/to/prior/runs` to reuse prior
targets, replay, and warm starts; otherwise the output directory's parent is used.
`--resume` overrides a configured `no_resume` value.

## Outputs

The selected output directory contains `run_config.json`, `run_summary.json`, `best_params.npy`,
`checkpoint_final.pt`, `diff_replay.npz`, evaluation geometry under `eval_*/`,
and run records under `records/`. History indexes are maintained under the
history root.

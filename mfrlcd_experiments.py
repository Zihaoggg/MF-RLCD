#!/usr/bin/env python3

from benchmark.solvers import (
    CMAESSolver,
    DiffusionDirectSolver,
    MFRLCDSolver,
    RLOnlySolver,
    SolveResult,
    is_success,
    load_framework_module,
    make_task_suite,
    reset_env_workdir,
    set_global_seed,
    z_to_params_vector,
)
from benchmark.main import build_parser, main

if __name__ == "__main__":
    main()

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from neuralhydrology.evaluation import metrics
from neuralhydrology.nh_run import start_run, eval_run, continue_run


def main():
    # by default we assume that you have at least one CUDA-capable NVIDIA GPU
    if torch.cuda.is_available():
        continue_run(run_dir=Path("runs/grdc_basins_W365L1_regression_runoff_exp1_0603_174850"))

    # fall back to CPU-only mode
    else:
        continue_run(run_dir=Path("runs/grdc_basins_W365L1_regression_runoff_exp1_0603_174850"), gpu=-1)

if __name__ == '__main__':
    main()

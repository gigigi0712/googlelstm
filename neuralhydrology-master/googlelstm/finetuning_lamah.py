import sys
sys.path.insert(0, "/home/dalhxwlyjsuo/criait_gaozy/google/neuralhydrology-master/neuralhydrology-master")
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from neuralhydrology.evaluation import metrics
from neuralhydrology.nh_run import finetune


def main():
    # by default we assume that you have at least one CUDA-capable NVIDIA GPU
    if torch.cuda.is_available():
        finetune(config_file=Path("grdc_lamah_W365L1_regression_runoff_finetune_exp1.yml"))

    # fall back to CPU-only mode
    else:
        finetune(config_file=Path("grdc_lamah_W365L1_regression_runoff_finetune_exp1.yml"), gpu=-1)

if __name__ == '__main__':
    main()

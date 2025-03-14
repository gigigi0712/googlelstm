import sys
sys.path.insert(0, "/home/dalhxwlyjsuo/criait_gaozy/google/neuralhydrology-master/neuralhydrology-master")
import neuralhydrology  # 显式导入 neuralhydrology 模块

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from neuralhydrology.evaluation import metrics
from neuralhydrology.nh_run import start_run, eval_run
print("neuralhydrology loaded from:", neuralhydrology.__file__)

def main():
    # by default we assume that you have at least one CUDA-capable NVIDIA GPU
    if torch.cuda.is_available():
        start_run(config_file=Path("caravan_all_hanoffLSTM_CMAL.yml"))

    # fall back to CPU-only mode
    else:
        start_run(config_file=Path("caravan_all_hanoffLSTM_CMAL.yml"), gpu=-1)

if __name__ == '__main__':
    main()

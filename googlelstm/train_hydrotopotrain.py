import neuralhydrology  
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
        start_run(config_file=Path("/mnt/inaisfs/data/home/gaozy_criait/criait_gaozy/gzy/criait_gaozy/google/neuralhydrology-master/neuralhydrology-weather/googlelstm/hydrotopo_yml/hydrotopotest_5101162_regress_h0_7_direct_2180_seed2_2years_exp2.yml"))

if __name__ == '__main__':
    main()

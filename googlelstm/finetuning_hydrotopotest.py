from pathlib import Path
import matplotlib.pyplot as plt
import torch
from googlelstm.model import metrics
from neuralhydrology.nh_run import finetune


def main():
    # by default we assume that you have at least one CUDA-capable NVIDIA GPU
    path = "/mnt/inaisfs/data/home/gaozy_criait/criait_gaozy/gzy/criait_gaozy/google/neuralhydrology-master/neuralhydrology-weather/googlelstm/hydrotopo_yml"
    if torch.cuda.is_available():
        finetune(config_file=Path(fr"{path}/hydrotopotest_4127800_regress_h0_7_finetune_2180_seed1_5years_exp1.yml"))
if __name__ == '__main__':
    main()

import sys
sys.path.insert(0, "/home/dalhxwlyjsuo/criait_gaozy/google/neuralhydrology-master/neuralhydrology-master")

import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import torch
from cudalstm.model import metrics
from neuralhydrology.nh_run import start_run, eval_run
import numpy as np
from xarray import DataArray


# 读取站点ID的函数（每行一个站点ID）
def read_station_ids(filename):
    with open(filename, 'r') as file:
        station_ids = [line.strip() for line in file]
    return station_ids


# 计算ACC0.01, ACC0.03, ACC0.05, ACC0.10, ACC0.20
def calculate_accuracy(obs: DataArray, sim: DataArray,
                       thresholds: tuple[float] = (0.03, 0.04, 0.10, 0.15, 0.20)) -> dict:
    """Calculate ACC0.01, ACC0.03, ACC0.05, ACC0.10, ACC0.20 for observed and simulated data."""

    # 使用 _check_all_nan 进行检查
    if _check_all_nan(obs, sim):
        return -1  # 直接返回空字典，跳过该站点
    # Calculate the relative error
    rel_error = np.abs((sim - obs) / obs)

    # Calculate accuracy for each threshold
    accuracy = {}
    for threshold in thresholds:
        accuracy[f'ACC{threshold}'] = np.mean(rel_error <= threshold).item()  # Convert to float

    return accuracy

# 绘制并保存流量图的函数
def plot_and_save_flow_data(station_id, qobs, qsim, img_folder):
    """绘制站点的观测流量（qobs）和模拟流量（qsim），并保存为JPG文件"""
    plt.figure(figsize=(20, 5))
    plt.plot(qobs, label="Observed Flow (qobs)", color='blue', linewidth=1.5)
    plt.plot(qsim, label="Simulated Flow (qsim)", color='red', linewidth=1.5)
    plt.title(f"Flow Data for Station {station_id}", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Flow (m³/s)", fontsize=12)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 保存图像为 JPG 文件
    img_filename = img_folder / f"station_{station_id}_flow_plot.jpg"
    plt.savefig(img_filename, format="jpg")
    plt.close()

    print(f"Flow plot for Station {station_id} saved as {img_filename}")

def main():
    # # 读取保存的均值和方差的 CSV 文件
    # stats_df = pd.read_csv("G:/dataset/yellowriver/static/93_qobs_statistics.csv")
    # 从文件中读取站点ID
    station_ids = read_station_ids("grdc_5670.txt")  # 假设文件名为 "1_danube_basin.txt"

    # 创建存储图像的文件夹
    img_folder = Path("/home/dalhxwlyjsuo/criait_gaozy/google/neuralhydrology-master/neuralhydrology-master/cudalstm/results/Camelus_W365L1_531_runoff")
    img_folder.mkdir(parents=True, exist_ok=True)


    # 存储所有站点的 ACC、NSE 和其他指标
    all_accuracy = {f'ACC{threshold}': [] for threshold in [0.03, 0.04, 0.10, 0.15, 0.20]}
    all_nse = []
    all_metrics = {}  # 存储所有站点的其他指标

    # 路径设置
    run_dir = Path("runs/cudalstm_caravan_grdc_5670_runoff_basins_exp1_1203_190544")
    eval_run(run_dir=run_dir, period="test", epoch=15)

    with open(run_dir / "test" / "model_epoch015" / "test_results.p", "rb") as fp:
        results = pickle.load(fp)

    # 打印可用的结果键
    print(results.keys())

    # 遍历所有站点ID
    for station_id in station_ids:
        # 获取每个站点的观测数据和模拟数据
        try:
            qobs = results[station_id]['1D']['xr']['streamflow_obs']
            qsim = results[station_id]['1D']['xr']['streamflow_sim']
            # qsim = qsim.mean(dim="samples")
        except KeyError:
            print(f"站点 {station_id} 没有找到相关数据")
            continue

        # # 获取均值和方差
        # stats = stats_df[stats_df['station_id'] == f'ID_{station_id}']
        # qobs_mean = stats['qobs_mean'].values[0]
        # qobs_std = stats['qobs_std'].values[0]
        #
        # # 反归一化流量数据
        # qobs = qobs * qobs_std + qobs_mean
        # qsim = qsim * qobs_std + qobs_mean

        # # 绘制并保存流量图
        # plot_and_save_flow_data(station_id, qobs, qsim, img_folder)

        # 计算指标
        values = metrics.calculate_all_metrics(qobs.isel(time_step=-1), qsim.isel(time_step=-1))
        for key, val in values.items():
            print(f"Station {station_id} - {key}: {val:.3f}")

        # 将每个站点的其他指标存储到 all_metrics 中
        if not all_metrics:
            # 初始化字典
            all_metrics = {key: [] for key in values.keys()}
        for key, val in values.items():
            all_metrics[key].append(val)

        # 获取 NSE 和添加到所有 NSE 列表
        if str(station_id) not in results:
            print(f"Warning: No results for station {station_id}, skipping.")
            continue  # 直接跳过该站点

        if '1D' not in results[str(station_id)]:
            print(f"Warning: No '1D' data for station {station_id}, skipping.")
            continue  # 继续跳过

        if 'NSE' not in results[str(station_id)]['1D']:
            print(f"Warning: No 'NSE' metric for station {station_id}, skipping.")
            continue  # 继续跳过

        nse = results[str(station_id)]['1D']['NSE']
        all_nse.append(nse)

        nse = results[str(station_id)]['1D']['NSE']
        all_nse.append(nse)

        # 计算 ACC0.01, ACC0.03, ACC0.05, ACC0.10, ACC0.20
        accuracy = calculate_accuracy(qobs.isel(time_step=-1), qsim.isel(time_step=-1),
                                      thresholds=(0.03, 0.04, 0.10, 0.15, 0.20))

        # 将每个站点的 ACC 存储到 all_accuracy 中
        for threshold in [0.03, 0.04, 0.10, 0.15, 0.20]:
            all_accuracy[f'ACC{threshold}'].append(accuracy[f'ACC{threshold}'])

    # 计算所有站点的平均 ACC 和 NSE
    avg_accuracy = {key: np.mean(values) for key, values in all_accuracy.items()}
    avg_nse = np.mean(all_nse)
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}  # 计算所有站点其他指标的均值

    print("\nAverage ACC, NSE and other metrics across all stations:")

    # 打印平均 ACC
    for key, val in avg_accuracy.items():
        print(f"{key}: {val:.3f}")

    # 打印平均 NSE
    print(f"Average NSE: {avg_nse:.3f}")

    # 打印平均其他指标
    for key, val in avg_metrics.items():
        print(f"Average {key}: {val:.3f}")

    # 创建 DataFrame 保存每个站点的结果
    accuracy_df = pd.DataFrame(all_accuracy)
    accuracy_df['station_id'] = station_ids  # Add station IDs as a column
    accuracy_df['NSE'] = all_nse  # Add NSE as a column
    for key, values in all_metrics.items():
        accuracy_df[key] = values  # 添加其他指标列

    # 添加平均值行
    avg_row = {key: val for key, val in avg_accuracy.items()}
    avg_row['station_id'] = 'Average'
    avg_row['NSE'] = avg_nse
    for key, val in avg_metrics.items():
        avg_row[key] = val

    # 将平均值添加到 DataFrame 的第一行
    avg_row_df = pd.DataFrame([avg_row])  # 将 avg_row 转换为 DataFrame
    accuracy_df = pd.concat([avg_row_df, accuracy_df], ignore_index=True)  # 将平均值放在第一行

    # 保存为 Excel 文件
    accuracy_df.to_excel("/home/dalhxwlyjsuo/criait_gaozy/google/neuralhydrology-master/neuralhydrology-master/cudalstm/results/cudalstm_caravan_grdc_5670_runoff_basins_exp1.xlsx", index=False)


def _check_all_nan(obs: DataArray, sim: DataArray):
    """Check if all observations or simulations are NaN and return a flag if this is the case."""
    if obs.isnull().all():
        print("Warning: All observed values are NaN, skipping this station.")
        return True  # 标记为全 NaN

    if sim.isnull().all():
        print("Warning: All simulated values are NaN, skipping this station.")
        return True  # 标记为全 NaN

    return False  # 正常数据


class AllNaNError(Exception):
    """Raised by `calculate_(all_)metrics` if all observations or all simulations are NaN."""



if __name__ == '__main__':
    main()

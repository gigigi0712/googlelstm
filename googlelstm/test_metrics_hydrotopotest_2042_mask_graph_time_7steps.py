import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xarray import DataArray
from output_metrics import metrics
from multiprocessing import Pool, cpu_count


# ========= 把原本的 for h 循环体打包成函数 =========
def process_single_model(h):
    model = f"google_h{h}"
    distance = 10
    input_dir = fr"/mnt/inaisfs/data/home/gaozy_criait/criait_gaozy/gzy/criait_gaozy/google/neuralhydrology-master/neuralhydrology-weather/googlelstm/output_hydrotopo_metrics/{output_metrics_name}/google_ours_h{h}"

    # 设置中文显示（如果绘图用到）
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # ────────────────────────────────────────────────
    # 读取只允许计算的站点ID列表（从 txt 文件）
    # 请修改为你的实际路径
    allowed_ids_path = "/mnt/inaisfs/data/home/gaozy_criait/criait_gaozy/gzy/criait_gaozy/google/neuralhydrology-master/neuralhydrology-weather/googlelstm/hydrotopotest_2042.txt"   # ← 这里改成真实路径！
    # allowed_ids_path = "/mnt/inaisfs/.../grdc_train_id_4208025.txt"  # 示例

    with open(allowed_ids_path, "r") as f:
        allowed_lines = [line.strip() for line in f if line.strip()]

    allowed_ids = set(
        line.replace("hydrotopotest_", "")
        for line in allowed_lines
    )
    print(f"h{h} | 读取到 {len(allowed_ids)} 个允许的站点ID，将只处理这些站点")
    # ────────────────────────────────────────────────

    # ========== 阶段一：计算并保存原始指标 ==========
    results = []
    processed_count = 0
    skipped_count = 0
    no_data_count = 0

    for file_name in os.listdir(input_dir):
        if not (file_name.startswith("ID_") and file_name.endswith("_done.csv")):
            continue

        file_id = file_name.split("_")[1]   # 如 "4208130"

        if file_id not in allowed_ids:
            skipped_count += 1
            continue

        # ─── 新增：从时期表中获取该站点的测试时间段 ───
        if file_id in period_dict:
            start_year, end_year = period_dict[file_id]
            print(f"  {file_name} 使用测试时期: {start_year}-{end_year}")
        else:
            print(f"  {file_name} 未在时期表中找到，跳过")
            skipped_count += 1
            continue
        # ────────────────────────────────────────

        processed_count += 1
        file_path = os.path.join(input_dir, file_name)

        try:
            data = pd.read_csv(file_path, sep=";", usecols=["date", "qobs_pre", "qsim"])
            data["date"] = pd.to_datetime(data["date"])
            data = data.dropna(subset=["qobs_pre", "qsim"])

            filtered_data = data[(data["date"].dt.year >= start_year) & (data["date"].dt.year <= end_year)]
            if filtered_data.empty:
                print(f"  {file_name} 在 {start_year}-{end_year} 年没有数据，跳过")
                no_data_count += 1
                continue

            qobs = DataArray(
                filtered_data["qobs_pre"].values.reshape(-1, 1),
                dims=["date", "time_step"],
                coords={"date": filtered_data["date"], "time_step": [1]}
            )
            qsim = DataArray(
                filtered_data["qsim"].values.reshape(-1, 1),
                dims=["date", "time_step"],
                coords={"date": filtered_data["date"], "time_step": [1]}
            )

            qobs_selected = qobs.isel(time_step=-1)
            qsim_selected = qsim.isel(time_step=-1)
            values = metrics.calculate_all_metrics(qobs_selected, qsim_selected)

            result_row = {"id": file_id}
            result_row.update(values)
            results.append(result_row)

        except Exception as e:
            print(f"处理 {file_name} 出错：{e}")
            continue

    print(f"h{h} | 实际处理：{processed_count} 个站点 | 跳过（不在列表）：{skipped_count} | 无数据跳过：{no_data_count}")

    if not results:
        print(f"警告：h{h} 没有任何站点被成功计算！")
        return

    results_df = pd.DataFrame(results)
    results_df.columns = results_df.columns.str.lower()

    original_path = f"{output_dir}/metrics_summary_original_{model}_distance{distance}.csv"
    results_df.to_csv(original_path, index=False)
    print(f"✅ 原始结果已保存：{original_path}")

    # ====== 整体统计指标保存 ======
    re_peak_timing_flat = [item for sublist in results_df["re-peak-timing"] for item in sublist]
    re_peak_mape_flat = [item for sublist in results_df["re-peak-mape"] for item in sublist]

    stats = {
        "Total_RE-Peak-Timing_Count": len(re_peak_timing_flat),
        "RE-Peak-Timing<=6_Count": sum(1 for x in re_peak_timing_flat if x <= 6),
        "RE-Peak-Timing<=0_Count": sum(1 for x in re_peak_timing_flat if x <= 0),
        "0<RE-Peak-Timing<=6_Count": sum(1 for x in re_peak_timing_flat if 0 < x <= 6),
        "RE-Peak-Timing==0": sum(1 for x in re_peak_timing_flat if x == 0),
        "PEAK-ACC": sum(1 for x in re_peak_timing_flat if x == 0) / len(re_peak_timing_flat) if re_peak_timing_flat else 0,
        "RE-Peak-MAPE_Mean(RE-Peak-Timing==0)": (
            sum(re_peak_mape_flat[i] for i, x in enumerate(re_peak_timing_flat) if x == 0) /
            sum(1 for x in re_peak_timing_flat if x == 0)
        ) if any(x == 0 for x in re_peak_timing_flat) else np.nan,
        "RE-Peak-MAPE_Mean(RE-Peak-Timing<=0)": (
                sum(re_peak_mape_flat[i] for i, x in enumerate(re_peak_timing_flat) if x <= 0) /
                sum(1 for x in re_peak_timing_flat if x <= 0)
        ),
        "RE-Peak-MAPE_Mean(0<RE-Peak-Timing<=6)": (
                sum(re_peak_mape_flat[i] for i, x in enumerate(re_peak_timing_flat) if 0 < x <= 6) /
                sum(1 for x in re_peak_timing_flat if 0 < x <= 6)
        ),
        "RE-Peak-Timing_Mean(RE-Peak-Timing<=0)": (
                sum(x for x in re_peak_timing_flat if x <= 0) /
                sum(1 for x in re_peak_timing_flat if x <= 0)
        ),
        "RE-Peak-Timing_Mean(0<RE-Peak-Timing<=6)": (
                sum(x for x in re_peak_timing_flat if 0 < x <= 6) /
                sum(1 for x in re_peak_timing_flat if 0 < x <= 6)
        ),
        "RE-Peak-Timing_Mean(RE-Peak-Timing<=6)": (
                sum(abs(x) for x in re_peak_timing_flat if x <= 6) /
                sum(1 for x in re_peak_timing_flat if x <= 6)
        ),
        "NSE<0_Count": (results_df["nse"] < 0).sum(),
        "NSE>0_Count": (results_df["nse"] > 0).sum(),
    }

    # 你原来的其他统计项可以继续加在这里，例如 NSE>0 的平均值等
    nse_positive = results_df[results_df["nse"] > 0]
    metrics_to_average = [
        "acc10", "acc20", "nse", "mse", "rmse", "kge",
        "alpha-nse", "beta-kge", "beta-nse", "pearson-r",
        "fhv", "fms", "flv", "peak-timing", "peak-mape"
    ]
    for metric in metrics_to_average:
        if metric in nse_positive.columns:
            stats[f"{metric}_mean"] = nse_positive[metric].mean()

    stats_df = pd.DataFrame([stats])
    stats_path = f"{output_dir}/metrics_summary_with_stats_{model}_distance{distance}.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"✅ 整体统计结果已保存：{stats_path}")

    # ========== 阶段二：分段峰值指标测试 ==========
    # （以下部分基本保持原样，只是数据源已经是被筛选过的 results_df）

    def parse_array_column(column_data):
        if pd.isna(column_data):
            return []
        return [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', str(column_data))]

    def process_models_by_percentile_ranges(file_paths, model_names, rate):
        metrics_segmented = {}
        for file_path, model_name in zip(file_paths, model_names):
            print(f"处理分段指标：{model_name}")
            df = pd.read_csv(file_path)
            model_data = []
            for index, row in df.iterrows():
                try:
                    re_peak_timing = eval(row['re-peak-timing']) if isinstance(row['re-peak-timing'], str) else row['re-peak-timing']
                    get_obs_peak_value = parse_array_column(row['get-obs-peak-value'])
                    get_sim_peak_value = parse_array_column(row['get-sim-peak-value'])
                except Exception as e:
                    print(f"第{index+1}行解析失败：{e}")
                    continue

                data_seg = pd.DataFrame({
                    "re_peak_timing": re_peak_timing,
                    "get_obs_peak_value": get_obs_peak_value,
                    "get_sim_peak_value": get_sim_peak_value
                })
                data_seg['peak_value_error'] = abs(data_seg['get_sim_peak_value'] - data_seg['get_obs_peak_value']) / data_seg['get_obs_peak_value']
                model_data.append(data_seg)

            if not model_data:
                print(f"{model_name} 无有效峰值数据")
                continue

            combined_data = pd.concat(model_data, ignore_index=True)
            combined_data_sorted = combined_data.sort_values(by="get_obs_peak_value").reset_index(drop=True)
            total_len = len(combined_data_sorted)

            counts = [int(total_len * r / 100) for r in rate]
            counts[-1] = total_len - sum(counts[:-1])

            start_idx, results = 0, []
            for i, count in enumerate(counts):
                if count == 0:
                    continue
                subset = combined_data_sorted.iloc[start_idx:start_idx + count]
                peak_acc = (subset['re_peak_timing'] == 0).sum() / len(subset) if len(subset) > 0 else 0
                correct_timing_subset = subset[subset['re_peak_timing'] == 0]
                peak_error = correct_timing_subset['peak_value_error'].mean() if len(correct_timing_subset) > 0 else np.nan
                lower, upper = sum(rate[:i]), sum(rate[:i+1])
                label = f"{lower}-{upper}%"
                results.append({
                    "range_label": label,
                    "peak-acc": round(peak_acc, 3),
                    "peak-error": round(peak_error, 3) if not np.isnan(peak_error) else "N/A",
                    "sample_count": len(subset)
                })
                start_idx += count

            result_df = pd.DataFrame(results)
            metrics_segmented[model_name] = result_df
            save_name = f"{output_dir}/{model_name}_分段指标结果.csv"
            result_df.to_csv(save_name, index=False, encoding='utf-8-sig')
            print(f"✅ 已保存分段结果：{save_name}")

        return metrics_segmented

    file_paths = [original_path]
    model_names = ["Google"]
    rate = [60, 20, 15, 2.5, 1.5, 1]
    metrics_segmented = process_models_by_percentile_ranges(file_paths, model_names, rate)

    for model_name, df in metrics_segmented.items():
        print(f"\n{model_name} 的分段指标结果：")
        print(df.to_string(index=False))

    # 保存合并的分段结果
    all_results = []
    for model_name, df in metrics_segmented.items():
        df_copy = df.copy()
        df_copy.insert(0, "model_name", model_name)
        all_results.append(df_copy)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        stats_path_seg = f"{output_dir}/all_models_metrics_segmented_{model}_distance{distance}.csv"
        final_df.to_csv(stats_path_seg, index=False, encoding="utf-8-sig")
        print(f"分段结果合并保存：{stats_path_seg}")


# ========= 主进程并行执行所有 h =========
if __name__ == "__main__":
    # ─── 新增：读取时期表 CSV ──────────────────────────────
    periods_path = "/mnt/inaisfs/data/dataset/gaozy/new_dataset_4261/hydrotopotest_2042_periods.csv"
    df_periods = pd.read_csv(periods_path)
    df_periods['station_id'] = df_periods['station_id'].astype(str)  # 确保字符串格式
    period_dict = dict(zip(df_periods['station_id'], zip(df_periods['test_start_year'], df_periods['test_end_year'])))
    print(f"读取到 {len(period_dict)} 个站点的测试时期信息")
    # ─────────────────────────────────────────────────────

    for seed in range(1, 6):  # seed1-5 可自行扩展
        output_metrics_name = fr"hydrotopotest_2042_direct_0_7steps_seed{seed}_5years_exp1"

        output_metrics_graph = fr"hydrotopotest_2042_direct_0_7steps_seed{seed}_5years_exp1"

        output_dir = fr"/mnt/inaisfs/data/home/gaozy_criait/criait_gaozy/gzy/criait_gaozy/google/neuralhydrology-master/neuralhydrology-weather/googlelstm/output_hydrotopo_metrics/{output_metrics_graph}"
        os.makedirs(output_dir, exist_ok=True)

        num_workers = 24   # 可根据机器调整
        with Pool(num_workers) as pool:
            pool.map(process_single_model, range(0, 8))

        print("✅ 所有模型计算完成，开始执行合并部分...")

        # ─── 合并所有 h 的分段结果 ────────────────────────────────
        files = [f"{output_dir}/all_models_metrics_segmented_google_h{i}_distance10.csv" for i in range(0, 8)]
        range_labels = ['0-60%', '60-80%', '80-95%', '95-97.5%', '97.5-99.0%', '99.0-100.0%']
        acc_matrix = []
        err_matrix = []

        for file in files:
            if not os.path.exists(file):
                print(f"未找到：{file}，跳过")
                continue
            df = pd.read_csv(file)
            df = df.set_index('range_label').reindex(range_labels)
            acc_matrix.append(df['peak-acc'].values)
            err_matrix.append(df['peak-error'].values)

        if acc_matrix:
            acc_df = pd.DataFrame(acc_matrix, index=[f'h{i}' for i in range(0, 8)], columns=range_labels)
            err_df = pd.DataFrame(err_matrix, index=[f'h{i}' for i in range(0, 8)], columns=range_labels)
            acc_df.to_csv(os.path.join(output_dir, 'peak_acc_merged.csv'))
            err_df.to_csv(os.path.join(output_dir, 'peak_error_merged.csv'))
            print("峰值准确率 & 误差 已合并保存")

        # ─── 合并所有 h 的统计汇总 ────────────────────────────────
        output_file = os.path.join(output_dir, f"metrics_summary_with_stats_google_merged_h0_h7_distance10.csv")
        merged_data = pd.DataFrame()

        for i in range(0, 8):
            file_name = f'metrics_summary_with_stats_google_h{i}_distance10.csv'
            file_path = os.path.join(output_dir, file_name)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                merged_data = pd.concat([merged_data, df], ignore_index=True)
                print(f'已合并 {file_name}')
            else:
                print(f'警告: {file_name} 不存在，跳过')

        if not merged_data.empty:
            merged_data.to_csv(output_file, index=False)
            print(f'\n合并完成！结果保存到: {output_file}')
            print(f'总行数: {len(merged_data)}')
        else:
            print('错误: 没有找到任何统计文件，无法合并。')
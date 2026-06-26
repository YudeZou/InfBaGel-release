#!/usr/bin/env python3
"""
Success rate re-screening script:
recompute experiment success rates based on the configured thresholds.
Supports batch testing of multiple experiments and multiple thresholds, outputting results in table form.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple

def load_experiment_data(experiment_name: str) -> Tuple[List[Dict], int, int]:
    """
    Load all data for the specified experiment

    Returns:
        tuple: (list of metrics for all samples, number of processed scenes, total number of samples)
    """
    base_output_dir = f'hosi_results/{experiment_name}'
    
    if not os.path.exists(base_output_dir):
        print(f"错误：找不到实验目录 {base_output_dir}")
        return [], 0, 0
    
    all_scenes_metrics = []
    processed_scenes = 0
    
    # iterate over all scene directories
    scene_dirs = [d for d in os.listdir(base_output_dir) 
                  if os.path.isdir(os.path.join(base_output_dir, d))]
    
    for scene_name in scene_dirs:
        scene_output_dir = os.path.join(base_output_dir, scene_name)
        evaluation_summary_path = os.path.join(scene_output_dir, 'evaluation_summary.json')
        
        if os.path.exists(evaluation_summary_path):
            try:
                with open(evaluation_summary_path, 'r') as f:
                    scene_results = json.load(f)
                
                if 'individual_metrics' in scene_results:
                    metrics = scene_results['individual_metrics']
                    all_scenes_metrics.extend(metrics)
                    processed_scenes += 1
            except Exception as e:
                print(f"警告：无法读取场景 {scene_name} 的结果: {e}")
    
    return all_scenes_metrics, processed_scenes, len(all_scenes_metrics)

def calculate_success_rate(metrics_list: List[Dict], threshold: float) -> Dict:
    """
    Compute success rate based on the threshold

    Args:
        metrics_list: list of metric data
        threshold: threshold (cm)

    Returns:
        dict: success rate statistics
    """
    if not metrics_list:
        return {
            'success_count': 0,
            'total_count': 0,
            'success_rate': 0.0,
            'pelvis_success_count': 0,
            'object_success_count': 0,
            'both_success_count': 0
        }
    
    success_count = 0
    pelvis_success_count = 0
    object_success_count = 0
    total_count = len(metrics_list)
    
    for metric in metrics_list:
        pelvis_error = metric.get('xy_points_err', float('inf'))
        object_error = metric.get('end_obj_trans_err', float('inf'))
        
        pelvis_success = pelvis_error < threshold
        object_success = object_error < threshold
        both_success = pelvis_success and object_success

        if pelvis_success:
            pelvis_success_count += 1
        if object_success:
            object_success_count += 1
        if both_success:
            success_count += 1
    
    return {
        'success_count': success_count,
        'total_count': total_count,
        'success_rate': success_count / total_count * 100 if total_count > 0 else 0.0,
        'pelvis_success_count': pelvis_success_count,
        'object_success_count': object_success_count,
        'both_success_count': success_count,
        'pelvis_success_rate': pelvis_success_count / total_count * 100 if total_count > 0 else 0.0,
        'object_success_rate': object_success_count / total_count * 100 if total_count > 0 else 0.0
    }

def _collect_success_rates(experiment_results, threshold):
    """Collect per-experiment success rates for a given threshold."""
    rates = []
    for exp_data in experiment_results.values():
        if threshold in exp_data:
            rates.append(exp_data[threshold]['success_rate'])
    return rates

def create_results_table(experiment_results: Dict[str, Dict[float, Dict]]) -> pd.DataFrame:
    """
    Create the results table

    Args:
        experiment_results: {experiment_name: {threshold: results}}

    Returns:
        pandas DataFrame: results table
    """
    if not experiment_results:
        return pd.DataFrame()
    
    # get all thresholds
    all_thresholds = set()
    for exp_data in experiment_results.values():
        all_thresholds.update(exp_data.keys())
    all_thresholds = sorted(list(all_thresholds))
    
    # build table data
    table_data = []
    for exp_name, exp_data in experiment_results.items():
        row = {'Experiment': exp_name}
        for threshold in all_thresholds:
            if threshold in exp_data:
                success_rate = exp_data[threshold]['success_rate']
                row[f'{threshold:.0f}cm'] = f"{success_rate:.2f}%"
            else:
                row[f'{threshold:.0f}cm'] = "N/A"
        table_data.append(row)

    # compute the average row
    if len(experiment_results) > 1:
        avg_row = {'Experiment': 'Average'}
        for threshold in all_thresholds:
            rates = _collect_success_rates(experiment_results, threshold)
            if rates:
                avg_rate = np.mean(rates)
                avg_row[f'{threshold:.0f}cm'] = f"{avg_rate:.2f}%"
            else:
                avg_row[f'{threshold:.0f}cm'] = "N/A"
        table_data.append(avg_row)
    
    return pd.DataFrame(table_data)

def print_detailed_results(experiment_name: str, results: Dict[float, Dict], verbose: bool = False):
    """
    Print detailed results
    """
    print(f"\n=== {experiment_name} ===")
    print(f"总样本数: {results[list(results.keys())[0]]['total_count']}")
    
    if verbose:
        for threshold in sorted(results.keys()):
            result = results[threshold]
            print(f"\n阈值 {threshold:.0f}cm:")
            print(f"  总体成功率: {result['success_rate']:.2f}% ({result['success_count']}/{result['total_count']})")
            print(f"  骨盆位置成功率: {result['pelvis_success_rate']:.2f}% ({result['pelvis_success_count']}/{result['total_count']})")
            print(f"  物体位置成功率: {result['object_success_rate']:.2f}% ({result['object_success_count']}/{result['total_count']})")
    else:
        print("阈值成功率:", end=" ")
        for threshold in sorted(results.keys()):
            result = results[threshold]
            print(f"{threshold:.0f}cm: {result['success_rate']:.2f}%", end=", ")
        print()

def main():
    parser = argparse.ArgumentParser(description='重新计算实验成功率')
    parser.add_argument('experiments', nargs='*', help='实验名称列表')
    parser.add_argument('--experiments', dest='exp_list', nargs='+', help='实验名称列表（通过选项指定）')
    parser.add_argument('--thresholds', nargs='+', type=float, default=[5, 10, 15, 20], 
                       help='阈值列表（cm）')
    parser.add_argument('--output', '-o', help='输出CSV文件路径')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细信息')
    
    args = parser.parse_args()
    
    # process the experiment name list
    experiments = args.experiments or []
    if args.exp_list:
        experiments.extend(args.exp_list)
    
    if not experiments:
        print("错误：请指定至少一个实验名称")
        print("用法: python recompute_success_rate.py exp1 [exp2 ...] [--thresholds 5 10 15 20]")
        print("或: python recompute_success_rate.py --experiments exp1 exp2 --thresholds 5 10 15 20")
        return
    
    thresholds = sorted(args.thresholds)
    print(f"分析 {len(experiments)} 个实验，使用 {len(thresholds)} 个阈值: {thresholds}")
    
    # collect results from all experiments
    experiment_results = {}
    
    for experiment_name in tqdm(experiments, desc="处理实验"):
        print(f"\n正在处理实验: {experiment_name}")
        
        # load experiment data
        metrics_list, processed_scenes, total_samples = load_experiment_data(experiment_name)
        
        if not metrics_list:
            print(f"警告：实验 {experiment_name} 没有找到有效数据")
            continue
        
        print(f"加载了 {processed_scenes} 个场景，共 {total_samples} 个样本")
        
        # compute success rates under different thresholds
        experiment_results[experiment_name] = {}
        for threshold in thresholds:
            results = calculate_success_rate(metrics_list, threshold)
            experiment_results[experiment_name][threshold] = results
        
        # print detailed results
        if args.verbose or len(experiments) == 1:
            print_detailed_results(experiment_name, experiment_results[experiment_name], args.verbose)
    
    # create and display the results table
    if experiment_results:
        print(f"\n{'='*60}")
        print("成功率对比表")
        print(f"{'='*60}")
        
        results_df = create_results_table(experiment_results)
        print(results_df.to_string(index=False))
        
        # save to a CSV file
        if args.output:
            # build a numeric version of the table for CSV export
            csv_data = []
            for exp_name, exp_data in experiment_results.items():
                row = {'Experiment': exp_name}
                for threshold in sorted(set().union(*[exp_data.keys() for exp_data in experiment_results.values()])):
                    if threshold in exp_data:
                        row[f'{threshold:.0f}cm'] = exp_data[threshold]['success_rate']
                    else:
                        row[f'{threshold:.0f}cm'] = np.nan
                csv_data.append(row)

            # compute the average row
            if len(experiment_results) > 1:
                avg_row = {'Experiment': 'Average'}
                for threshold in sorted(set().union(*[exp_data.keys() for exp_data in experiment_results.values()])):
                    rates = []
                    for exp_data in experiment_results.values():
                        if threshold in exp_data:
                            rates.append(exp_data[threshold]['success_rate'])
                    if rates:
                        avg_row[f'{threshold:.0f}cm'] = np.mean(rates)
                    else:
                        avg_row[f'{threshold:.0f}cm'] = np.nan
                csv_data.append(avg_row)
            
            csv_df = pd.DataFrame(csv_data)
            csv_df.to_csv(args.output, index=False)
            print(f"\n结果已保存至: {args.output}")
        
        # print the statistics summary
        print(f"\n{'='*60}")
        print("统计摘要")
        print(f"{'='*60}")
        
        total_samples = sum(
            list(exp_data.values())[0]['total_count'] 
            for exp_data in experiment_results.values()
        )
        print(f"总实验数: {len(experiment_results)}")
        print(f"总样本数: {total_samples}")
        
        # show the average success rate under different thresholds
        print("\n平均成功率:")
        for threshold in thresholds:
            rates = _collect_success_rates(experiment_results, threshold)
            if rates:
                avg_rate = np.mean(rates)
                print(f"  {threshold:.0f}cm 阈值: {avg_rate:.2f}%")

if __name__ == "__main__":
    main()
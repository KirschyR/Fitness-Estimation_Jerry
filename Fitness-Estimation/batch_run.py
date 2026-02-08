#!/usr/bin/env python3
"""
批量并行运行脚本
===============

完成四个任务：
1. 估计 female_fecundity 和 male_fecundity (固定viability=1.0)
2. 估计 female_viability 和 male_viability (固定fecundity=1.0)
3. 模型选择：改变 female_fecundity 和 male_fecundity，运行模型选择
4. 模型选择：改变 female_viability 和 male_viability，运行模型选择

参数范围：0.3 到 1.0，步长 0.1
每个任务：8 × 8 = 64 个参数组合，全部并行运行
"""

import os
import sys
import subprocess
import random
import itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime


# 参数范围
PARAM_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MAX_WORKERS = 8  # 并行进程数


def run_command(cmd, task_name):
    """运行单个命令"""
    print(f"[{task_name}] 启动: {cmd[-6:-1]}")  # 显示参数摘要
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            print(f"[{task_name}] ✓ 完成")
            return True, None
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def task1_estimate_fecundity():
    """任务1：估计 female_fecundity 和 male_fecundity"""
    print("\n" + "="*70)
    print("任务 1: 估计 Fecundity (Female 和 Male)")
    print("="*70)
    
    tasks = []
    for f_fec, m_fec in itertools.product(PARAM_VALUES, PARAM_VALUES):
        seed = random.randint(1000, 9999)
        run_name = f"est_fec_f{f_fec:.1f}_m{m_fec:.1f}_seed{seed}"
        
        cmd = [
            'python', 'pmcmc_inference_multi.py',
            '--true_female_fecundity', str(f_fec),
            '--true_male_fecundity', str(m_fec),
            '--true_female_viability', '1.0',
            '--true_male_viability', '1.0',
            '--estimate_params', 'female_fecundity', 'male_fecundity',
            '--seed', str(seed),
            '--run_name', run_name,
            '--json_dir', 'results_json'
        ]
        
        tasks.append((cmd, run_name))
    
    print(f"生成 {len(tasks)} 个任务...")
    return tasks


def task2_estimate_viability():
    """任务2：估计 female_viability 和 male_viability"""
    print("\n" + "="*70)
    print("任务 2: 估计 Viability (Female 和 Male)")
    print("="*70)
    
    tasks = []
    for f_vib, m_vib in itertools.product(PARAM_VALUES, PARAM_VALUES):
        seed = random.randint(1000, 9999)
        run_name = f"est_vib_f{f_vib:.1f}_m{m_vib:.1f}_seed{seed}"
        
        cmd = [
            'python', 'pmcmc_inference_multi.py',
            '--true_female_viability', str(f_vib),
            '--true_male_viability', str(m_vib),
            '--true_female_fecundity', '1.0',
            '--true_male_fecundity', '1.0',
            '--estimate_params', 'female_viability', 'male_viability',
            '--seed', str(seed),
            '--run_name', run_name,
            '--json_dir', 'results_json'
        ]
        
        tasks.append((cmd, run_name))
    
    print(f"生成 {len(tasks)} 个任务...")
    return tasks


def task3_model_comparison_fecundity():
    """任务3：改变 female_fecundity 和 male_fecundity，运行模型选择"""
    print("\n" + "="*70)
    print("任务 3: 模型选择 - Fecundity (Female 和 Male)")
    print("="*70)
    
    tasks = []
    for f_fec, m_fec in itertools.product(PARAM_VALUES, PARAM_VALUES):
        seed = random.randint(1000, 9999)
        run_name = f"mc_fec_f{f_fec:.1f}_m{m_fec:.1f}_seed{seed}"
        
        cmd = [
            'python', 'quick_model_comparison.py',
            '--true_female_viability', '1.0',
            '--true_male_viability', '1.0',
            '--true_female_fecundity', str(f_fec),
            '--true_male_fecundity', str(m_fec),
            '--seed', str(seed),
            '--json_dir', 'results_json',
            '--fig_dir', 'figs'
        ]
        
        tasks.append((cmd, run_name))
    
    print(f"生成 {len(tasks)} 个任务...")
    return tasks


def task4_model_comparison_viability():
    """任务4：改变 female_viability 和 male_viability，运行模型选择"""
    print("\n" + "="*70)
    print("任务 4: 模型选择 - Viability (Female 和 Male)")
    print("="*70)
    
    tasks = []
    for f_vib, m_vib in itertools.product(PARAM_VALUES, PARAM_VALUES):
        seed = random.randint(1000, 9999)
        run_name = f"mc_vib_f{f_vib:.1f}_m{m_vib:.1f}_seed{seed}"
        
        cmd = [
            'python', 'quick_model_comparison.py',
            '--true_female_viability', str(f_vib),
            '--true_male_viability', str(m_vib),
            '--true_female_fecundity', '1.0',
            '--true_male_fecundity', '1.0',
            '--seed', str(seed),
            '--json_dir', 'results_json',
            '--fig_dir', 'figs'
        ]
        
        tasks.append((cmd, run_name))
    
    print(f"生成 {len(tasks)} 个任务...")
    return tasks


def run_tasks_in_parallel(tasks):
    """并行运行任务"""
    completed = 0
    failed = 0
    
    print(f"\n使用 {MAX_WORKERS} 个并行进程开始运行...")
    print(f"总任务数: {len(tasks)}")
    print("="*70 + "\n")
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_task = {
            executor.submit(run_command, cmd, task_name): task_name
            for cmd, task_name in tasks
        }
        
        # 处理完成的任务
        for future in as_completed(future_to_task):
            task_name = future_to_task[future]
            try:
                success, error = future.result()
                if success:
                    completed += 1
                else:
                    failed += 1
                    print(f"[{task_name}] ✗ 失败: {error}")
            except Exception as e:
                failed += 1
                print(f"[{task_name}] ✗ 异常: {e}")
            
            # 定期输出进度
            total_done = completed + failed
            if total_done % 10 == 0:
                print(f"\n进度: {total_done}/{len(tasks)} "
                      f"(成功: {completed}, 失败: {failed})\n")
    
    return completed, failed


def main():
    # 检查工作目录
    if not Path('pmcmc_inference_multi.py').exists():
        print("错误：请在包含 pmcmc_inference_multi.py 的目录运行此脚本")
        sys.exit(1)
    
    # 创建结果目录
    Path('figs').mkdir(exist_ok=True)
    Path('results_json').mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("批量并行运行脚本")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"参数范围: {PARAM_VALUES[0]} 到 {PARAM_VALUES[-1]}, 步长 0.1")
    print(f"每个任务: {len(PARAM_VALUES)} × {len(PARAM_VALUES)} = {len(PARAM_VALUES)**2} 个参数组合")
    print(f"总计: 4 个任务 × {len(PARAM_VALUES)**2} = {4 * len(PARAM_VALUES)**2} 个运行")
    print(f"并行进程数: {MAX_WORKERS}")
    print("="*70)
    
    all_completed = 0
    all_failed = 0
    
    # 任务1：估计 Fecundity
    print("\n--- 任务 1/4: 估计 Fecundity ---")
    tasks1 = task1_estimate_fecundity()
    c1, f1 = run_tasks_in_parallel(tasks1)
    all_completed += c1
    all_failed += f1
    
    # 任务2：估计 Viability
    print("\n--- 任务 2/4: 估计 Viability ---")
    tasks2 = task2_estimate_viability()
    c2, f2 = run_tasks_in_parallel(tasks2)
    all_completed += c2
    all_failed += f2
    
    # 任务3：模型选择 Fecundity
    print("\n--- 任务 3/4: 模型选择 Fecundity ---")
    tasks3 = task3_model_comparison_fecundity()
    c3, f3 = run_tasks_in_parallel(tasks3)
    all_completed += c3
    all_failed += f3
    
    # 任务4：模型选择 Viability
    print("\n--- 任务 4/4: 模型选择 Viability ---")
    tasks4 = task4_model_comparison_viability()
    c4, f4 = run_tasks_in_parallel(tasks4)
    all_completed += c4
    all_failed += f4
    
    # 输出最终统计
    print("\n" + "="*70)
    print("运行完成")
    print("="*70)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n任务统计:")
    print(f"  任务 1 (估计 Fecundity): {c1} 成功, {f1} 失败")
    print(f"  任务 2 (估计 Viability): {c2} 成功, {f2} 失败")
    print(f"  任务 3 (模型选择 Fecundity): {c3} 成功, {f3} 失败")
    print(f"  任务 4 (模型选择 Viability): {c4} 成功, {f4} 失败")
    print(f"\n总计: {all_completed} 成功, {all_failed} 失败")
    print(f"总任务数: {all_completed + all_failed}")
    print(f"成功率: {all_completed / (all_completed + all_failed) * 100:.1f}%")
    
    print(f"\n输出位置:")
    print(f"  图片: figs/")
    print(f"  JSON: results_json/")
    print("="*70 + "\n")
    
    return 0 if all_failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())

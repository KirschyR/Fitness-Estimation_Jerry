#!/usr/bin/env python3
"""
检查批量运行中哪些参数组合缺失

用法:
  python check_missing_runs.py
"""

import os
import json
from pathlib import Path
from collections import defaultdict

PARAM_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def parse_dirname(dirname):
    """从目录名提取参数"""
    parts = dirname.split('_')
    
    # 对于 est_fec_f0.3_m0.4_seed1234 格式
    if dirname.startswith('est_fec_'):
        try:
            f_val = float(parts[2][1:])  # 'f0.3' -> 0.3
            m_val = float(parts[3][1:])  # 'm0.4' -> 0.4
            return ('est_fec', f_val, m_val)
        except:
            return None
    
    elif dirname.startswith('est_vib_'):
        try:
            f_val = float(parts[2][1:])
            m_val = float(parts[3][1:])
            return ('est_vib', f_val, m_val)
        except:
            return None
    
    return None


def check_json_params(json_dir):
    """从 JSON 文件提取参数 - model_comparison 的 JSON 在时间戳目录中"""
    mc_fec_params = set()
    mc_vib_params = set()
    
    if not os.path.exists(json_dir):
        return mc_fec_params, mc_vib_params
    
    for item in Path(json_dir).iterdir():
        # est_* 目录是 pmcmc_inference 的，跳过
        if item.name.startswith('est_'):
            continue
        
        # 时间戳目录（如 20260107-073415）是 model_comparison 的
        if item.is_dir():
            json_file = item / 'model_comparison_results.json'
            if json_file.exists():
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                        true_vib = data['config']['true_viability']
                        true_fec = data['config']['true_fecundity']
                        
                        # 如果 fecundity=1.0，说明是 viability 变化
                        if abs(true_fec - 1.0) < 0.01:
                            mc_vib_params.add((true_vib, true_vib))
                        # 如果 viability=1.0，说明是 fecundity 变化
                        elif abs(true_vib - 1.0) < 0.01:
                            mc_fec_params.add((true_fec, true_fec))
                except:
                    continue
    
    return mc_fec_params, mc_vib_params


def main():
    print("\n" + "="*70)
    print("检查批量运行结果 - 找出缺失的参数组合")
    print("="*70)
    
    # 期望的所有组合
    all_combinations = [(f, m) for f in PARAM_VALUES for m in PARAM_VALUES]
    total_per_task = len(all_combinations)
    
    print(f"\n每个任务期望运行数: {total_per_task}")
    print(f"期望总运行数: {total_per_task * 4} (4个任务)\n")
    
    # 检查图片目录 - est_* 是参数估计
    figs_dir = Path('pmcmcres/figs')
    found_combinations = {
        'est_fec': set(),
        'est_vib': set(),
        'mc_fec': set(),
        'mc_vib': set()
    }
    
    if figs_dir.exists():
        for dirname in os.listdir(figs_dir):
            # est_* 目录是参数估计的图片
            parsed = parse_dirname(dirname)
            if parsed:
                task_type, f_val, m_val = parsed
                found_combinations[task_type].add((f_val, m_val))
    
    # 检查 JSON 目录 - 时间戳目录是 model_comparison
    json_dir = Path('pmcmcres/results_json')
    mc_fec_json, mc_vib_json = check_json_params(json_dir)
    
    # model_comparison 从 JSON 中获取（figs/seed_* 没有参数信息）
    found_combinations['mc_fec'] = mc_fec_json
    found_combinations['mc_vib'] = mc_vib_json
    
    # 打印统计
    print("="*70)
    print("任务完成统计")
    print("="*70)
    
    tasks = [
        ('任务1: 估计 Fecundity (est_fec)', 'est_fec'),
        ('任务2: 估计 Viability (est_vib)', 'est_vib'),
        ('任务3: 模型选择 Fecundity (mc_fec)', 'mc_fec'),
        ('任务4: 模型选择 Viability (mc_vib)', 'mc_vib')
    ]
    
    all_missing = {}
    
    for task_name, task_key in tasks:
        found = found_combinations[task_key]
        missing = set(all_combinations) - found
        all_missing[task_key] = missing
        
        print(f"\n{task_name}:")
        print(f"  完成: {len(found)}/{total_per_task}")
        print(f"  缺失: {len(missing)}")
        
        if len(missing) > 0:
            print(f"  完成率: {len(found)/total_per_task*100:.1f}%")
    
    # 详细列出缺失的组合
    print("\n" + "="*70)
    print("缺失的参数组合详情")
    print("="*70)
    
    for task_name, task_key in tasks:
        missing = all_missing[task_key]
        if len(missing) > 0:
            print(f"\n{task_name} - 缺失 {len(missing)} 个组合:")
            missing_sorted = sorted(list(missing))
            for f_val, m_val in missing_sorted:
                print(f"  female={f_val:.1f}, male={m_val:.1f}")
        else:
            print(f"\n{task_name}: ✓ 全部完成")
    
    # 生成重新运行的命令
    print("\n" + "="*70)
    print("重新运行缺失任务的命令")
    print("="*70)
    
    has_missing = False
    
    for task_name, task_key in tasks:
        missing = all_missing[task_key]
        if len(missing) > 0:
            has_missing = True
            print(f"\n# {task_name}")
            
            if task_key == 'est_fec':
                for f_val, m_val in sorted(list(missing)):
                    seed = 1234  # 可以改成随机数
                    print(f"python pmcmc_inference_multi.py --true_fecundity {f_val} --estimate_params female_fecundity male_fecundity --seed {seed} --fig_dir pmcmcres/figs --json_dir pmcmcres/results_json")
            
            elif task_key == 'est_vib':
                for f_val, m_val in sorted(list(missing)):
                    seed = 1234
                    print(f"python pmcmc_inference_multi.py --true_viability {f_val} --estimate_params female_viability male_viability --seed {seed} --fig_dir pmcmcres/figs --json_dir pmcmcres/results_json")
            
            elif task_key == 'mc_fec':
                for f_val, m_val in sorted(list(missing)):
                    seed = 1234
                    print(f"python quick_model_comparison.py --true_fecundity {f_val} --true_viability 1.0 --seed {seed} --json_dir pmcmcres/results_json")
            
            elif task_key == 'mc_vib':
                for f_val, m_val in sorted(list(missing)):
                    seed = 1234
                    print(f"python quick_model_comparison.py --true_viability {f_val} --true_fecundity 1.0 --seed {seed} --json_dir pmcmcres/results_json")
    
    if not has_missing:
        print("\n✓ 所有任务都已完成，没有缺失的参数组合！")
    
    print("\n" + "="*70)
    print("总结")
    print("="*70)
    total_found = sum(len(found_combinations[k]) for k in ['est_fec', 'est_vib', 'mc_fec', 'mc_vib'])
    total_missing = sum(len(all_missing[k]) for k in ['est_fec', 'est_vib', 'mc_fec', 'mc_vib'])
    print(f"总完成: {total_found}/{total_per_task * 4}")
    print(f"总缺失: {total_missing}")
    print(f"完成率: {total_found/(total_per_task*4)*100:.1f}%")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

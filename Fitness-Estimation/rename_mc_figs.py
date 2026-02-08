#!/usr/bin/env python3
"""
根据 results_json 中的参数信息，重命名 figs 中的 model_comparison 目录

用法:
  python rename_mc_figs.py
"""

import os
import json
from pathlib import Path
import shutil

def main():
    print("\n" + "="*70)
    print("重命名 Model Comparison 图片目录")
    print("="*70)
    
    json_dir = Path('pmcmcres/results_json')
    figs_dir = Path('pmcmcres/figs')
    
    if not json_dir.exists():
        print("错误: pmcmcres/results_json 不存在")
        return 1
    
    if not figs_dir.exists():
        print("错误: pmcmcres/figs 不存在")
        return 1
    
    # 收集所有 model_comparison 的映射关系
    # seed -> (true_fec, true_vib, timestamp)
    seed_to_params = {}
    
    print("\n第1步: 扫描 results_json 目录...")
    for item in json_dir.iterdir():
        if not item.is_dir():
            continue
        
        # 跳过 est_* 目录（参数估计的）
        if item.name.startswith('est_'):
            continue
        
        # 时间戳目录包含 model_comparison 结果
        json_file = item / 'model_comparison_results.json'
        if json_file.exists():
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    seed = data['config']['seed']
                    true_vib = data['config']['true_viability']
                    true_fec = data['config']['true_fecundity']
                    
                    # 如果同一个 seed 有多个记录，保留最新的（按目录名排序）
                    if seed not in seed_to_params:
                        seed_to_params[seed] = (true_fec, true_vib, item.name)
                    else:
                        # 比较时间戳，保留更新的
                        if item.name > seed_to_params[seed][2]:
                            seed_to_params[seed] = (true_fec, true_vib, item.name)
                    
                    print(f"  找到: seed={seed}, fec={true_fec:.1f}, vib={true_vib:.1f} ({item.name})")
            except Exception as e:
                print(f"  跳过 {item.name}: {e}")
                continue
    
    print(f"\n总共找到 {len(seed_to_params)} 个不同的 seed")
    
    # 扫描 figs 目录，找到需要重命名的 seed_* 目录
    print("\n第2步: 扫描 figs 目录中的 seed_* 目录...")
    rename_list = []
    
    for item in figs_dir.iterdir():
        if not item.is_dir():
            continue
        
        # 只处理 seed_* 格式的目录
        if item.name.startswith('seed_'):
            try:
                seed = int(item.name.split('_')[1])
                if seed in seed_to_params:
                    true_fec, true_vib, timestamp = seed_to_params[seed]
                    
                    # 判断是 fecundity 还是 viability 变化
                    if abs(true_vib - 1.0) < 0.01:
                        # viability=1.0，说明是 fecundity 变化
                        new_name = f"mc_fec_f{true_fec:.1f}_m{true_fec:.1f}_seed{seed}"
                    elif abs(true_fec - 1.0) < 0.01:
                        # fecundity=1.0，说明是 viability 变化
                        new_name = f"mc_vib_f{true_vib:.1f}_m{true_vib:.1f}_seed{seed}"
                    else:
                        # 两个都变化（不应该出现在我们的任务中）
                        print(f"  警告: {item.name} 的参数不符合预期 (fec={true_fec}, vib={true_vib})")
                        continue
                    
                    new_path = figs_dir / new_name
                    
                    # 检查目标目录是否已存在
                    if new_path.exists():
                        print(f"  跳过 {item.name}: 目标已存在 {new_name}")
                    else:
                        rename_list.append((item, new_path, new_name))
                        print(f"  计划: {item.name} -> {new_name}")
                else:
                    print(f"  跳过 {item.name}: 在 JSON 中找不到对应的参数信息")
            except Exception as e:
                print(f"  跳过 {item.name}: {e}")
                continue
    
    if not rename_list:
        print("\n没有需要重命名的目录")
        return 0
    
    # 询问确认
    print("\n" + "="*70)
    print(f"将重命名 {len(rename_list)} 个目录")
    print("="*70)
    
    response = input("\n确认执行重命名? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("取消操作")
        return 0
    
    # 执行重命名
    print("\n第3步: 执行重命名...")
    success_count = 0
    fail_count = 0
    
    for old_path, new_path, new_name in rename_list:
        try:
            old_path.rename(new_path)
            print(f"  ✓ {old_path.name} -> {new_name}")
            success_count += 1
        except Exception as e:
            print(f"  ✗ 失败 {old_path.name}: {e}")
            fail_count += 1
    
    # 总结
    print("\n" + "="*70)
    print("重命名完成")
    print("="*70)
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print("="*70 + "\n")
    
    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())

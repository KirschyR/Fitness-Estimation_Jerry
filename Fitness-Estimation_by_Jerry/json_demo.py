#!/usr/bin/env python3
"""
JSON 输出格式演示脚本

这个脚本展示了如何从生成的 JSON 文件中读取和重建人类可读的信息。
"""

import json
import sys
from pathlib import Path


def demonstrate_pmcmc_json(json_file):
    """演示如何读取和使用 pmcmc_inference_multi.py 的 JSON 输出"""
    
    print("\n" + "="*70)
    print("PMCMC 推断 JSON 输出演示")
    print("="*70)
    
    if not Path(json_file).exists():
        print(f"❌ 文件不存在: {json_file}")
        print("\n使用方法:")
        print("  python json_demo.py <json_file_path>")
        print("\n示例:")
        print("  python json_demo.py results_json/20260107-120000/inference_results.json")
        return False
    
    with open(json_file, 'r') as f:
        data = json.loads(f.read().strip())
    
    # 显示配置信息
    print("\n【配置信息】")
    config = data['config']
    print(f"  种子 (Seed):           {config['seed']}")
    print(f"  PMCMC 迭代次数:        {config['n_iterations']}")
    print(f"  粒子滤波粒子数:        {config['n_particles']}")
    print(f"  估计的参数:            {', '.join(config['estimate_params'])}")
    print(f"  模拟代数:              {config['n_generations']}")
    print(f"  有效种群大小:          {config['effective_population_size']}")
    print(f"  观测噪声 (sigma):      {config['obs_sigma']}")
    
    # 显示参数估计结果
    print("\n【参数估计结果】")
    print("-" * 70)
    
    for param_name, param_data in data['parameters'].items():
        true_val = param_data['true_value']
        mean = param_data['posterior_mean']
        std = param_data['posterior_std']
        ci_lower = param_data['ci_lower']
        ci_upper = param_data['ci_upper']
        in_ci = param_data['in_ci']
        
        print(f"\n参数: {param_name}")
        print(f"  真实值:                {true_val:.6f}")
        print(f"  后验估计 (均值±标差):  {mean:.6f} ± {std:.6f}")
        print(f"  95% 置信区间:          [{ci_lower:.6f}, {ci_upper:.6f}]")
        print(f"  真实值在 CI 内:        {'✓ 是' if in_ci else '✗ 否'}")
        
        # 计算偏差
        bias = mean - true_val
        relative_error = abs(bias) / true_val * 100
        print(f"  偏差:                  {bias:+.6f} ({relative_error:+.2f}%)")
    
    # 显示输出路径
    print("\n【输出文件路径】")
    print(f"  图表目录:              {data['output_paths']['figures_dir']}")
    print(f"  完整分析图:            {data['output_paths']['full_figure']}")
    
    return True


def demonstrate_model_comparison_json(json_file):
    """演示如何读取和使用 quick_model_comparison.py 的 JSON 输出"""
    
    print("\n" + "="*70)
    print("模型比较 JSON 输出演示")
    print("="*70)
    
    if not Path(json_file).exists():
        print(f"❌ 文件不存在: {json_file}")
        print("\n使用方法:")
        print("  python json_demo.py <json_file_path>")
        print("\n示例:")
        print("  python json_demo.py results_json/20260107-120000/model_comparison_results.json")
        return False
    
    with open(json_file, 'r') as f:
        data = json.loads(f.read().strip())
    
    # 显示配置
    print("\n【配置信息】")
    config = data['config']
    print(f"  种子:                  {config['seed']}")
    print(f"  迭代次数:              {config['n_iterations']}")
    print(f"  粒子数:                {config['n_particles']}")
    print(f"  真实可活性:            {config['true_viability']:.4f}")
    print(f"  真实生育力:            {config['true_fecundity']:.4f}")
    
    # 显示边际似然
    print("\n【边际似然 (log 尺度)】")
    print("-" * 70)
    ml = data['marginal_likelihoods']
    best_model = max(ml, key=ml.get)
    for model_name in sorted(ml.keys()):
        value = ml[model_name]
        marker = " ← 最佳模型" if model_name == best_model else ""
        print(f"  {model_name:25} : {value:10.2f}{marker}")
    
    # 显示贝叶斯因子
    print("\n【贝叶斯因子 (证据强度)】")
    print("-" * 70)
    bf = data['bayes_factors']
    for pair_name in sorted(bf.keys()):
        value = bf[pair_name]
        m1, m2 = pair_name.rsplit('_vs_', 1)
        
        if value > 1:
            interpretation = f"支持 {m1} ({value:.2f}倍)"
            strength = ""
            if value > 30:
                strength = " [非常强的证据]"
            elif value > 10:
                strength = " [强证据]"
            elif value > 3:
                strength = " [中等证据]"
        else:
            interpretation = f"支持 {m2} ({1/value:.2f}倍)"
            inv_value = 1/value
            strength = ""
            if inv_value > 30:
                strength = " [非常强的证据]"
            elif inv_value > 10:
                strength = " [强证据]"
            elif inv_value > 3:
                strength = " [中等证据]"
        
        print(f"  {m1} vs {m2}")
        print(f"    BF = {value:8.2f}  →  {interpretation}{strength}")
    
    # 显示参数估计
    print("\n【各模型参数估计】")
    print("-" * 70)
    estimates = data['model_estimates']
    for model_name in sorted(estimates.keys()):
        print(f"\n{model_name}:")
        params = estimates[model_name]
        for param_name in sorted(params.keys()):
            mean = params[param_name]['mean']
            std = params[param_name]['std']
            print(f"  {param_name:20} : {mean:.6f} ± {std:.6f}")
    
    return True


def main():
    if len(sys.argv) < 2:
        print("JSON 输出文件格式演示")
        print("\n使用方法:")
        print("  python json_demo.py <json_file_path>")
        print("\n支持的文件类型:")
        print("  - inference_results.json (pmcmc_inference_multi.py 输出)")
        print("  - model_comparison_results.json (quick_model_comparison.py 输出)")
        print("\n示例:")
        print("  # PMCMC 推断结果")
        print("  python json_demo.py results_json/20260107-120000/inference_results.json")
        print("\n  # 模型比较结果")
        print("  python json_demo.py results_json/20260107-120000/model_comparison_results.json")
        return 1
    
    json_file = sys.argv[1]
    json_file = Path(json_file)
    
    if not json_file.exists():
        print(f"❌ 错误: 文件不存在 - {json_file}")
        return 1
    
    # 根据文件名判断类型
    if 'inference_results' in json_file.name:
        success = demonstrate_pmcmc_json(str(json_file))
    elif 'model_comparison' in json_file.name:
        success = demonstrate_model_comparison_json(str(json_file))
    else:
        # 尝试读取文件来判断内容
        with open(json_file, 'r') as f:
            data = json.loads(f.read().strip())
        
        if 'parameters' in data:
            success = demonstrate_pmcmc_json(str(json_file))
        elif 'marginal_likelihoods' in data:
            success = demonstrate_model_comparison_json(str(json_file))
        else:
            print(f"❌ 错误: 无法识别 JSON 文件格式")
            return 1
    
    if success:
        print("\n" + "="*70 + "\n")
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())

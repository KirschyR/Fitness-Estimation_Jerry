"""
PMCMC 模型选择演示
==================

对比三个模型：
1. Model V:  仅估计 Viability Selection
2. Model F:  仅估计 Fecundity Selection
3. Model VF: 同时估计 Viability 和 Fecundity Selection
"""

import numpy as np
import sys
from pmcmc_inference_multi import (
    generate_data, run_inference, analyze_and_plot
)
from model_selection import compute_bayes_factors, plot_model_comparison, interpret_bayes_factor

def run_model_comparison(true_viability=0.75, true_fecundity=1.0, 
                        n_iter=500, n_particles=200, seed=42):
    """运行三个模型的PMCMC推断并进行贝叶斯模型比较"""
    
    print("\n" + "="*70)
    print("PMCMC 模型选择：Fecundity Selection vs Viability Selection")
    print("="*70)
    
    # 生成真实数据
    print(f"\n生成数据：viability={true_viability:.2f}, fecundity={true_fecundity:.2f}")
    pop, observations, obs_rule, obs_names = generate_data(
        true_viability, true_fecundity, true_viability, true_fecundity,
        n_gen=15, ne=300, seed=seed
    )
    
    results = {}
    true_theta_full = np.array([true_viability, true_fecundity, true_viability, true_fecundity])
    
    # ===== 模型 V: 仅估计 Viability =====
    print("\n" + "-"*70)
    print("运行 Model V: 仅估计 Viability")
    print("-"*70)
    result_v = run_inference(
        pop, observations, obs_rule, true_theta_full,
        estimate_params=['female_viability', 'male_viability'],
        n_iter=n_iter, n_particles=n_particles, seed=seed
    )
    results['Model V (Viability)'] = result_v
    
    # ===== 模型 F: 仅估计 Fecundity =====
    print("\n" + "-"*70)
    print("运行 Model F: 仅估计 Fecundity")
    print("-"*70)
    result_f = run_inference(
        pop, observations, obs_rule, true_theta_full,
        estimate_params=['female_fecundity', 'male_fecundity'],
        n_iter=n_iter, n_particles=n_particles, seed=seed + 10
    )
    results['Model F (Fecundity)'] = result_f
    
    # ===== 模型 VF: 同时估计 Viability 和 Fecundity =====
    print("\n" + "-"*70)
    print("运行 Model VF: 同时估计 Viability 和 Fecundity")
    print("-"*70)
    result_vf = run_inference(
        pop, observations, obs_rule, true_theta_full,
        estimate_params=['female_viability', 'female_fecundity', 'male_viability', 'male_fecundity'],
        n_iter=n_iter, n_particles=n_particles, seed=seed + 20
    )
    results['Model VF (Both)'] = result_vf
    
    # ===== 模型比较 =====
    print("\n" + "="*70)
    log_marglik, bayes_factors = compute_bayes_factors(results, method='harmonic_mean')
    
    print("\n边际似然（对数）：")
    for model in sorted(results.keys()):
        print(f"  {model}: {log_marglik[model]:.2f}")
    
    print("\n贝叶斯因子解释：")
    print("  BF > 1: 支持第一个模型")
    print("  BF > 3: 中等证据支持")
    print("  BF > 10: 强证据支持")
    print("  BF > 30: 非常强的证据支持")
    
    # 关键对比
    bf_v_vs_f = bayes_factors[('Model V (Viability)', 'Model F (Fecundity)')]
    bf_v_vs_vf = bayes_factors[('Model V (Viability)', 'Model VF (Both)')]
    bf_f_vs_vf = bayes_factors[('Model F (Fecundity)', 'Model VF (Both)')]
    
    print(f"\n关键对比：")
    print(f"  Model V vs Model F:  BF = {bf_v_vs_f:.2f}  ({interpret_bayes_factor(bf_v_vs_f)})")
    if bf_v_vs_f > 1:
        print(f"    → Viability Selection 更优")
    else:
        print(f"    → Fecundity Selection 更优")
    
    print(f"\n  Model V vs Model VF: BF = {bf_v_vs_vf:.2f}  ({interpret_bayes_factor(bf_v_vs_vf)})")
    if bf_v_vs_vf > 1:
        print(f"    → 简单模型（仅Viability）足够")
    else:
        print(f"    → 复杂模型（同时Viability和Fecundity）必要")
    
    print(f"\n  Model F vs Model VF: BF = {bf_f_vs_vf:.2f}  ({interpret_bayes_factor(bf_f_vs_vf)})")
    if bf_f_vs_vf > 1:
        print(f"    → 简单模型（仅Fecundity）足够")
    else:
        print(f"    → 复杂模型（同时Viability和Fecundity）必要")
    
    # 绘图
    print("\n绘制模型比较图...")
    plot_model_comparison(results, true_model=None, output_file='model_comparison.png')
    
    return results, log_marglik, bayes_factors


if __name__ == '__main__':
    # 场景1: 真实是纯 Viability Selection（fecundity=1.0）
    print("\n【场景1】数据由纯 Viability Selection 生成")
    results1, logml1, bf1 = run_model_comparison(
        true_viability=0.75, true_fecundity=1.0,
        n_iter=300, n_particles=150, seed=42
    )
    
    # 场景2: 真实是纯 Fecundity Selection（viability=1.0）
    print("\n\n【场景2】数据由纯 Fecundity Selection 生成")
    results2, logml2, bf2 = run_model_comparison(
        true_viability=1.0, true_fecundity=0.75,
        n_iter=300, n_particles=150, seed=123
    )
    
    # 场景3: 真实是同时有两种选择
    print("\n\n【场景3】数据由同时 Viability 和 Fecundity Selection 生成")
    results3, logml3, bf3 = run_model_comparison(
        true_viability=0.75, true_fecundity=0.85,
        n_iter=300, n_particles=150, seed=999
    )
    
    print("\n" + "="*70)
    print("所有场景已完成")
    print("="*70)

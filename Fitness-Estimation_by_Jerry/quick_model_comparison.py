"""
快速模型选择对比
===============

用法:
  python quick_model_comparison.py --true_viability 0.75 --true_fecundity 1.0
  python quick_model_comparison.py --true_viability 1.0 --true_fecundity 0.75
  python quick_model_comparison.py --true_viability 0.75 --true_fecundity 0.85
"""

import numpy as np
import argparse
import os
from datetime import datetime
import json
import matplotlib.pyplot as plt
from pmcmc_inference_multi import generate_data, run_inference
from model_selection import compute_bayes_factors, plot_model_comparison


def main():
    parser = argparse.ArgumentParser(description='PMCMC 模型选择：Viability vs Fecundity Selection')
    # 支持按性别分开的参数
    parser.add_argument('--true_female_viability', type=float, default=None)
    parser.add_argument('--true_male_viability', type=float, default=None)
    parser.add_argument('--true_female_fecundity', type=float, default=None)
    parser.add_argument('--true_male_fecundity', type=float, default=None)
    # 兼容旧的参数（female=male）
    parser.add_argument('--true_viability', type=float, default=0.75)
    parser.add_argument('--true_fecundity', type=float, default=1.0)
    parser.add_argument('--n_gen', type=int, default=20, help='Number of generations to simulate')
    parser.add_argument('--n_e', type=int, default=300, help='Effective population size')
    parser.add_argument('--n_iter', type=int, default=500)
    parser.add_argument('--n_particles', type=int, default=250)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--json_dir', type=str, default='results_json', help='Directory to store JSON results')
    parser.add_argument('--fig_dir', type=str, default='figs', help='Directory to store figure outputs')
    
    args = parser.parse_args()
    
    # 处理参数：优先使用按性别分开的参数
    true_f_vib = args.true_female_viability if args.true_female_viability is not None else args.true_viability
    true_m_vib = args.true_male_viability if args.true_male_viability is not None else args.true_viability
    true_f_fec = args.true_female_fecundity if args.true_female_fecundity is not None else args.true_fecundity
    true_m_fec = args.true_male_fecundity if args.true_male_fecundity is not None else args.true_fecundity
    
    print("\n" + "="*70)
    print("PMCMC 模型选择")
    print("="*70)
    print(f"\n配置：")
    print(f"  Seed:       {args.seed}")
    print(f"  Generations: {args.n_gen}")
    print(f"  Population Size: {args.n_e}")
    print(f"  Iterations: {args.n_iter}")
    print(f"  Particles:  {args.n_particles}")
    print(f"\n真实参数：")
    print(f"  Female Viability: {true_f_vib:.2f}")
    print(f"  Male Viability:   {true_m_vib:.2f}")
    print(f"  Female Fecundity: {true_f_fec:.2f}")
    print(f"  Male Fecundity:   {true_m_fec:.2f}")
    
    # 生成数据
    pop, observations, obs_rule, obs_names = generate_data(
        true_f_vib, true_f_fec,
        true_m_vib, true_m_fec,
        n_gen=args.n_gen, ne=args.n_e, seed=args.seed
    )
    
    true_theta = np.array([true_f_vib, true_f_fec, true_m_vib, true_m_fec])
    
    results = {}
    
    # 模型 V: 仅 Viability
    print("\n运行 Model V: 仅估计 Viability...")
    result_v = run_inference(pop, observations, obs_rule, true_theta,
                            estimate_params=['female_viability', 'male_viability'],
                            n_iter=args.n_iter, n_particles=args.n_particles,
                            seed=args.seed)
    results['V (Viability only)'] = result_v
    
    # 模型 F: 仅 Fecundity
    print("\n运行 Model F: 仅估计 Fecundity...")
    result_f = run_inference(pop, observations, obs_rule, true_theta,
                            estimate_params=['female_fecundity', 'male_fecundity'],
                            n_iter=args.n_iter, n_particles=args.n_particles,
                            seed=args.seed + 10)
    results['F (Fecundity only)'] = result_f
    
    # 模型 VF: 同时估计
    print("\n运行 Model VF: 同时估计 Viability 和 Fecundity...")
    result_vf = run_inference(pop, observations, obs_rule, true_theta,
                             estimate_params=['female_viability', 'female_fecundity',
                                             'male_viability', 'male_fecundity'],
                             n_iter=args.n_iter, n_particles=args.n_particles,
                             seed=args.seed + 20)
    results['VF (Both)'] = result_vf
    
    # 模型比较
    print("\n" + "="*70)
    log_marglik, bayes_factors = compute_bayes_factors(results)
    
    print("\n边际似然 (log scale):")
    for m in ['V (Viability only)', 'F (Fecundity only)', 'VF (Both)']:
        print(f"  {m:25} : {log_marglik[m]:8.2f}")
    
    bf_v_f = bayes_factors[('V (Viability only)', 'F (Fecundity only)')]
    bf_v_vf = bayes_factors[('V (Viability only)', 'VF (Both)')]
    bf_f_vf = bayes_factors[('F (Fecundity only)', 'VF (Both)')]
    
    print("\n贝叶斯因子（证据强度）:")
    print(f"  V vs F  : BF = {bf_v_f:7.2f}  →  ", end="")
    if bf_v_f > 1:
        print(f"Viability Selection 更优 ({bf_v_f:.0f}倍)")
    else:
        print(f"Fecundity Selection 更优 ({1/bf_v_f:.0f}倍)")
    
    print(f"  V vs VF : BF = {bf_v_vf:7.2f}  →  ", end="")
    if bf_v_vf > 1:
        print(f"简单模型(V)足够")
    else:
        print(f"需要复杂模型(VF)")
    
    print(f"  F vs VF : BF = {bf_f_vf:7.2f}  →  ", end="")
    if bf_f_vf > 1:
        print(f"简单模型(F)足够")
    else:
        print(f"需要复杂模型(VF)")
    
    print("\n解释:")
    print("  BF > 1  : 支持第一个模型")
    print("  BF > 3  : 中等证据")
    print("  BF > 10 : 强证据")
    print("  BF > 30 : 非常强的证据")
    
    # 最终结论
    print("\n" + "="*70)
    print("结论:")
    if bf_v_f > 3:
        print("✓ 证据强烈支持 Viability Selection（而非 Fecundity）")
    elif bf_v_f < 1/3:
        print("✓ 证据强烈支持 Fecundity Selection（而非 Viability）")
    else:
        print("✓ 两种选择都有证据支持（模型选择不确定）")
    
    # 生成包含参数信息的 run_name
    # 判断主要变化的是 viability 还是 fecundity
    if abs(true_f_fec - 1.0) < 0.01 and abs(true_m_fec - 1.0) < 0.01:
        # fecundity = 1.0，说明主要是 viability 变化
        run_name = f"mc_vib_f{true_f_vib:.1f}_m{true_m_vib:.1f}_seed{args.seed}"
    elif abs(true_f_vib - 1.0) < 0.01 and abs(true_m_vib - 1.0) < 0.01:
        # viability = 1.0，说明主要是 fecundity 变化
        run_name = f"mc_fec_f{true_f_fec:.1f}_m{true_m_fec:.1f}_seed{args.seed}"
    else:
        # 两者都变化
        run_name = f"mc_both_fv{true_f_vib:.1f}_ff{true_f_fec:.1f}_mv{true_m_vib:.1f}_mf{true_m_fec:.1f}_seed{args.seed}"
    
    # 绘图 - 保存到 fig_dir
    fig_output_dir = os.path.join(args.fig_dir, run_name)
    os.makedirs(fig_output_dir, exist_ok=True)
    fig_output_path = os.path.join(fig_output_dir, 'model_comparison_result.png')
    plot_model_comparison(results, output_file=fig_output_path, run_name=run_name)
    print(f"✓ 图片已保存: {fig_output_path}")
    
    # Save JSON results - 使用相同的 run_name 命名
    json_out_dir = os.path.join(args.json_dir, run_name)
    os.makedirs(json_out_dir, exist_ok=True)
    
    # Collect model comparison results
    json_data = {
        'config': {
            'seed': args.seed,
            'n_generations': args.n_gen,
            'effective_population_size': args.n_e,
            'n_iterations': args.n_iter,
            'n_particles': args.n_particles,
            'true_female_viability': true_f_vib,
            'true_male_viability': true_m_vib,
            'true_female_fecundity': true_f_fec,
            'true_male_fecundity': true_m_fec
        },
        'marginal_likelihoods': {model: float(ml) for model, ml in log_marglik.items()},
        'bayes_factors': {f"{m1}_vs_{m2}": float(bf) for (m1, m2), bf in bayes_factors.items()},
        'model_estimates': {}
    }
    
    # Add parameter estimates for each model
    for model, result in results.items():
        theta_mean = np.mean(result.theta_chain, axis=0)
        theta_std = np.std(result.theta_chain, axis=0)
        param_list = ['female_viability', 'male_viability'] if 'Viability' in model else (['female_fecundity', 'male_fecundity'] if 'Fecundity' in model else ['female_viability', 'female_fecundity', 'male_viability', 'male_fecundity'])
        json_data['model_estimates'][model] = {
            param_list[i]: {'mean': float(theta_mean[i]), 'std': float(theta_std[i])}
            for i in range(len(param_list))
        }
    
    json_file = os.path.join(json_out_dir, 'model_comparison_results.json')
    
    # 添加输出路径信息
    json_data['output_paths'] = {
        'figure': fig_output_path,
        'json': json_file
    }
    
    with open(json_file, 'w') as f:
        f.write(json.dumps(json_data, indent=None) + '\n')
    print(f"✓ JSON结果已保存: {json_file}")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

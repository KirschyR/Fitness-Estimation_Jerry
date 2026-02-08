"""
PMCMC 模型选择：区分 Fecundity Selection vs Viability Selection
==================================================================

使用贝叶斯因子 (Bayes Factor) 比较不同的选择方式
"""

import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import logsumexp

def estimate_marginal_likelihood_harmonic_mean(loglik_chain):
    """
    使用调和平均估计边际似然（简单但对尾部敏感）
    
    M̂ = 1 / E[1/L(θ)]  =>  log M̂ ≈ -log E[exp(-loglik)]
    
    Args:
        loglik_chain: shape (n_samples,) 的对数似然向量
    
    Returns:
        float: 边际似然的对数估计
    """
    # logsumexp 数值稳定的计算 log(sum(exp(x)))
    return -logsumexp(-loglik_chain) + np.log(len(loglik_chain))


def estimate_marginal_likelihood_thermodynamic(loglik_chain, method='simple'):
    """
    使用热力学积分估计边际似然（更稳健）
    
    简单版本：使用 loglik 链的均值和方差进行贝叶斯高斯近似
    
    Args:
        loglik_chain: shape (n_samples,) 的对数似然向量
        method: 'simple', 'gaussian', 'laplace'
    
    Returns:
        float: 边际似然的对数估计
    """
    if method == 'simple':
        # 使用后验均值（假设后验集中在最大值附近）
        return np.mean(loglik_chain)
    
    elif method == 'gaussian':
        # 高斯近似：log M ≈ log L̂ - k/2 * log(n)
        # 其中 k 是参数维度，n 是样本数
        k = 1  # 简化版本，假设 k=1
        n = len(loglik_chain)
        loglik_mean = np.mean(loglik_chain)
        bic_correction = (k / 2) * np.log(n)
        return loglik_mean - bic_correction
    
    else:
        return np.mean(loglik_chain)


def compute_bayes_factors(results_dict, method='harmonic_mean'):
    """
    计算多个模型之间的贝叶斯因子
    
    Args:
        results_dict: {model_name: result_object} 字典
        method: 'harmonic_mean', 'thermodynamic'
    
    Returns:
        dict: {model_name: log_marginal_likelihood}
        dict: {(m1, m2): bayes_factor} 成对贝叶斯因子
    """
    log_marglik = {}
    
    for model_name, result in results_dict.items():
        if method == 'harmonic_mean':
            log_marglik[model_name] = estimate_marginal_likelihood_harmonic_mean(result.loglik_chain)
        elif method == 'thermodynamic':
            log_marglik[model_name] = estimate_marginal_likelihood_thermodynamic(result.loglik_chain)
        else:
            log_marglik[model_name] = np.mean(result.loglik_chain)
    
    # 计算成对贝叶斯因子
    bayes_factors = {}
    model_names = list(results_dict.keys())
    for i, m1 in enumerate(model_names):
        for m2 in model_names[i+1:]:
            log_bf = log_marglik[m1] - log_marglik[m2]
            bf = np.exp(log_bf)
            bayes_factors[(m1, m2)] = bf
            bayes_factors[(m2, m1)] = 1.0 / bf
    
    return log_marglik, bayes_factors


def interpret_bayes_factor(bf):
    """
    解释贝叶斯因子的强度
    
    按照 Kass & Raftery (1995) 标准：
    """
    abs_log_bf = np.log(bf)
    
    if abs_log_bf < 1:
        return "Weak"
    elif abs_log_bf < 3:
        return "Moderate"
    elif abs_log_bf < 5:
        return "Strong"
    else:
        return "Very Strong"


def plot_model_comparison(results_dict, true_model=None, output_file='model_comparison.png', output_dir='figs', run_name=None):
    """
    绘制模型选择结果，并将全图和各子图保存到独立文件夹
    
    Args:
        results_dict: {model_name: result_object}
        true_model: 真实的数据生成模型（可选）
        output_file: 输出文件名（将被存放在 output_dir/run_name 下）
        output_dir: 基础图像目录，默认 figs
        run_name: 自定义子目录名，默认使用时间戳
    """
    # 计算贝叶斯因子
    log_marglik, bayes_factors = compute_bayes_factors(results_dict, method='harmonic_mean')
    
    print("\n" + "="*70)
    print("模型选择：贝叶斯因子分析")
    print("="*70)
    
    print("\n边际似然 (log scale):")
    for model, logml in sorted(log_marglik.items()):
        marker = " ← 真实模型" if model == true_model else ""
        print(f"  {model}: {logml:.2f}{marker}")
    
    print("\n参数估计 (Posterior Mean ± Std):")
    for model, result in results_dict.items():
        theta_mean = np.mean(result.theta_chain, axis=0)
        theta_std = np.std(result.theta_chain, axis=0)
        
        # Format output based on model type
        if 'viability' in model.lower() and 'fecundity' not in model.lower():
            # V-only model
            if len(theta_mean) == 1:
                print(f"  {model}: {theta_mean[0]:.4f} ± {theta_std[0]:.4f}")
            elif len(theta_mean) == 2:
                print(f"  {model}: male={theta_mean[0]:.4f}±{theta_std[0]:.4f}, female={theta_mean[1]:.4f}±{theta_std[1]:.4f}")
            else:
                print(f"  {model}: {', '.join([f'{m:.4f}±{s:.4f}' for m, s in zip(theta_mean, theta_std)])}")
        
        elif 'fecundity' in model.lower() and 'viability' not in model.lower():
            # F-only model
            if len(theta_mean) == 1:
                print(f"  {model}: {theta_mean[0]:.4f} ± {theta_std[0]:.4f}")
            elif len(theta_mean) == 2:
                print(f"  {model}: male={theta_mean[0]:.4f}±{theta_std[0]:.4f}, female={theta_mean[1]:.4f}±{theta_std[1]:.4f}")
            else:
                print(f"  {model}: {', '.join([f'{m:.4f}±{s:.4f}' for m, s in zip(theta_mean, theta_std)])}")
        
        else:
            # VF model or other
            if len(theta_mean) == 1:
                print(f"  {model}: {theta_mean[0]:.4f} ± {theta_std[0]:.4f}")
            elif len(theta_mean) == 2:
                print(f"  {model}: param1={theta_mean[0]:.4f}±{theta_std[0]:.4f}, param2={theta_mean[1]:.4f}±{theta_std[1]:.4f}")
            elif len(theta_mean) == 4:
                print(f"  {model}: V_male={theta_mean[0]:.4f}±{theta_std[0]:.4f}, V_female={theta_mean[1]:.4f}±{theta_std[1]:.4f}, F_male={theta_mean[2]:.4f}±{theta_std[2]:.4f}, F_female={theta_mean[3]:.4f}±{theta_std[3]:.4f}")
            else:
                print(f"  {model}: {', '.join([f'{m:.4f}±{s:.4f}' for m, s in zip(theta_mean, theta_std)])}")
    
    print("\n贝叶斯因子 (M1 vs M2):")
    model_names = sorted(results_dict.keys())
    for i, m1 in enumerate(model_names):
        for m2 in model_names[i+1:]:
            bf = bayes_factors.get((m1, m2), 0)
            log_bf = np.log(bf)
            strength = interpret_bayes_factor(bf)
            if bf > 1:
                print(f"  {m1} vs {m2}: BF = {bf:.2f} (log BF = {log_bf:.2f}) [{strength}]")
                print(f"    → {m1} 支持度为 {bf:.2f}倍")
            else:
                print(f"  {m1} vs {m2}: BF = {1/bf:.2f} 倍于 {m1}")
    
    # Create plots
    run_id = run_name or datetime.now().strftime('%Y%m%d-%H%M%S')
    out_dir = os.path.join(output_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Log-likelihood chains
    ax = axes[0, 0]
    for model, result in results_dict.items():
        ax.plot(result.loglik_chain, alpha=0.7, label=model, linewidth=1.2)
    ax.set_title('Log-Likelihood Chains: Model Comparison', fontsize=12, fontweight='bold')
    ax.set_xlabel('MCMC Iteration', fontsize=11)
    ax.set_ylabel('log p(y|θ)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. Marginal likelihood bar chart
    ax = axes[0, 1]
    models = list(log_marglik.keys())
    marglik = list(log_marglik.values())
    colors = ['red' if m == true_model else 'blue' for m in models]
    bars = ax.bar(models, marglik, color=colors, alpha=0.6, edgecolor='black', linewidth=1.5)
    ax.set_title('Marginal Likelihood Estimate (Harmonic Mean)', fontsize=12, fontweight='bold')
    ax.set_ylabel('log p(y|Model)', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for bar, ml in zip(bars, marglik):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{ml:.1f}', ha='center', va='bottom', fontsize=10)
    
    # 3. Bayes factor heatmap
    ax = axes[1, 0]
    n_models = len(models)
    bf_matrix = np.zeros((n_models, n_models))
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i == j:
                bf_matrix[i, j] = 1.0
            elif i < j:
                bf_matrix[i, j] = bayes_factors.get((m1, m2), 1.0)
            else:
                bf_matrix[i, j] = 1.0 / bayes_factors.get((m2, m1), 1.0)
    
    im = ax.imshow(np.log(bf_matrix), cmap='RdBu_r', aspect='auto')
    ax.set_xticks(range(n_models))
    ax.set_yticks(range(n_models))
    ax.set_xticklabels(models, fontsize=10)
    ax.set_yticklabels(models, fontsize=10)
    ax.set_title('Bayes Factor Matrix (log scale)', fontsize=12, fontweight='bold')
    
    # Add values
    for i in range(n_models):
        for j in range(n_models):
            text = ax.text(j, i, f'{np.log(bf_matrix[i, j]):.1f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=ax, label='log(BF)')
    
    # 4. Posterior distributions for Viability and Fecundity
    ax = axes[1, 1]
    for model, result in results_dict.items():
        if 'viability' in model.lower() or 'V' in model:
            ax.hist(result.theta_chain[:, 0], bins=30, alpha=0.5, label=f'{model} (est.)', density=True)
        elif 'fecundity' in model.lower() or 'F' in model:
            if result.theta_chain.shape[1] > 0:
                ax.hist(result.theta_chain[:, 0], bins=30, alpha=0.5, label=f'{model} (est.)', density=True)
    ax.set_title('Posterior Distributions (Main Parameters)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Parameter Value', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()

    # 保存全图
    full_path = os.path.join(out_dir, os.path.basename(output_file))
    plt.savefig(full_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 模型比较图已保存: {full_path}")

    # 保存子图为独立图像
    def _slugify(text, fallback):
        base = text.strip() or fallback
        safe = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in base)
        return safe.strip('_') or fallback

    def _save_subplot_individually(ax, filepath, subplot_type='generic', figsize=(10, 8), **kwargs):
        """Save a single subplot as an independent figure by recreating from scratch"""
        fig_single = plt.figure(figsize=figsize)
        ax_single = fig_single.add_subplot(111)
        
        if subplot_type == 'lines':
            # Line plots: Log-likelihood chains
            for line in ax.get_lines():
                xdata = line.get_xdata()
                ydata = line.get_ydata()
                if len(xdata) > 0 and len(ydata) > 0:
                    ax_single.plot(xdata, ydata,
                                  color=line.get_color(), 
                                  linestyle=line.get_linestyle(),
                                  linewidth=line.get_linewidth() * 1.2,
                                  marker=line.get_marker(),
                                  markersize=line.get_markersize(), 
                                  alpha=line.get_alpha(),
                                  label=line.get_label() if line.get_label()[0] != '_' else '')
        
        elif subplot_type == 'bar':
            # Bar chart: Marginal likelihood
            model_names = kwargs.get('model_names', [])
            values = kwargs.get('values', [])
            colors_list = kwargs.get('colors', [])
            
            x_pos = np.arange(len(model_names))
            bars = ax_single.bar(x_pos, values, color=colors_list, 
                               alpha=0.6, edgecolor='black', linewidth=1.5)
            ax_single.set_xticks(x_pos)
            ax_single.set_xticklabels(model_names, fontsize=14)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax_single.text(bar.get_x() + bar.get_width()/2., height,
                              f'{val:.1f}', ha='center', va='bottom', fontsize=13)
        
        elif subplot_type == 'heatmap':
            # Heatmap: Bayes factor matrix
            data_matrix = kwargs.get('data_matrix', None)
            model_names = kwargs.get('model_names', [])
            
            if data_matrix is not None:
                im = ax_single.imshow(data_matrix, cmap='RdBu_r', aspect='auto')
                ax_single.set_xticks(range(len(model_names)))
                ax_single.set_yticks(range(len(model_names)))
                ax_single.set_xticklabels(model_names, fontsize=14)
                ax_single.set_yticklabels(model_names, fontsize=14)
                
                # Add text annotations
                for i in range(len(model_names)):
                    for j in range(len(model_names)):
                        text = ax_single.text(j, i, f'{data_matrix[i, j]:.1f}',
                                            ha="center", va="center", 
                                            color="black", fontsize=12)
                
                plt.colorbar(im, ax=ax_single, label='log(BF)')
        
        else:
            # Generic: Try to copy what we can
            for line in ax.get_lines():
                xdata = line.get_xdata()
                ydata = line.get_ydata()
                if len(xdata) > 0 and len(ydata) > 0:
                    ax_single.plot(xdata, ydata,
                                  color=line.get_color(), 
                                  linestyle=line.get_linestyle(),
                                  linewidth=line.get_linewidth(),
                                  marker=line.get_marker(),
                                  alpha=line.get_alpha(),
                                  label=line.get_label() if line.get_label()[0] != '_' else '')
        
        # Copy axis properties with increased font sizes
        ax_single.set_xlim(ax.get_xlim())
        ax_single.set_ylim(ax.get_ylim())
        ax_single.set_xlabel(ax.get_xlabel(), fontsize=16)
        ax_single.set_ylabel(ax.get_ylabel(), fontsize=16)
        ax_single.set_title(ax.get_title(), fontsize=18, fontweight='bold')
        ax_single.tick_params(labelsize=14)
        
        # Recreate legend if present
        legend = ax.get_legend()
        if legend and subplot_type == 'lines':
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax_single.legend(handles, labels, fontsize=14)
        
        # Add grid for line and bar plots
        if subplot_type in ['lines', 'bar']:
            gridlines = ax.xaxis.get_gridlines() + ax.yaxis.get_gridlines()
            if gridlines and any(line.get_visible() for line in gridlines):
                ax_single.grid(True, alpha=0.3)
        
        fig_single.tight_layout()
        fig_single.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig_single)

    # Prepare data for subplot saving
    models = list(log_marglik.keys())
    marglik = list(log_marglik.values())
    colors = ['red' if m == true_model else 'blue' for m in models]
    
    # BF matrix for heatmap
    n_models = len(models)
    bf_matrix = np.zeros((n_models, n_models))
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i == j:
                bf_matrix[i, j] = 1.0
            elif i < j:
                bf_matrix[i, j] = bayes_factors.get((m1, m2), 1.0)
            else:
                bf_matrix[i, j] = 1.0 / bayes_factors.get((m2, m1), 1.0)
    bf_matrix_log = np.log(bf_matrix)
    
    axes_all = list(fig.axes)
    subplot_idx = 1
    
    # Save subplot 1: Log-likelihood chains
    panel_path = os.path.join(out_dir, f'{subplot_idx:02d}_Log-Likelihood_Chains_Model_Comparison.png')
    _save_subplot_individually(axes_all[0], panel_path, subplot_type='lines')
    print(f"  └─ 子图已保存: {panel_path}")
    subplot_idx += 1
    
    # Save subplot 2: Marginal likelihood bar chart
    panel_path = os.path.join(out_dir, f'{subplot_idx:02d}_Marginal_Likelihood_Estimate.png')
    _save_subplot_individually(axes_all[1], panel_path, subplot_type='bar',
                             model_names=models, values=marglik, colors=colors)
    print(f"  └─ 子图已保存: {panel_path}")
    subplot_idx += 1
    
    # Save subplot 3: Bayes factor matrix
    panel_path = os.path.join(out_dir, f'{subplot_idx:02d}_Bayes_Factor_Matrix.png')
    _save_subplot_individually(axes_all[2], panel_path, subplot_type='heatmap',
                             data_matrix=bf_matrix_log, model_names=models)
    print(f"  └─ 子图已保存: {panel_path}")
    subplot_idx += 1
    
    # Save individual posterior distributions for each model
    print(f"  └─ 保存各模型后验分布:")
    for model, result in results_dict.items():
        n_params = result.theta_chain.shape[1]
        
        if n_params == 4:  # VF model with 4 parameters (female_v, female_f, male_v, male_f)
            fig_post = plt.figure(figsize=(16, 12))
            
            # Create layout: joint distributions on top row, marginals on bottom
            gs = fig_post.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
            
            # Parameter names and colors for all 4 parameters
            param_names = ['Female Viability', 'Female Fecundity', 'Male Viability', 'Male Fecundity']
            colors = ['steelblue', 'coral', 'seagreen', 'darkorange']
            
            # Row 1: Two joint distributions (span 2 cols each)
            ax_vib = fig_post.add_subplot(gs[0, 0:2])
            ax_vib.scatter(result.theta_chain[:, 0], result.theta_chain[:, 2], 
                          alpha=0.4, s=15, c='steelblue')
            
            # Add mean lines to viability joint distribution
            mean_vib_f = np.mean(result.theta_chain[:, 0])
            mean_vib_m = np.mean(result.theta_chain[:, 2])
            ax_vib.axvline(mean_vib_f, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            ax_vib.axhline(mean_vib_m, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            
            ax_vib.set_xlabel('Female Viability', fontsize=12)
            ax_vib.set_ylabel('Male Viability', fontsize=12)
            ax_vib.set_title('Viability: Joint Distribution', fontsize=13, fontweight='bold')
            ax_vib.tick_params(labelsize=10)
            ax_vib.grid(True, alpha=0.3)
            
            ax_fec = fig_post.add_subplot(gs[0, 2:4])
            ax_fec.scatter(result.theta_chain[:, 1], result.theta_chain[:, 3], 
                          alpha=0.4, s=15, c='coral')
            
            # Add mean lines to fecundity joint distribution
            mean_fec_f = np.mean(result.theta_chain[:, 1])
            mean_fec_m = np.mean(result.theta_chain[:, 3])
            ax_fec.axvline(mean_fec_f, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            ax_fec.axhline(mean_fec_m, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            
            ax_fec.set_xlabel('Female Fecundity', fontsize=12)
            ax_fec.set_ylabel('Male Fecundity', fontsize=12)
            ax_fec.set_title('Fecundity: Joint Distribution', fontsize=13, fontweight='bold')
            ax_fec.tick_params(labelsize=10)
            ax_fec.grid(True, alpha=0.3)
            
            # Row 2: Four marginal distributions
            for idx in range(4):
                ax = fig_post.add_subplot(gs[1, idx])
                ax.hist(result.theta_chain[:, idx], bins=30, alpha=0.7, 
                       color=colors[idx], edgecolor='black', linewidth=0.8, density=True)
                
                mean_val = np.mean(result.theta_chain[:, idx])
                std_val = np.std(result.theta_chain[:, idx])
                
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2)
                ax.set_title(f'{param_names[idx]}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Value', fontsize=10)
                ax.set_ylabel('Density', fontsize=10)
                ax.tick_params(labelsize=9)
                ax.grid(True, alpha=0.3)
                
                # Add text with mean ± std
                ax.text(0.98, 0.97, f'{mean_val:.4f} ± {std_val:.4f}',
                       transform=ax.transAxes, fontsize=9, verticalalignment='top',
                       horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            fig_post.suptitle(f'Posterior Distributions (All 4 Parameters): {model}', 
                             fontsize=14, fontweight='bold', y=0.995)
            fig_post.tight_layout()
            model_slug = _slugify(model, 'model')
            post_path = os.path.join(out_dir, f'{subplot_idx:02d}_Posterior_{model_slug}.png')
            fig_post.savefig(post_path, dpi=150, bbox_inches='tight')
            plt.close(fig_post)
            print(f"     • {post_path}")
            subplot_idx += 1
            
        elif n_params == 2:  # V model (2 viability) or F model (2 fecundity)
            fig_post = plt.figure(figsize=(16, 7))
            
            # Determine parameter names based on model type
            if 'viability' in model.lower() or 'V' in model:
                param_names = ['Female Viability', 'Male Viability']
                colors = ['steelblue', 'steelblue']
                param_label = 'Viability'
            elif 'fecundity' in model.lower() or 'F' in model:
                param_names = ['Female Fecundity', 'Male Fecundity']
                colors = ['coral', 'coral']
                param_label = 'Fecundity'
            else:
                param_names = ['Parameter 1', 'Parameter 2']
                colors = ['steelblue', 'coral']
                param_label = 'Parameter'
            
            # Joint distribution (2D scatter)
            ax_joint = fig_post.add_subplot(1, 3, 1)
            ax_joint.scatter(result.theta_chain[:, 0], result.theta_chain[:, 1], 
                            alpha=0.4, s=15, c=colors[0])
            
            # Add mean lines to joint distribution
            mean_val_0 = np.mean(result.theta_chain[:, 0])
            mean_val_1 = np.mean(result.theta_chain[:, 1])
            ax_joint.axvline(mean_val_0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            ax_joint.axhline(mean_val_1, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            
            ax_joint.set_xlabel(param_names[0], fontsize=12)
            ax_joint.set_ylabel(param_names[1], fontsize=12)
            ax_joint.set_title('Joint Distribution', fontsize=13, fontweight='bold')
            ax_joint.tick_params(labelsize=10)
            ax_joint.grid(True, alpha=0.3)
            
            # Marginal distributions
            for idx in range(2):
                ax = fig_post.add_subplot(1, 3, idx + 2)
                ax.hist(result.theta_chain[:, idx], bins=30, alpha=0.7, 
                       color=colors[idx], edgecolor='black', linewidth=0.8, density=True)
                
                mean_val = np.mean(result.theta_chain[:, idx])
                std_val = np.std(result.theta_chain[:, idx])
                
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2)
                ax.set_title(f'{param_names[idx]}', fontsize=13, fontweight='bold')
                ax.set_xlabel(f'{param_label} Value', fontsize=12)
                ax.set_ylabel('Density', fontsize=12)
                ax.tick_params(labelsize=10)
                ax.grid(True, alpha=0.3)
                
                # Add text with mean ± std
                ax.text(0.98, 0.97, f'{mean_val:.4f} ± {std_val:.4f}',
                       transform=ax.transAxes, fontsize=10, verticalalignment='top',
                       horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            fig_post.suptitle(f'Posterior Distributions: {model}', 
                             fontsize=14, fontweight='bold', y=0.98)
            fig_post.tight_layout()
            model_slug = _slugify(model, 'model')
            post_path = os.path.join(out_dir, f'{subplot_idx:02d}_Posterior_{model_slug}.png')
            fig_post.savefig(post_path, dpi=150, bbox_inches='tight')
            plt.close(fig_post)
            print(f"     • {post_path}")
            subplot_idx += 1
            
        elif n_params == 1:  # Single parameter model (shouldn't occur in typical use)
            fig_post = plt.figure(figsize=(10, 8))
            ax_post = fig_post.add_subplot(111)
            
            ax_post.hist(result.theta_chain[:, 0], bins=30, alpha=0.7, 
                        color='gray', edgecolor='black', linewidth=0.8, density=True)
            
            mean_val = np.mean(result.theta_chain[:, 0])
            std_val = np.std(result.theta_chain[:, 0])
            ax_post.axvline(mean_val, color='red', linestyle='--', linewidth=2)
            ax_post.set_title(f'Posterior Distribution: {model}', fontsize=16, fontweight='bold')
            ax_post.set_xlabel('Parameter Value', fontsize=14)
            ax_post.set_ylabel('Density', fontsize=14)
            ax_post.tick_params(labelsize=12)
            ax_post.grid(True, alpha=0.3)
            ax_post.text(0.95, 0.95, f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}',
                        transform=ax_post.transAxes, fontsize=12,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            fig_post.tight_layout()
            
            model_slug = _slugify(model, 'model')
            post_path = os.path.join(out_dir, f'{subplot_idx:02d}_Posterior_{model_slug}.png')
            fig_post.savefig(post_path, dpi=150, bbox_inches='tight')
            plt.close(fig_post)
            print(f"     • {post_path}")
            subplot_idx += 1

    plt.close()
    
    return log_marglik, bayes_factors


if __name__ == '__main__':
    print("模型选择模块已加载")
    print("\n使用示例:")
    print("  python model_selection.py [配置文件]")
    print("\n或在其他脚本中导入:")
    print("  from model_selection import compute_bayes_factors, plot_model_comparison")

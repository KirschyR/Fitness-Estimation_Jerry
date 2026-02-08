"""
PMCMC 多参数灵活估计
===================================================

支持灵活选择要估计的参数。
未指定的参数固定为 1.0（无适应度代价）。

使用方法：
python pmcmc_inference_multi.py \
    --true_female_viability 0.85 \
    --true_male_viability 0.7 \
    --estimate_params female_fecundity male_fecundity \
    --n_iter 1000 --n_particles 300
"""

import os
import json
from datetime import datetime
import numpy as np
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from utils import Species, AgeStructuredPopulation
from utils import simulation_kernels as sk
from samplers.pmcmc import (
    make_init_sampler,
    make_transition_fn,
    make_obs_loglik_fn,
    run_pmcmc,
    log_uniform_prior,
)
from samplers.observation import apply_rule


# =============================================================================
# 种群配置
# =============================================================================

SPECIES = Species.from_dict(name="AnophelesGambiae", structure={
    "chr1": {"A": ["WT", "Drive"]}
})

INITIAL_DISTRIBUTION = {
    "female": {
        "WT|WT":    [0, 6, 6, 5, 4, 3, 2, 1],
        "WT|Drive": [0, 0, 0, 0, 0, 0, 0, 0],
    },
    "male": {
        "WT|WT":    [0, 6, 6, 4, 2],
        "WT|Drive": [0, 0, 12, 0, 0],
    },
}

def DRIVE_MOD(population):
    return {
        "Drive|WT": {("Drive"): 0.8, ("WT"): 0.2},
        "WT|Drive": {("Drive"): 0.8, ("WT"): 0.2},
    }

OBS_GROUPS = {
    "F_WT":  {"genotype": ["WT|WT"], "sex": ["female"], "age": [2, 3, 4, 5, 6, 7]},
    "F_has_D": {"genotype": ["WT|Drive", "Drive|WT", "Drive|Drive"], "sex": ["female"], "age": [2, 3, 4, 5, 6, 7]},
    "M_WT":  {"genotype": ["WT|WT"], "sex": ["male"], "age": [2, 3, 4]},
    "M_has_D": {"genotype": ["WT|Drive", "Drive|WT", "Drive|Drive"], "sex": ["male"], "age": [2, 3, 4]},
}


# =============================================================================
# 命令行参数
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='PMCMC 灵活多参数估计')
    
    # 真实参数值（可选）
    parser.add_argument('--true_female_viability', type=float, default=None)
    parser.add_argument('--true_female_fecundity', type=float, default=None)
    parser.add_argument('--true_male_viability', type=float, default=None)
    parser.add_argument('--true_male_fecundity', type=float, default=None)
    parser.add_argument('--true_viability', type=float, default=None)
    parser.add_argument('--true_fecundity', type=float, default=None)
    
    # 指定要估计的参数（默认：全部）
    parser.add_argument('--estimate_params', type=str, nargs='+',
                        default=['female_viability', 'female_fecundity', 'male_viability', 'male_fecundity'],
                        choices=['female_viability', 'female_fecundity', 'male_viability', 'male_fecundity'],
                        help='要估计的参数列表，未指定的参数固定为1.0')
    
    parser.add_argument('--n_gen', type=int, default=20, help='Number of generations to simulate')
    parser.add_argument('--n_e', type=int, default=300, help='Effective population size')
    parser.add_argument('--n_iter', type=int, default=1000)
    parser.add_argument('--n_particles', type=int, default=500)
    parser.add_argument('--obs_sigma', type=float, default=0.5)  # 恢复为0.5
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='pmcmc_multi_results.png')
    parser.add_argument('--fig_dir', type=str, default='figs', help='Base directory to store figures')
    parser.add_argument('--json_dir', type=str, default='results_json', help='Directory to store JSON results')
    parser.add_argument('--run_name', type=str, default=None, help='Optional run name for subfolder; default uses timestamp')
    
    return parser.parse_args()


# =============================================================================
# 数据生成
# =============================================================================

def generate_data(fv_f, ff_f, fv_m, ff_m, n_gen=15, ne=300, seed=42):
    """生成观测数据"""
    print(f"\n生成数据: f_v_f={fv_f:.3f}, f_f_f={ff_f:.3f}, f_v_m={fv_m:.3f}, f_f_m={ff_m:.3f}")
    
    POPULATION_PARAMS = dict(
        species=SPECIES,
        n_ages=8,
        initial_population_distribution=INITIAL_DISTRIBUTION,
        female_adult_ages=[2, 3, 4, 5, 6, 7],
        male_adult_ages=[2, 3, 4],
        female_survival_rates=[1.0, 1.0, 5/6, 4/5, 3/4, 2/3, 1/2, 0],
        male_survival_rates=[1.0, 1.0, 2/3, 1/2, 0],
        offspring_per_female=50,
        use_sperm_storage=True,
        effective_population_size=ne,
        gamete_modifiers=[(0, "drive", DRIVE_MOD)],
    )
    
    pop = AgeStructuredPopulation(name="MosquitoPop", **POPULATION_PARAMS)
    
    # 设置适应度
    pop.set_viability("Drive|Drive", fv_f, sex='female')
    pop.set_viability("WT|Drive", np.sqrt(fv_f), sex='female')
    pop.set_viability("Drive|WT", np.sqrt(fv_f), sex='female')
    pop.set_fecundity("Drive|Drive", ff_f, sex='female')
    pop.set_fecundity("WT|Drive", np.sqrt(ff_f), sex='female')
    pop.set_fecundity("Drive|WT", np.sqrt(ff_f), sex='female')
    
    pop.set_viability("Drive|Drive", fv_m, sex='male')
    pop.set_viability("WT|Drive", np.sqrt(fv_m), sex='male')
    pop.set_viability("Drive|WT", np.sqrt(fv_m), sex='male')
    pop.set_fecundity("Drive|Drive", ff_m, sex='male')
    pop.set_fecundity("WT|Drive", np.sqrt(ff_m), sex='male')
    pop.set_fecundity("Drive|WT", np.sqrt(ff_m), sex='male')
    
    np.random.seed(seed)
    pop.run(n_gen)
    
    # 提取观测
    from samplers.observation import ObservationFilter
    obs_filter = ObservationFilter(pop.registry)
    obs_rule, obs_names = obs_filter.build_filter(
        pop, diploid_genotypes=pop.registry.index_to_genotype, groups=OBS_GROUPS
    )
    
    observations = []
    for week_data in pop._history:
        if isinstance(week_data, tuple) and len(week_data) >= 2:
            _, state_tuple = week_data
            individual_count = state_tuple[0]
            obs = apply_rule(individual_count, obs_rule)
            obs_flat = obs.sum(axis=(1, 2)) if obs.ndim == 3 else obs.sum(axis=1)
            observations.append(obs_flat)
    
    observations = np.array(observations)
    print(f"  观测形状: {observations.shape}")
    
    return pop, observations, obs_rule, obs_names


# =============================================================================
# PMCMC 推断
# =============================================================================

def run_inference(pop, observations, obs_rule, true_theta, estimate_params, 
                  n_iter=1000, n_particles=300, obs_sigma=0.5, seed=42):
    """运行PMCMC推断"""
    
    print(f"\n估计参数: {estimate_params}")
    n_params = len(estimate_params)
    
    # 关键：为推断创建新的种群，不设置fitness（默认全是1.0）
    # fitness会通过fitness_modifier动态设置
    POPULATION_PARAMS_INFERENCE = dict(
        species=pop.species,
        n_ages=8,
        initial_population_distribution=INITIAL_DISTRIBUTION,
        female_adult_ages=[2, 3, 4, 5, 6, 7],
        male_adult_ages=[2, 3, 4],
        female_survival_rates=[1.0, 1.0, 5/6, 4/5, 3/4, 2/3, 1/2, 0],
        male_survival_rates=[1.0, 1.0, 2/3, 1/2, 0],
        offspring_per_female=50,
        use_sperm_storage=True,
        effective_population_size=300,
        gamete_modifiers=[(0, "drive", DRIVE_MOD)],
    )
    pop_pmcmc = AgeStructuredPopulation(name="MosquitoPop_PMCMC", **POPULATION_PARAMS_INFERENCE)
    
    state, config, _ = sk.export_state(pop_pmcmc)
    ind = state[sk.STATE_INDIVIDUAL_COUNT]
    sperm = state[sk.STATE_SPERM_STORAGE]
    fem = state[sk.STATE_FEMALE_OCC]
    shapes = (ind.shape, sperm.shape, fem.shape)
    
    # 获取基因型索引
    homozygote_geno = pop.species.get_genotype_from_str('Drive|Drive')
    heterozygote_geno_1 = pop.species.get_genotype_from_str('WT|Drive')
    heterozygote_geno_2 = pop.species.get_genotype_from_str('Drive|WT')
    
    homozygote_idx = pop.registry.genotype_to_index[homozygote_geno]
    heterozygote_idx_1 = pop.registry.genotype_to_index[heterozygote_geno_1]
    heterozygote_idx_2 = pop.registry.genotype_to_index[heterozygote_geno_2]
    
    # 参数映射
    param_names_full = ['female_viability', 'female_fecundity', 'male_viability', 'male_fecundity']
    param_to_idx = {name: i for i, name in enumerate(estimate_params)}
    
    def fitness_modifier(theta, config):
        """只修改要估计的参数，其他参数保持config中的原始值"""
        new_config = list(config)
        
        # 复制原始的fitness数组
        new_female_viability = new_config[sk.CFG_FEMALE_VIABILITY].copy()
        new_female_fecundity = new_config[sk.CFG_FEMALE_FECUNDITY].copy()
        new_male_viability = new_config[sk.CFG_MALE_VIABILITY].copy()
        new_male_fecundity = new_config[sk.CFG_MALE_FECUNDITY].copy()
        
        # 只修改正在估计的参数
        if 'female_viability' in estimate_params:
            fv_f = theta[param_to_idx['female_viability']]
            fv_f_het = np.sqrt(fv_f)
            new_female_viability[homozygote_idx] = fv_f
            new_female_viability[heterozygote_idx_1] = fv_f_het
            new_female_viability[heterozygote_idx_2] = fv_f_het
        
        if 'female_fecundity' in estimate_params:
            ff_f = theta[param_to_idx['female_fecundity']]
            ff_f_het = np.sqrt(ff_f)
            new_female_fecundity[homozygote_idx] = ff_f
            new_female_fecundity[heterozygote_idx_1] = ff_f_het
            new_female_fecundity[heterozygote_idx_2] = ff_f_het
        
        if 'male_viability' in estimate_params:
            fv_m = theta[param_to_idx['male_viability']]
            fv_m_het = np.sqrt(fv_m)
            new_male_viability[homozygote_idx] = fv_m
            new_male_viability[heterozygote_idx_1] = fv_m_het
            new_male_viability[heterozygote_idx_2] = fv_m_het
        
        if 'male_fecundity' in estimate_params:
            ff_m = theta[param_to_idx['male_fecundity']]
            ff_m_het = np.sqrt(ff_m)
            new_male_fecundity[homozygote_idx] = ff_m
            new_male_fecundity[heterozygote_idx_1] = ff_m_het
            new_male_fecundity[heterozygote_idx_2] = ff_m_het
        
        # 更新config
        new_config[sk.CFG_FEMALE_VIABILITY] = new_female_viability
        new_config[sk.CFG_FEMALE_FECUNDITY] = new_female_fecundity
        new_config[sk.CFG_MALE_VIABILITY] = new_male_viability
        new_config[sk.CFG_MALE_FECUNDITY] = new_male_fecundity
        
        return tuple(new_config)
    
    # 定义聚合函数（与demo一致）
    def apply_rule_aggregated(individual_count, obs_rule):
        """应用观测规则并聚合性别和年龄维度"""
        obs = apply_rule(individual_count, obs_rule)
        if obs.ndim == 3:
            return obs.sum(axis=(1, 2))
        elif obs.ndim == 2:
            return obs.sum(axis=1)
        return obs
    
    # PMCMC设置
    init_sampler = make_init_sampler(state, seed=seed)
    transition_fn = make_transition_fn(config, shapes, seed=seed + 1000,
                                      config_modifier=fitness_modifier)
    obs_loglik_fn = make_obs_loglik_fn(obs_sigma, obs_rule, 
                                       apply_rule_aggregated,
                                       shapes=shapes)
    
    bounds = [(0.0, 1.0)] * n_params
    log_prior = lambda theta: log_uniform_prior(theta, bounds)
    theta_init = np.array([0.5] * n_params)
    
    print(f"  n_params={n_params}, n_iter={n_iter}, n_particles={n_particles}")
    print(f"  真实值: {dict(zip(estimate_params, [true_theta[param_names_full.index(p)] for p in estimate_params]))}")
    
    result = run_pmcmc(
        observations=observations,
        n_particles=n_particles,
        init_sampler=init_sampler,
        transition_fn=transition_fn,
        obs_loglik_fn=obs_loglik_fn,
        theta_init=theta_init,
        n_iter=n_iter,
        step_sizes=np.array([0.15] * n_params),
        log_prior_fn=log_prior,
        bounds=bounds,
        resample_threshold=0.5,
        burnin=n_iter // 4,
        thin=3,
        adapt=True,
        adapt_interval=50,
        target_accept=0.234,
        seed=seed + 2000,
        verbose=True
    )
    
    return result


# =============================================================================
# 结果分析和可视化
# =============================================================================

def analyze_and_plot(result, true_theta, estimate_params, observations, obs_names, output_file, fig_dir='figs', run_name=None, json_dir='results_json', seed=42, n_iter=1000, n_particles=300, n_gen=20, n_e=300):
    """分析结果并绘图"""
    
    param_names_full = ['female_viability', 'female_fecundity', 'male_viability', 'male_fecundity']
    param_display = {'female_viability': 'F_Viab', 'female_fecundity': 'F_Fec',
                     'male_viability': 'M_Viab', 'male_fecundity': 'M_Fec'}
    
    n_params = len(estimate_params)
    true_vals = [true_theta[param_names_full.index(p)] for p in estimate_params]
    
    # 分析
    print("\n" + "="*60)
    for i, param_name in enumerate(estimate_params):
        full_idx = param_names_full.index(param_name)
        display_name = param_display[param_name]
        
        mean = result.theta_chain[:, i].mean()
        std = result.theta_chain[:, i].std()
        ci_low = np.percentile(result.theta_chain[:, i], 2.5)
        ci_high = np.percentile(result.theta_chain[:, i], 97.5)
        true_val = true_theta[full_idx]
        
        print(f"\n{display_name}:")
        print(f"  真实值: {true_val:.4f}")
        print(f"  后验均值±std: {mean:.4f}±{std:.4f}")
        print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
        print(f"  在CI内: {ci_low <= true_val <= ci_high}")
    
    # 绘图 - 完整的分析图版
    print("\n绘制完整图表...")
    # 布局: 第一行轨迹，第二行后验，第三行ACF，第四行联合分布(如果>=2参数)和诊断，最后一行观测
    has_joint = n_params >= 2
    n_rows = 4 + (1 if has_joint else 0)
    n_cols = max(n_params, 3)

    # Prepare output directory
    run_id = run_name or datetime.now().strftime('%Y%m%d-%H%M%S')
    out_dir = os.path.join(fig_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(5*n_cols, 3.2*n_rows))
    
    # === 第1行：参数轨迹 ===
    for i in range(n_params):
        ax = plt.subplot(n_rows, n_cols, i+1)
        ax.plot(result.theta_chain[:, i], alpha=0.8, linewidth=0.9)
        ax.axhline(true_vals[i], color='r', linestyle='--', linewidth=1.6, label='True Value')
        mean_val = np.mean(result.theta_chain[:, i])
        ax.axhline(mean_val, color='g', linestyle='--', linewidth=1.6, label='Posterior Mean')
        ax.set_title(f'{param_display[estimate_params[i]]} / Trace', fontsize=11, fontweight='bold')
        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # === Row 2: Marginal Posteriors + KDE ===
    for i in range(n_params):
        ax = plt.subplot(n_rows, n_cols, n_cols + i + 1)
        ax.hist(result.theta_chain[:, i], bins=30, density=True, alpha=0.6,
                edgecolor='black', linewidth=0.8, label='Posterior Samples')
        if len(result.theta_chain) > 10:
            try:
                kde = gaussian_kde(result.theta_chain[:, i])
                x_range = np.linspace(result.theta_chain[:, i].min(), result.theta_chain[:, i].max(), 200)
                ax.plot(x_range, kde(x_range), 'b-', linewidth=2.0, label='KDE')
            except Exception:
                pass
        ax.axvline(true_vals[i], color='r', linestyle='--', linewidth=1.6, label='True Value')
        ax.set_title(f'{param_display[estimate_params[i]]} / Posterior', fontsize=11, fontweight='bold')
        ax.set_xlabel('Parameter Value', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # === Row 3: ACF ===
    for i in range(n_params):
        ax = plt.subplot(n_rows, n_cols, 2*n_cols + i + 1)
        max_lag = min(50, len(result.theta_chain) // 2)
        if max_lag > 1:
            acf_values = [1.0]
            for lag in range(1, max_lag):
                acf = np.corrcoef(result.theta_chain[:-lag, i], result.theta_chain[lag:, i])[0, 1]
                acf_values.append(acf)
            ax.bar(range(max_lag), acf_values, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axhline(0, color='k', linestyle='-', linewidth=0.8)
        ax.set_title(f'{param_display[estimate_params[i]]} / ACF', fontsize=11, fontweight='bold')
        ax.set_xlabel('Lag', fontsize=10)
        ax.set_ylabel('ACF', fontsize=10)
        ax.grid(True, alpha=0.3)

    # === Row 4: Joint Posterior (if >=2 parameters) + Diagnostics ===
    base_row = 3
    if has_joint:
        ax_joint = plt.subplot(n_rows, n_cols, base_row*n_cols + 1)
        x = result.theta_chain[:, 0]
        y = result.theta_chain[:, 1]
        ax_joint.hist2d(x, y, bins=40, cmap='Blues', density=True)
        ax_joint.plot(true_vals[0], true_vals[1], 'r*', markersize=10, label='True Value')
        ax_joint.set_title('Joint Posterior (First Two Parameters)', fontsize=11, fontweight='bold')
        ax_joint.set_xlabel(param_display[estimate_params[0]], fontsize=10)
        ax_joint.set_ylabel(param_display[estimate_params[1]], fontsize=10)
        ax_joint.legend(fontsize=9)
        ax_joint.grid(True, alpha=0.2)

        ax_loglik = plt.subplot(n_rows, n_cols, base_row*n_cols + 2)
    else:
        ax_loglik = plt.subplot(n_rows, n_cols, base_row*n_cols + 1)

    ax_loglik.plot(result.loglik_chain, alpha=0.8, linewidth=1, color='navy')
    ax_loglik.set_title('Log-Likelihood Trace', fontsize=11, fontweight='bold')
    ax_loglik.set_xlabel('Iteration', fontsize=10)
    ax_loglik.set_ylabel('Log-Likelihood', fontsize=10)
    ax_loglik.grid(True, alpha=0.3)

    if has_joint:
        ax_accept = plt.subplot(n_rows, n_cols, base_row*n_cols + 3)
    else:
        ax_accept = plt.subplot(n_rows, n_cols, base_row*n_cols + 2)

    accept_rate = np.cumsum(result.accepted) / np.arange(1, len(result.accepted)+1)
    ax_accept.plot(accept_rate, alpha=0.8, linewidth=1, color='darkgreen')
    ax_accept.axhline(0.234, color='r', linestyle='--', linewidth=1.6, label='Target Acceptance Rate')
    ax_accept.set_title('Cumulative Acceptance Rate', fontsize=11, fontweight='bold')
    ax_accept.set_xlabel('Iteration', fontsize=10)
    ax_accept.set_ylabel('Acceptance Rate', fontsize=10)
    ax_accept.set_ylim([0, 1])
    ax_accept.legend(fontsize=9)
    ax_accept.grid(True, alpha=0.3)

    # === Last Row: Observed Data Time Series (Each Observation Group) ===
    ax_obs = plt.subplot(n_rows, 1, n_rows)
    n_groups = observations.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, n_groups))
    for g in range(n_groups):
        label = obs_names[g] if g < len(obs_names) else f'Obs. Group {g}'
        ax_obs.plot(observations[:, g], marker='o', markersize=4,
                    label=label, alpha=0.8, linewidth=1.6, color=colors[g])
    ax_obs.set_title('Observed Data Time Series (Each Observation Group)', fontsize=12, fontweight='bold')
    ax_obs.set_xlabel('Time Step (Generation)', fontsize=10)
    ax_obs.set_ylabel('Count', fontsize=10)
    ax_obs.legend(ncol=min(n_groups, 5), fontsize=9, loc='best')
    ax_obs.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save full figure
    full_path = os.path.join(out_dir, os.path.basename(output_file))
    plt.savefig(full_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 完整图表已保存: {full_path}")

    # Save individual subplots as independent figures
    def _slugify(text, fallback):
        base = text.strip() or fallback
        safe = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in base)
        return safe.strip('_') or fallback

    def _save_subplot_individually(ax, filepath, figsize=(10, 8)):
        """Save a single subplot as an independent figure by recreating from data"""
        fig_single = plt.figure(figsize=figsize)
        ax_single = fig_single.add_subplot(111)
        
        # Recreate lines from data
        for line in ax.get_lines():
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            if len(xdata) > 0 and len(ydata) > 0:
                ax_single.plot(xdata, ydata,
                              color=line.get_color(), 
                              linestyle=line.get_linestyle(),
                              linewidth=line.get_linewidth(), 
                              marker=line.get_marker(),
                              markersize=line.get_markersize(), 
                              alpha=line.get_alpha(),
                              label=line.get_label() if line.get_label()[0] != '_' else '')
        
        # Recreate bar plots from patches
        if ax.patches:
            try:
                x_vals = []
                heights = []
                colors = []
                for patch in ax.patches:
                    bbox = patch.get_bbox()
                    x_vals.append(bbox.x0 + bbox.width / 2)
                    heights.append(bbox.height)
                    colors.append(patch.get_facecolor())
                if x_vals:
                    ax_single.bar(range(len(x_vals)), heights, color=colors, 
                                 alpha=0.7, edgecolor='black', linewidth=0.5)
            except:
                pass
        
        # Recreate histograms from collections
        for coll in ax.collections:
            try:
                offsets = coll.get_offsets()
                if len(offsets) > 0:
                    colors = coll.get_facecolors()
                    sizes = coll.get_sizes()
                    ax_single.scatter(offsets[:, 0], offsets[:, 1], 
                                    c=colors, s=sizes, alpha=coll.get_alpha())
            except:
                pass
        
        # Recreate images (hist2d, imshow)
        for img in ax.images:
            try:
                data = img.get_array()
                extent = img.get_extent()
                ax_single.imshow(data, extent=extent, aspect='auto', 
                               cmap=img.get_cmap(), alpha=img.get_alpha())
            except:
                pass
        
        # Copy axis properties with increased font sizes
        ax_single.set_xlim(ax.get_xlim())
        ax_single.set_ylim(ax.get_ylim())
        ax_single.set_xlabel(ax.get_xlabel(), fontsize=14)
        ax_single.set_ylabel(ax.get_ylabel(), fontsize=14)
        ax_single.set_title(ax.get_title(), fontsize=16, fontweight='bold')
        ax_single.tick_params(labelsize=12)
        
        # Recreate legend
        legend = ax.get_legend()
        if legend:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                try:
                    ax_single.legend(handles, labels, 
                                   fontsize=legend.get_texts()[0].get_fontsize() if legend.get_texts() else 9)
                except:
                    ax_single.legend(fontsize=9)
        
        # Add grid if present
        gridlines = ax.xaxis.get_gridlines() + ax.yaxis.get_gridlines()
        if gridlines and any(line.get_visible() for line in gridlines):
            ax_single.grid(True, alpha=0.3)
        
        fig_single.tight_layout()
        fig_single.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig_single)

    axes_all = list(fig.axes)
    for idx, ax in enumerate(axes_all, start=1):
        title = ax.get_title() or f'panel_{idx}'
        slug = _slugify(title, f'panel_{idx}')
        panel_path = os.path.join(out_dir, f'{idx:02d}_{slug}.png')
        _save_subplot_individually(ax, panel_path)
        print(f"  └─ 子图已保存: {panel_path}")

    plt.close()
    
    # Save JSON results for machine readability
    json_out_dir = os.path.join(json_dir, run_id)
    os.makedirs(json_out_dir, exist_ok=True)
    
    # Collect all parameter results
    param_results = {}
    for i, param_name in enumerate(estimate_params):
        full_idx = param_names_full.index(param_name)
        param_results[param_name] = {
            'true_value': float(true_theta[full_idx]),
            'posterior_mean': float(result.theta_chain[:, i].mean()),
            'posterior_std': float(result.theta_chain[:, i].std()),
            'ci_lower': float(np.percentile(result.theta_chain[:, i], 2.5)),
            'ci_upper': float(np.percentile(result.theta_chain[:, i], 97.5)),
            'in_ci': bool(np.percentile(result.theta_chain[:, i], 2.5) <= true_theta[full_idx] <= np.percentile(result.theta_chain[:, i], 97.5))
        }
    
    json_data = {
        'config': {
            'seed': int(seed),
            'n_iterations': int(n_iter),
            'n_particles': int(n_particles),
            'n_generations': int(n_gen),
            'effective_population_size': int(n_e),
            'estimate_params': estimate_params,
            'obs_sigma': 0.5
        },
        'parameters': param_results,
        'output_paths': {
            'figures_dir': out_dir,
            'full_figure': full_path
        }
    }
    
    json_file = os.path.join(json_out_dir, 'inference_results.json')
    with open(json_file, 'w') as f:
        f.write(json.dumps(json_data, indent=None) + '\n')
    print(f"\n✓ JSON结果已保存: {json_file}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("PMCMC 灵活多参数估计")
    print("="*60)
    
    # 初始化参数为1.0（无适应度代价）
    true_theta = np.array([1.0, 1.0, 1.0, 1.0])
    param_names_full = ['female_viability', 'female_fecundity', 'male_viability', 'male_fecundity']
    
    # 设置指定的参数值
    for i, param_name in enumerate(param_names_full):
        if param_name == 'female_viability':
            if args.true_female_viability is not None:
                true_theta[i] = args.true_female_viability
            elif args.true_viability is not None:
                true_theta[i] = args.true_viability
        elif param_name == 'female_fecundity':
            if args.true_female_fecundity is not None:
                true_theta[i] = args.true_female_fecundity
            elif args.true_fecundity is not None:
                true_theta[i] = args.true_fecundity
        elif param_name == 'male_viability':
            if args.true_male_viability is not None:
                true_theta[i] = args.true_male_viability
            elif args.true_viability is not None:
                true_theta[i] = args.true_viability
        elif param_name == 'male_fecundity':
            if args.true_male_fecundity is not None:
                true_theta[i] = args.true_male_fecundity
            elif args.true_fecundity is not None:
                true_theta[i] = args.true_fecundity
    
    print(f"\n配置：")
    print(f"  Seed:       {args.seed}")
    print(f"  Generations: {args.n_gen}")
    print(f"  Population Size: {args.n_e}")
    print(f"  Iterations: {args.n_iter}")
    print(f"  Particles:  {args.n_particles}")
    print(f"  Estimate Parameters: {args.estimate_params}")
    print(f"\n真实参数: {dict(zip(param_names_full, true_theta))}")
    print(f"估计参数: {args.estimate_params}")
    
    # 生成数据
    pop, observations, obs_rule, obs_names = generate_data(
        true_theta[0], true_theta[1], true_theta[2], true_theta[3],
        args.n_gen, args.n_e, args.seed
    )
    
    # 运行推断
    result = run_inference(
        pop, observations, obs_rule, true_theta,
        args.estimate_params, args.n_iter, args.n_particles,
        args.obs_sigma, args.seed
    )
    
    # 分析和绘图
    if args.run_name is None:
        run_name = f"seed_{args.seed}"
    else:
        run_name = args.run_name
    analyze_and_plot(result, true_theta, args.estimate_params, observations, obs_names,
                    args.output, fig_dir=args.fig_dir, run_name=run_name, json_dir=args.json_dir,
                    seed=args.seed, n_iter=args.n_iter, n_particles=args.n_particles, n_gen=args.n_gen, n_e=args.n_e)
    
    print("\n" + "="*60)
    print("完成！")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

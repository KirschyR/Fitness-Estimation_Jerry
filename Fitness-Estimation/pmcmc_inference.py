"""
PMCMC 采样器 - 灵活配置版本
===================================================

本脚本演示如何：
1. 创建一个年龄结构化种群
2. 模拟种群动态并生成观测数据（使用真实的适应度参数）
3. 使用 PMCMC 推断隐藏的适应度参数
4. 可视化推断结果

场景：转基因蚊子种群，估计转基因相对于野生型的生存力

使用方法：
=========
# 基本使用（使用默认参数）
python pmcmc_inference.py

# 自定义真实参数和迭代次数
python pmcmc_inference.py --true_viability 0.8 --n_iter 2000 --n_particles 300

# 调整遗传漂变和观测噪声
python pmcmc_inference.py --effective_population_size 500 --obs_sigma 1.0

# 完整示例
python pmcmc_inference.py \
    --true_viability 0.75 \
    --n_generations 20 \
    --n_iter 1000 \
    --n_particles 200 \
    --effective_population_size 300 \
    --obs_sigma 0.5 \
    --burnin 250 \
    --thin 3 \
    --seed 42 \
    --output pmcmc_results.png

关键参数调优指南：
===================
1. effective_population_size: 控制遗传漂变强度
   - 真实生物学现象，不应简单设为0
   - 值越小漂变越弱，似然函数越"确定"
   - 值越大随机性越强，需要更多粒子和更大观测噪声

2. obs_sigma (观测噪声标准差):
   - 太小(如0): 似然函数对随机波动过敏，MCMC接受率异常高
   - 太大: 信号淹没，参数不可识别
   - 建议: 设为观测量级的5-10%

3. n_particles: 粒子数
   - 随机性越强需要越多粒子
   - 建议: effective_population_size > 0 时至少200+

4. step_sizes: MCMC步长
   - 自动调节已添加上下界（参数范围的0.1%-20%）
   - 初始值建议为参数范围的10-20%
"""

import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# 配置 Matplotlib 以支持中文图注
mpl.rcParams['font.sans-serif'] = [
    'PingFang SC', 'Heiti SC', 'SimHei', 'Arial Unicode MS', 'Noto Sans CJK SC'
]
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.unicode_minus'] = False

from utils import Species, AgeStructuredPopulation
from utils import simulation_kernels as sk
from samplers.pmcmc import (
    PMCMC,
    make_fitness_modifier,
    make_init_sampler,
    make_transition_fn,
    make_obs_loglik_fn,
    run_pmcmc,
    log_uniform_prior,
)
from samplers.observation import ObservationFilter, apply_rule


def apply_rule_aggregated(individual_count, obs_rule):
    """应用观测规则并聚合性别和年龄维度。
    
    返回每个观测组的总个体数。
    """
    obs = apply_rule(individual_count, obs_rule)
    # obs 形状为 (n_groups, n_sexes, n_ages)，聚合为 (n_groups,)
    if obs.ndim == 3:
        return obs.sum(axis=(1, 2))
    elif obs.ndim == 2:
        return obs.sum(axis=1)
    return obs


# =============================================================================
# 全局配置：种群固定参数
# =============================================================================

# 物种定义
SPECIES = Species.from_dict(name="AnophelesGambiae", structure={
    "chr1": {"A": ["WT", "Drive"]}
})

# 初始种群分布（按性别和年龄）
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

# Gene drive 转换函数
def DRIVE_MOD(population):
    return {
        "Drive|WT": {("Drive"): 0.8, ("WT"): 0.2},
        "WT|Drive": {("Drive"): 0.8, ("WT"): 0.2},
    }

# 观测组定义
OBS_GROUPS = {
    "F_WT":  {"genotype": ["WT|WT"], "sex": ["female"], "age": [2, 3, 4, 5, 6, 7]},
    "F_has_D": {"genotype": ["WT|Drive", "Drive|WT", "Drive|Drive"], "sex": ["female"], "age": [2, 3, 4, 5, 6, 7]},
    "M_WT":  {"genotype": ["WT|WT"], "sex": ["male"], "age": [2, 3, 4]},
    "M_has_D": {"genotype": ["WT|Drive", "Drive|WT", "Drive|Drive"], "sex": ["male"], "age": [2, 3, 4]},
}


# =============================================================================
# 步骤 1: 创建种群模型
# =============================================================================

def create_population_and_generate_data(
    true_viability: float,
    n_generations: int = 20,
    effective_population_size: int = 300,
    seed: int = 42,
    obs_noise: float = 0.0
):
    """
    创建种群，设置真实适应度参数，运行模拟生成观测数据。
    
    Args:
        true_viability: Drive 基因型的真实生存力（相对于 WT）
        n_generations: 模拟的世代数
        effective_population_size: 有效种群大小，控制遗传漂变强度
        seed: 随机种子
        obs_noise: 观测噪声标准差
        
    Returns:
        pop_for_pmcmc: 初始种群对象（用于 PMCMC 初始化）
        observations: 观测数据矩阵 (n_generations, n_obs_groups)
        obs_rule: 观测规则矩阵
        obs_names: 观测组名称列表
    """
    print("=" * 70)
    print("步骤 1: 创建种群并生成观测数据")
    print("=" * 70)
    
    # 种群参数（可配置）
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
        effective_population_size=effective_population_size,
        gamete_modifiers=[(0, "drive", DRIVE_MOD)],
    )
    
    # 创建数据生成种群
    pop_data = AgeStructuredPopulation(name="MosquitoPop_Data", **POPULATION_PARAMS)
    
    # 设置真实的适应度参数
    # 部分显性模型：纯合子 = θ，杂合子 = √θ
    pop_data.set_viability("Drive|Drive", true_viability)
    pop_data.set_viability("WT|Drive", np.sqrt(true_viability))
    pop_data.set_viability("Drive|WT", np.sqrt(true_viability))
    
    print(f"  物种: {SPECIES.name}")
    print(f"  真实 Drive|Drive 生存力 (θ): {true_viability:.4f}")
    print(f"  真实 WT|Drive 生存力 (√θ): {np.sqrt(true_viability):.4f}")
    print(f"  有效种群大小 (Ne): {effective_population_size}")
    
    # 运行模拟生成观测数据
    print(f"\n  运行 {n_generations} 代模拟...")
    np.random.seed(seed)
    pop_data.run(n_generations)
    
    # 构建观测规则
    obs_filter = ObservationFilter(pop_data.registry)
    obs_rule, obs_names = obs_filter.build_filter(
        pop_data,
        diploid_genotypes=pop_data.registry.index_to_genotype,
        groups=OBS_GROUPS,
    )
    
    # 从历史记录中提取观测
    observations = []
    for week_data in pop_data._history:
        if isinstance(week_data, tuple) and len(week_data) >= 2:
            _, state_tuple = week_data
            individual_count = state_tuple[0]
            obs = apply_rule(individual_count, obs_rule)
            obs_flat = obs.sum(axis=(1, 2)) if obs.ndim == 3 else obs.sum(axis=1)
            observations.append(obs_flat)
    
    observations = np.array(observations)
    
    # 添加观测噪声（模拟实际采样/测量误差）
    # 关键：适当的噪声可以"平滑"似然函数，使MCMC更易收敛
    # 建议：设为观测量级的5-10%
    if obs_noise > 0:
        np.random.seed(seed + 100)  # 使用不同种子避免与模拟相关
        noise = np.random.normal(0, obs_noise, observations.shape)
        observations = np.maximum(observations + noise, 0)
        print(f"  添加观测噪声: σ = {obs_noise:.2f}")
    
    print(f"\n✓ 数据生成完成")
    print(f"  观测序列形状: {observations.shape}")
    print(f"  观测组: {obs_names}")
    print(f"  观测数据范围: [{observations.min():.1f}, {observations.max():.1f}]")
    
    # 创建推断用的种群（不设置 viability）
    # 使用相同的 effective_population_size
    pop_for_pmcmc = AgeStructuredPopulation(
        name="MosquitoPop_PMCMC",
        species=SPECIES,
        n_ages=8,
        initial_population_distribution=INITIAL_DISTRIBUTION,
        female_adult_ages=[2, 3, 4, 5, 6, 7],
        male_adult_ages=[2, 3, 4],
        female_survival_rates=[1.0, 1.0, 5/6, 4/5, 3/4, 2/3, 1/2, 0],
        male_survival_rates=[1.0, 1.0, 2/3, 1/2, 0],
        offspring_per_female=50,
        use_sperm_storage=True,
        effective_population_size=effective_population_size,
        gamete_modifiers=[(0, "drive", DRIVE_MOD)],
    )
    
    return pop_for_pmcmc, observations, obs_rule, obs_names


# =============================================================================
# 步骤 2: 设置并运行 PMCMC
# =============================================================================

def run_pmcmc_inference(
    pop,
    observations,
    obs_rule,
    true_theta,
    n_iter=2000,
    n_particles=500,
    obs_sigma=0.5,
    burnin=None,
    thin=3,
    theta_init=None,
    seed=42,
    verbose=True
):
    """
    使用 PMCMC 推断适应度参数。
    
    Args:
        pop: 初始种群对象
        observations: 观测数据
        obs_rule: 观测规则矩阵
        true_theta: 真实参数值（用于比较）
        n_iter: MCMC 迭代次数
        n_particles: 粒子数
        obs_sigma: 观测噪声标准差（用于似然计算）
        burnin: burn-in 迭代数（默认为 n_iter // 4）
        thin: 稀疏化间隔
        theta_init: 初始参数猜测（默认为 0.5）
        seed: 随机种子
        verbose: 是否打印详细信息
        
    Returns:
        result: PMCMC 采样结果
    """
    print("\n" + "=" * 70)
    print("步骤 2: 运行 PMCMC 采样器")
    print("=" * 70)
    
    # 导出状态和配置
    state, config, _ = sk.export_state(pop)
    
    # 获取形状信息
    ind = state[sk.STATE_INDIVIDUAL_COUNT]
    sperm = state[sk.STATE_SPERM_STORAGE]
    fem = state[sk.STATE_FEMALE_OCC]
    shapes = (ind.shape, sperm.shape, fem.shape)
    
    # 创建适应度修改器
    # 部分显性模型：纯合子 = θ，杂合子 = √θ
    # 我们估计 θ（纯合子的 viability），然后推导杂合子
    # 
    # 由于 make_fitness_modifier 会直接赋值，我们需要自定义 modifier
    # 来实现 sqrt 关系
    # 预先获取基因型索引（避免每次调用时重复查找）
    homozygote_geno = pop.species.get_genotype_from_str('Drive|Drive')
    heterozygote_geno_1 = pop.species.get_genotype_from_str('WT|Drive')
    heterozygote_geno_2 = pop.species.get_genotype_from_str('Drive|WT')
    
    homozygote_idx = pop.registry.genotype_to_index[homozygote_geno]
    heterozygote_idx_1 = pop.registry.genotype_to_index[heterozygote_geno_1]
    heterozygote_idx_2 = pop.registry.genotype_to_index[heterozygote_geno_2]
    
    def custom_fitness_modifier(theta, config):
        """自定义适应度修改器，实现部分显性模型。
        
        θ = Drive|Drive 的 viability（纯合子）
        √θ = WT|Drive 的 viability（杂合子）
        """
        drive_viability = theta[0]
        heterozygote_viability = np.sqrt(drive_viability)
        
        # 修改 female 和 male 的 viability 数组
        new_female_viability = config[sk.CFG_FEMALE_VIABILITY].copy()
        new_male_viability = config[sk.CFG_MALE_VIABILITY].copy()
        
        new_female_viability[homozygote_idx] = drive_viability
        new_female_viability[heterozygote_idx_1] = heterozygote_viability
        new_female_viability[heterozygote_idx_2] = heterozygote_viability
        new_male_viability[homozygote_idx] = drive_viability
        new_male_viability[heterozygote_idx_1] = heterozygote_viability
        new_male_viability[heterozygote_idx_2] = heterozygote_viability
        
        # 返回修改后的配置
        new_config = list(config)
        new_config[sk.CFG_FEMALE_VIABILITY] = new_female_viability
        new_config[sk.CFG_MALE_VIABILITY] = new_male_viability
        return tuple(new_config)
    
    print(f"  参数: θ (Drive|Drive viability)")
    print(f"  模型: Drive|Drive = θ, WT|Drive = √θ (部分显性)")
    
    # 使用低级 API 创建各个组件
    
    # 初始状态采样器
    init_sampler = make_init_sampler(state, seed=seed)
    
    # 转移函数（带参数修改器）
    transition_fn = make_transition_fn(
        config, shapes, seed=seed + 1000,
        config_modifier=custom_fitness_modifier
    )
    
    # 观测似然函数（使用聚合的投影函数）
    obs_loglik_fn = make_obs_loglik_fn(
        obs_sigma, obs_rule, apply_rule_aggregated, shapes=shapes
    )
    
    # 定义先验
    bounds = [(0.0, 1.0)]  # Drive 生存力在 0%-100% 之间
    log_prior = lambda theta: log_uniform_prior(theta, bounds)
    
    # 初始猜测（故意偏离真实值）
    if theta_init is None:
        theta_init = np.array([0.5])
    else:
        theta_init = np.array([theta_init])
    
    # 默认 burnin
    if burnin is None:
        burnin = n_iter // 4
    
    print(f"\n  真实参数: {true_theta}")
    print(f"  初始猜测: {theta_init}")
    print(f"  先验范围: {bounds}")
    print(f"  粒子数: {n_particles}")
    print(f"  迭代数: {n_iter}")
    print(f"  Burn-in: {burnin}")
    print(f"  Thinning: {thin}")
    print(f"  观测噪声 (σ): {obs_sigma:.2f}")
    
    # 运行采样
    print("\n  开始 MCMC 采样...")
    result = run_pmcmc(
        observations=observations,
        n_particles=n_particles,
        init_sampler=init_sampler,
        transition_fn=transition_fn,
        obs_loglik_fn=obs_loglik_fn,
        theta_init=theta_init,
        n_iter=n_iter,
        step_sizes=np.array([0.2]),
        log_prior_fn=log_prior,
        bounds=bounds,
        resample_threshold=0.5,
        burnin=burnin,
        thin=thin,
        adapt=True,
        adapt_interval=50,
        target_accept=0.234,
        seed=seed + 2000,
        verbose=verbose
    )
    
    print(f"\n✓ PMCMC 采样完成")
    print(f"  保存样本数: {len(result.theta_chain)}")
    print(f"  接受率: {result.acceptance_rate:.3f}")
    
    return result


# =============================================================================
# 步骤 3: 分析结果
# =============================================================================

def analyze_results(result, true_theta):
    """分析和打印推断结果"""
    
    print("\n" + "=" * 70)
    print("步骤 3: 结果分析")
    print("=" * 70)
    
    # 后验统计
    posterior_mean = result.theta_chain.mean(axis=0)
    posterior_std = result.theta_chain.std(axis=0)
    posterior_median = np.median(result.theta_chain, axis=0)
    
    # 95% 可信区间
    ci_low = np.percentile(result.theta_chain, 2.5, axis=0)
    ci_high = np.percentile(result.theta_chain, 97.5, axis=0)
    
    print(f"\n后验分布统计:")
    print(f"  真实参数: {true_theta[0]:.4f}")
    print(f"  后验均值: {posterior_mean[0]:.4f} ± {posterior_std[0]:.4f}")
    print(f"  后验中位数: {posterior_median[0]:.4f}")
    print(f"  95% CI: [{ci_low[0]:.4f}, {ci_high[0]:.4f}]")
    print(f"  真实值在 CI 内: {ci_low[0] <= true_theta[0] <= ci_high[0]}")
    
    # 估计偏差
    bias = posterior_mean[0] - true_theta[0]
    rmse = np.sqrt(((result.theta_chain - true_theta[0])**2).mean())
    print(f"\n误差指标:")
    print(f"  偏差 (Bias): {bias:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    
    return {
        'mean': posterior_mean,
        'std': posterior_std,
        'median': posterior_median,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'bias': bias,
        'rmse': rmse
    }


# =============================================================================
# 步骤 4: 可视化
# =============================================================================

def plot_results(result, true_theta, observations, stats, output_file='pmcmc_inference_results.png'):
    """绘制结果图表"""
    
    print("\n生成可视化图表...")
    
    fig = plt.figure(figsize=(14, 10))
    
    # 1. 参数轨迹图
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(result.theta_chain[:, 0], alpha=0.8, linewidth=1)
    ax1.axhline(true_theta[0], color='r', linestyle='--', linewidth=2, label='真实值')
    ax1.axhline(stats['mean'][0], color='g', linestyle='--', linewidth=2, label='后验均值')
    ax1.set_xlabel('采样迭代')
    ax1.set_ylabel('Drive 生存力')
    ax1.set_title('参数轨迹图')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 后验分布直方图
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(result.theta_chain[:, 0], bins=30, density=True, alpha=0.7, edgecolor='black')
    
    # 添加 KDE 曲线
    if len(result.theta_chain) > 10:
        try:
            kde = gaussian_kde(result.theta_chain[:, 0])
            x_range = np.linspace(
                result.theta_chain[:, 0].min(),
                result.theta_chain[:, 0].max(),
                200
            )
            ax2.plot(x_range, kde(x_range), 'b-', linewidth=2, label='KDE')
        except:
            pass
    
    ax2.axvline(true_theta[0], color='r', linestyle='--', linewidth=2, label='真实值')
    ax2.set_xlabel('Drive 生存力')
    ax2.set_ylabel('概率密度')
    ax2.set_title('后验分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 自相关图
    ax3 = plt.subplot(2, 3, 3)
    max_lag = min(50, len(result.theta_chain) // 2)
    if max_lag > 1:
        acf_values = []
        for lag in range(max_lag):
            if lag == 0:
                acf_values.append(1.0)
            else:
                acf = np.corrcoef(
                    result.theta_chain[:-lag, 0],
                    result.theta_chain[lag:, 0]
                )[0, 1]
                acf_values.append(acf)
        ax3.bar(range(max_lag), acf_values, alpha=0.7, edgecolor='black')
    ax3.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('滞后步数')
    ax3.set_ylabel('自相关')
    ax3.set_title('自相关函数 (ACF)')
    ax3.grid(True, alpha=0.3)

    # 4. 观测数据时间序列
    ax4 = plt.subplot(2, 3, 4)
    for geno in range(observations.shape[1]):
        ax4.plot(observations[:, geno], marker='o', markersize=3, label=f'分组 {geno}', alpha=0.7)
    ax4.set_xlabel('时间步')
    ax4.set_ylabel('个体计数')
    ax4.set_title('观测数据 (各基因型)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 对数似然轨迹
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(result.loglik_chain, alpha=0.8, linewidth=1)
    ax5.set_xlabel('采样迭代')
    ax5.set_ylabel('对数似然')
    ax5.set_title('对数似然轨迹')
    ax5.grid(True, alpha=0.3)
    
    # 6. 累积接受率
    ax6 = plt.subplot(2, 3, 6)
    acceptance_cumsum = np.cumsum(result.accepted)
    iteration_nums = np.arange(1, len(acceptance_cumsum) + 1)
    acceptance_rate = acceptance_cumsum / iteration_nums
    ax6.plot(acceptance_rate, alpha=0.8, linewidth=1)
    ax6.axhline(0.234, color='r', linestyle='--', linewidth=2, label='目标接受率')
    ax6.set_xlabel('采样迭代')
    ax6.set_ylabel('累积接受率')
    ax6.set_title('接受率演变')
    ax6.set_ylim([0, 1])
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ 图表已保存: {output_file}")
    plt.close()


# =============================================================================
# 命令行参数解析
# =============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='PMCMC 参数推断 - 估计转基因蚊子的 Drive 生存力',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认参数
  python pmcmc_inference.py
  
  # 自定义真实参数和迭代次数
  python pmcmc_inference.py --true_viability 0.8 --n_iter 2000 --n_particles 300
  
  # 调整遗传漂变和观测噪声
  python pmcmc_inference.py --effective_population_size 500 --obs_sigma 1.0
  
  # 完整示例
  python pmcmc_inference.py \\
      --true_viability 0.75 \\
      --n_generations 20 \\
      --n_iter 1000 \\
      --n_particles 200 \\
      --effective_population_size 300 \\
      --obs_sigma 0.5 \\
      --burnin 250 \\
      --thin 3 \\
      --seed 42 \\
      --output my_results.png
        """
    )
    
    # 数据生成参数
    data_group = parser.add_argument_group('数据生成参数')
    data_group.add_argument(
        '--true_viability', type=float, default=0.75,
        help='Drive|Drive 的真实生存力 (默认: 0.75)'
    )
    data_group.add_argument(
        '--n_generations', type=int, default=15,
        help='模拟的世代数 (默认: 15)'
    )
    data_group.add_argument(
        '--effective_population_size', type=int, default=300,
        help='有效种群大小，控制遗传漂变强度 (默认: 300)'
    )
    data_group.add_argument(
        '--obs_noise', type=float, default=0.0,
        help='添加到观测数据的噪声标准差 (默认: 0.0)'
    )
    
    # PMCMC 参数
    pmcmc_group = parser.add_argument_group('PMCMC 采样参数')
    pmcmc_group.add_argument(
        '--n_iter', type=int, default=800,
        help='MCMC 总迭代次数 (默认: 800)'
    )
    pmcmc_group.add_argument(
        '--n_particles', type=int, default=200,
        help='粒子滤波的粒子数 (默认: 200)'
    )
    pmcmc_group.add_argument(
        '--obs_sigma', type=float, default=0.5,
        help='观测似然的噪声标准差 (默认: 0.5)'
    )
    pmcmc_group.add_argument(
        '--burnin', type=int, default=None,
        help='Burn-in 迭代数 (默认: n_iter//4)'
    )
    pmcmc_group.add_argument(
        '--thin', type=int, default=3,
        help='稀疏化间隔 (默认: 3)'
    )
    pmcmc_group.add_argument(
        '--theta_init', type=float, default=None,
        help='初始参数猜测 (默认: 0.5)'
    )
    
    # 其他参数
    other_group = parser.add_argument_group('其他参数')
    other_group.add_argument(
        '--seed', type=int, default=42,
        help='随机种子 (默认: 42)'
    )
    other_group.add_argument(
        '--output', type=str, default='pmcmc_inference_results.png',
        help='输出图表文件名 (默认: pmcmc_inference_results.png)'
    )
    other_group.add_argument(
        '--no_plot', action='store_true',
        help='不生成可视化图表'
    )
    other_group.add_argument(
        '--quiet', action='store_true',
        help='减少输出信息'
    )
    
    return parser.parse_args()


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数：完整演示流程"""
    
    # 解析命令行参数
    args = parse_args()
    
    if not args.quiet:
        print("\n")
        print("╔" + "═" * 68 + "╗")
        print("║" + " " * 68 + "║")
        print("║" + "PMCMC 采样器 - 参数推断".center(68) + "║")
        print("║" + " " * 68 + "║")
        print("║" + "场景：估计转基因蚊子的 Drive 生存力".center(68) + "║")
        print("║" + " " * 68 + "║")
        print("╚" + "═" * 68 + "╝")
    
    # 真实参数
    true_theta = np.array([args.true_viability])
    
    # 步骤 1: 创建种群并生成数据
    pop, observations, obs_rule, obs_names = create_population_and_generate_data(
        true_viability=args.true_viability,
        n_generations=args.n_generations,
        effective_population_size=args.effective_population_size,
        seed=args.seed,
        obs_noise=args.obs_noise
    )
    
    # 步骤 2: 运行 PMCMC
    result = run_pmcmc_inference(
        pop, 
        observations, 
        obs_rule,
        true_theta,
        n_iter=args.n_iter,
        n_particles=args.n_particles,
        obs_sigma=args.obs_sigma,
        burnin=args.burnin,
        thin=args.thin,
        theta_init=args.theta_init,
        seed=args.seed,
        verbose=not args.quiet
    )
    
    # 步骤 3: 分析结果
    stats = analyze_results(result, true_theta)
    
    # 步骤 4: 绘制结果
    if not args.no_plot:
        plot_results(result, true_theta, observations, stats, args.output)
    
    if not args.quiet:
        print("\n" + "=" * 70)
        print("演示完成！")
        print("=" * 70)
        print("\n总结:")
        print(f"  ✓ 真实参数: {true_theta[0]:.4f}")
        print(f"  ✓ 推断的后验均值: {stats['mean'][0]:.4f}")
        print(f"  ✓ 推断的后验标准差: {stats['std'][0]:.4f}")
        print(f"  ✓ 参数在 95% 可信区间内: {stats['ci_low'][0] <= true_theta[0] <= stats['ci_high'][0]}")
        print(f"  ✓ 接受率: {result.acceptance_rate:.3f}")
        print()


if __name__ == "__main__":
    main()

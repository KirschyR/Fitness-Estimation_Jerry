#!/usr/bin/env python3
"""
批量运行配置文件
===============

用户可以修改这个文件来自定义批量运行的参数
"""

# ============================================================================
# 全局配置
# ============================================================================

# 并行运行的进程数
MAX_WORKERS = 8

# 参数值范围 (从最小值到最大值，包括边界)
PARAM_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# 输出目录
FIG_DIR = 'figs'
JSON_DIR = 'results_json'

# ============================================================================
# 任务配置 - 设为 True 来启用任务，False 来禁用
# ============================================================================

TASK_CONFIG = {
    'task1_estimate_fecundity': True,      # 任务1: 估计 Fecundity
    'task2_estimate_viability': True,      # 任务2: 估计 Viability
    'task3_model_comparison_fecundity': True,  # 任务3: 模型选择 Fecundity
    'task4_model_comparison_viability': True,  # 任务4: 模型选择 Viability
}

# ============================================================================
# PMCMC 参数配置 (对应 pmcmc_inference_multi.py)
# ============================================================================

PMCMC_CONFIG = {
    'n_iter': 1000,          # PMCMC 迭代次数
    'n_particles': 300,      # 粒子滤波粒子数
    'n_gen': 20,             # 模拟代数
    'n_e': 300,              # 有效种群大小
}

# ============================================================================
# 模型选择参数配置 (对应 quick_model_comparison.py)
# ============================================================================

MODEL_COMPARISON_CONFIG = {
    'n_iter': 400,           # PMCMC 迭代次数
    'n_particles': 200,      # 粒子滤波粒子数
    'n_gen': 20,             # 模拟代数
    'n_e': 300,              # 有效种群大小
}

# ============================================================================
# Seed 配置
# ============================================================================

# 如果为 None，则随机生成 seed (1000-9999)
# 如果为正整数，则从该数字开始递增
SEED_START = None  # 设置为某个数字如 1000 来使用固定seed序列

# ============================================================================
# 输出配置
# ============================================================================

# Run name 前缀
RUN_NAME_PREFIXES = {
    'estimate_fecundity': 'est_fec',
    'estimate_viability': 'est_vib',
    'model_comparison_fecundity': 'mc_fec',
    'model_comparison_viability': 'mc_vib',
}

# 是否在 run_name 中包含参数值
INCLUDE_PARAM_IN_RUN_NAME = True

# 是否在 run_name 中包含 seed
INCLUDE_SEED_IN_RUN_NAME = True

# ============================================================================
# 高级配置
# ============================================================================

# 超时时间（秒）- 单个任务的最大运行时间
TASK_TIMEOUT = 3600

# 是否在运行前创建输出目录
CREATE_OUTPUT_DIRS = True

# 是否保留失败任务的信息用于重试
KEEP_FAILED_TASKS = True

# 进度输出间隔（每N个任务完成时输出一次）
PROGRESS_INTERVAL = 10

# ============================================================================
# 调试配置
# ============================================================================

# 详细日志输出
VERBOSE = False

# 仅生成命令但不执行（用于调试）
DRY_RUN = False

# ============================================================================
# 验证配置（如有更改请同步修改）
# ============================================================================

def validate_config():
    """验证配置的有效性"""
    errors = []
    
    # 检查并行进程数
    if MAX_WORKERS < 1 or MAX_WORKERS > 128:
        errors.append("MAX_WORKERS 应在 1-128 之间")
    
    # 检查参数值
    if len(PARAM_VALUES) < 1:
        errors.append("PARAM_VALUES 不能为空")
    
    if not all(0.0 <= v <= 1.0 for v in PARAM_VALUES):
        errors.append("PARAM_VALUES 中所有值应在 0.0-1.0 之间")
    
    # 检查 PMCMC 参数
    if PMCMC_CONFIG['n_iter'] < 100:
        errors.append("n_iter 应至少为 100")
    
    if PMCMC_CONFIG['n_particles'] < 10:
        errors.append("n_particles 应至少为 10")
    
    # 检查 seed 配置
    if SEED_START is not None and SEED_START < 0:
        errors.append("SEED_START 应为非负数或 None")
    
    # 检查超时时间
    if TASK_TIMEOUT < 60:
        errors.append("TASK_TIMEOUT 应至少为 60 秒")
    
    return errors


# ============================================================================
# 计算统计信息
# ============================================================================

def get_statistics():
    """获取批量运行的统计信息"""
    n_values = len(PARAM_VALUES)
    n_combinations = n_values * n_values
    
    n_tasks = sum(1 for v in TASK_CONFIG.values() if v)
    n_total_runs = n_combinations * n_tasks
    
    return {
        'n_param_values': n_values,
        'n_combinations_per_task': n_combinations,
        'n_enabled_tasks': n_tasks,
        'n_total_runs': n_total_runs,
        'max_workers': MAX_WORKERS,
        'estimated_time_hours': (n_total_runs * 0.5) / MAX_WORKERS,  # 假设每个任务30分钟
    }


# ============================================================================
# 主程序入口配置示例
# ============================================================================

if __name__ == '__main__':
    # 验证配置
    errors = validate_config()
    if errors:
        print("配置错误:")
        for error in errors:
            print(f"  - {error}")
        exit(1)
    
    # 显示统计信息
    stats = get_statistics()
    print("批量运行配置统计:")
    print(f"  参数值个数: {stats['n_param_values']}")
    print(f"  每个任务的参数组合数: {stats['n_combinations_per_task']}")
    print(f"  启用的任务数: {stats['n_enabled_tasks']}")
    print(f"  总运行数: {stats['n_total_runs']}")
    print(f"  并行进程数: {stats['max_workers']}")
    print(f"  估计耗时: {stats['estimated_time_hours']:.1f} 小时")
    print("\n配置有效 ✓")

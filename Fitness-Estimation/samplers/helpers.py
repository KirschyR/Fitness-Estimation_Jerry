import numpy as np
from typing import Callable, Tuple, List, Dict

# Try to import numba.objmode; if unavailable provide a no-op fallback
try:
    from numba import objmode
except Exception:
    from contextlib import contextmanager
    @contextmanager
    def objmode(*args, **kwargs):
        yield None

def particle_filter(
    x_obs, 
    N, 
    init_state_sampler, 
    transition_fn, 
    obs_loglik_fn, 
    theta,     
    resample_threshold=0.5, 
    resample_always=False,
    return_particle_stats: bool = False,
    return_debug: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    通用粒子滤波器 (Bootstrap PF)，只在 ESS 低于阈值时重采样

    参数:
    --------
    x_obs : array, shape (T, ...)
        观测数据序列
    N : int
        粒子数
    init_state_sampler : callable
        用于采样初始状态的函数: fn(N) -> shape (N, state_dim)
    transition_fn : callable
        状态转移采样函数: fn(prev_states, theta) -> next_states (shape (N, state_dim))
    obs_loglik_fn : callable
        观测模型 log 似然函数: fn(x_t, states, theta) -> log_likelihoods, shape (N,)
    theta : array-like
        当前模型参数
    resample_threshold : float in (0,1)
        ESS 阈值的比例（默认 0.5），当 ESS < resample_threshold * N 时触发重采样
    resample_always : bool
        如果为 True 则每一步都强制重采样（与原实现行为一致）

    返回:
    --------
    particles : ndarray, shape (T, N, state_dim)
        粒子轨迹（按时间存储）
    weights : ndarray, shape (T, N)
        各时间步归一化权重
    loglik : float
        整个序列的对数似然估计（逐步乘法的对数估计）
    """
    T = len(x_obs)

    with objmode():
        if verbose:
            print(f"particle_filter start: T={T}, N={N}, resample_threshold={resample_threshold:.3f}, resample_always={resample_always}")

    # Prepare history containers (no printing) if requested
    if return_debug:
        ess_history = np.zeros(T, dtype=np.float64)
        resampled_flags = np.zeros(T, dtype=np.bool_)
        resample_head = -np.ones((T, 10), dtype=np.int64)
        obs_loglik_stats = np.zeros((T, 3), dtype=np.float64)  # mean, min, max
        weight_stats = np.zeros((T, 3), dtype=np.float64)  # mean, min, max
        loglik_increment = np.zeros(T, dtype=np.float64)
        cumulative_loglik = np.zeros(T, dtype=np.float64)
        init_samples_shape_arr = None
        state_dim_arr = None

    # 预采样一次以获得 state_dim
    init_samples = init_state_sampler(N)
    assert init_samples is not None, "init_state_sampler returned None"
    if init_samples.ndim == 1:
        init_samples = init_samples.reshape(N, -1)
    state_dim = init_samples.shape[1]
    if return_debug:
        init_samples_shape_arr = np.array(init_samples.shape, dtype=np.int64)
        state_dim_arr = np.array(state_dim, dtype=np.int64)
    with objmode():
        if verbose:
            print(f"init_samples.shape={init_samples.shape}, state_dim={state_dim}")

    particles = np.zeros((T, N, state_dim))
    weights = np.zeros((T, N))

    # Optionally collect per-generation particle statistics (unweighted and weighted)
    if return_particle_stats:
        means = np.zeros((T, state_dim), dtype=np.float64)
        vars_ = np.zeros((T, state_dim), dtype=np.float64)
        w_means = np.zeros((T, state_dim), dtype=np.float64)
        w_vars = np.zeros((T, state_dim), dtype=np.float64)
    else:
        means = vars_ = w_means = w_vars = None

    # 初始化
    particles[0] = init_samples
    prev_weights = np.ones(N) / N
    log_prev = np.log(prev_weights)

    obs_loglik = obs_loglik_fn(x_obs[0], particles[0], theta)  # log p(y0 | x0_i)
    assert obs_loglik is not None, "obs_loglik_fn returned None for t=0"
    if return_debug:
        obs_loglik_stats[0, 0] = np.mean(obs_loglik)
        obs_loglik_stats[0, 1] = np.min(obs_loglik)
        obs_loglik_stats[0, 2] = np.max(obs_loglik)
    with objmode():
        if verbose:
            print(f"t=0 obs_loglik stats: mean={np.mean(obs_loglik):.4e}, min={np.min(obs_loglik):.4e}, max={np.max(obs_loglik):.4e}")

    log_w_raw = obs_loglik + log_prev  # 包含先验权重（均匀）
    max_log = np.max(log_w_raw)
    log_sum = max_log + np.log(np.sum(np.exp(log_w_raw - max_log)))  # logsumexp
    # 此处 log_sum 已经是 log( (1/N) * sum_i p(y0|x0_i) )，用作增量对数似然
    loglik = log_sum
    w = np.exp(log_w_raw - log_sum)
    weights[0] = w
    init_ess = 1.0 / np.sum(w ** 2)
    if return_debug:
        ess_history[0] = init_ess
        weight_stats[0, 0] = np.mean(w)
        weight_stats[0, 1] = np.min(w)
        weight_stats[0, 2] = np.max(w)
        loglik_increment[0] = log_sum
        cumulative_loglik[0] = loglik
    with objmode():
        if verbose:
            print(f"t=0 initialized weights: ess={init_ess:.2f}, loglik={loglik:.6f}")
            print(f"t=0 weight stats: mean={np.mean(w):.4e}, min={np.min(w):.4e}, max={np.max(w):.4e}")

    if return_particle_stats:
        means[0] = particles[0].mean(axis=0)
        vars_[0] = particles[0].var(axis=0)
        w_means[0] = np.sum(particles[0] * w[:, None], axis=0)
        w_vars[0] = np.sum(w[:, None] * (particles[0] - w_means[0]) ** 2, axis=0)
        with objmode():
            if verbose:
                print(f"t=0 particle mean={means[0]}, var={vars_[0]}, w_mean={w_means[0]}, w_var={w_vars[0]}")

    # 滤波循环
    for t in range(1, T):
        # 计算 ESS
        ess = 1.0 / np.sum(weights[t-1] ** 2)
        if return_debug:
            ess_history[t] = ess
        with objmode():
            if verbose:
                print(f"t={t} previous ess={ess:.2f} (threshold={resample_threshold * N:.2f})")

        if resample_always or ess < resample_threshold * N:
            # systematic_resampling expected signature: (particles_list, weights_list, num_samples)
            # It returns (indices, resampled_particles_list). We pass the current particle set
            # (as a list) and the current weights to obtain both indices and the resampled
            # particle states. Convert resampled list back to ndarray for downstream use.
            idx, resampled_particles = systematic_resampling(
                list(particles[t-1]),
                list(weights[t-1]),
                N,
            )
            try:
                particles_prev = np.array(resampled_particles)
            except Exception:
                particles_prev = particles[t-1][idx]
            prev_weights = np.ones(N) / N
            log_prev = np.log(prev_weights)
            # record some of the resampled indices (head)
            head = idx[:10].tolist() if hasattr(idx, 'tolist') else list(idx[:10])
            if return_debug:
                resampled_flags[t] = True
                for i, val in enumerate(head):
                    resample_head[t, i] = int(val)
            with objmode():
                if verbose:
                    print(f"t={t} resampled (always={resample_always}, ess={ess:.2f}). sample indices head={head}")
        else:
            particles_prev = particles[t-1]
            prev_weights = weights[t-1]
            log_prev = np.log(prev_weights)
            if return_debug:
                resampled_flags[t] = False
            with objmode():
                if verbose:
                    print(f"t={t} no resample performed")

        # 状态传播
        try:
            particles[t] = transition_fn(particles_prev, theta)
        except Exception as e:
            assert False, f"Error in transition_fn at t={t}: {e}"

        # 计算观测 log 权重（包含先验/以前权重）
        obs_loglik = obs_loglik_fn(x_obs[t], particles[t], theta)
        assert obs_loglik is not None, f"obs_loglik_fn returned None for t={t}"
        if return_debug:
            obs_loglik_stats[t, 0] = np.mean(obs_loglik)
            obs_loglik_stats[t, 1] = np.min(obs_loglik)
            obs_loglik_stats[t, 2] = np.max(obs_loglik)
        with objmode():
            if verbose:
                print(f"t={t} obs_loglik stats: mean={np.mean(obs_loglik):.4e}, min={np.min(obs_loglik):.4e}, max={np.max(obs_loglik):.4e}")

        log_w_raw = obs_loglik + log_prev  # log p(x_t|z_t) + log(prev_weight)
        max_log = np.max(log_w_raw)
        log_sum = max_log + np.log(np.sum(np.exp(log_w_raw - max_log)))  # logsumexp
        # 增加对数似然估计（注意此处 log_sum 已含 prev_weights）
        loglik += log_sum
        w = np.exp(log_w_raw - log_sum)
        weights[t] = w

        if return_debug:
            loglik_increment[t] = log_sum
            cumulative_loglik[t] = loglik
            weight_stats[t, 0] = np.mean(w)
            weight_stats[t, 1] = np.min(w)
            weight_stats[t, 2] = np.max(w)
            ess_history[t] = 1.0 / np.sum(w ** 2)
        with objmode():
            if verbose:
                print(f"t={t} updated: loglik_increment={log_sum:.6f}, cumulative_loglik={loglik:.6f}, ess={1.0 / np.sum(w ** 2):.2f}")
                print(f"t={t} weight stats: mean={np.mean(w):.4e}, min={np.min(w):.4e}, max={np.max(w):.4e}")

        # record per-generation particle statistics (unweighted and weighted)
        if return_particle_stats:
            means[t] = particles[t].mean(axis=0)
            vars_[t] = particles[t].var(axis=0)
            w_means[t] = np.sum(particles[t] * w[:, None], axis=0)
            w_vars[t] = np.sum(w[:, None] * (particles[t] - w_means[t]) ** 2, axis=0)
            if return_debug:
                with objmode():
                    if verbose:
                        print(f"t={t} particle mean={means[t]}, var={vars_[t]}, w_mean={w_means[t]}, w_var={w_vars[t]}")

    with objmode():
        if verbose:
            print(f"particle_filter finished: final_loglik={loglik:.6f}")

    if return_debug:
        # Consolidate history into a tuple of numpy arrays (numba-friendly)
        # Order: ess_history, resampled_flags, resample_head, obs_loglik_stats,
        # weight_stats, loglik_increment, cumulative_loglik, init_samples_shape_arr, state_dim_arr, final_loglik
        history = (
            ess_history,
            resampled_flags,
            resample_head,
            obs_loglik_stats,
            weight_stats,
            loglik_increment,
            cumulative_loglik,
            init_samples_shape_arr,
            state_dim_arr,
            np.array(loglik, dtype=np.float64),
        )

    if return_particle_stats:
        stats = (means, vars_, w_means, w_vars)
        if return_debug:
            return particles, weights, loglik, stats, history
        return particles, weights, loglik, stats
    if return_debug:
        return particles, weights, loglik, history
    return particles, weights, loglik

def particle_smoothing(z_particles, w_filter, transition_prob_fn):
    """
    FFBSm 粒子平滑
    z_particles: (T, N, state_dim)
    w_filter: (T, N) 滤波权重
    transition_prob_fn: 函数 p(z_next | z_current)

    w_{t|T}^{(i)} \propto w_t^{(i)} \sum_{j=1}^N \frac{p(z_{t+1}^{(j)} \mid z_t^{(i)}) \, w_{t+1|T}^{(j)}}{\sum_{m=1}^N p(z_{t+1}^{(j)} \mid z_t^{(m)})\, w_t^{(m)}}

    Returns:
      w_smooth: (T, N) 平滑边际权重
      w_joint: (T-1, N, N) 平滑联合权重
    """
    T, N, _ = z_particles.shape
    w_smooth = np.zeros_like(w_filter)
    w_joint = np.zeros((T-1, N, N))
    
    # 初始化
    w_smooth[-1] = w_filter[-1]
    
    # Backward
    for t in reversed(range(T-1)):
        # 计算归一化常数 C_j
        C = np.zeros(N)
        for j in range(N):
            C[j] = np.sum(transition_prob_fn(z_particles[t+1, j], z_particles[t]) * w_filter[t]) # TODO: 向量化支持，传入的 transition_prob_fn 必须能够处理向量化输入
        
        # 更新平滑权重
        for i in range(N):
            total = 0.0
            for j in range(N):
                p_trans = transition_prob_fn(z_particles[t+1, j], z_particles[t, i])
                total += p_trans * w_smooth[t+1, j] / C[j]
                w_joint[t, i, j] = w_filter[t, i] * p_trans * w_smooth[t+1, j] / C[j]
            w_smooth[t, i] = w_filter[t, i] * total
        
        # 归一化
        w_smooth[t] /= np.sum(w_smooth[t])
        w_joint[t] /= np.sum(w_joint[t])
    
    return w_smooth, w_joint


def systematic_resampling(
    particles: List[dict], 
    weights: List[float], 
    num_samples: int
) -> List[dict]:
    """
    Perform systematic resampling on a set of particles based on their weights.
    
    Parameters:
    - particles: A list of particles (dictionaries).
    - weights: A list of weights corresponding to the particles.
    - num_samples: The number of samples to draw.
    
    Returns:
    - A list of resampled particles.
    """
    cumulative_weights = np.cumsum(weights, dtype=np.float64)
    cumulative_weights /= cumulative_weights[-1]

    u0 = np.random.uniform(0, 1/num_samples)
    positions = u0 + (np.arange(num_samples) / num_samples)

    indices = np.searchsorted(cumulative_weights, positions)
    return indices, [particles[i] for i in indices]

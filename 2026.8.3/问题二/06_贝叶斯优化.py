"""
贝叶斯优化 v7 — GP+EI + 锚点搜索 (使用 common.py 共享物理模型)
============================================================
策略: Phase 0 网格搜索+NM找锚点 → Phase 1 局部BO (GP+EI)精细优化

参数化: [t_det, A_x, A_y, A_z] 直接表示起爆点位置。
  不可行点 → 平滑惩罚 → GP可学习可行性边界梯度。

物理模型: 统一使用 common.py (与问题一/三/四一致)
"""

import sys, os, time, math, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (
    FY1_0, G, V_MIN, V_MAX, TL,
    compute_T_eff, drone_feasible,
)
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.stats import norm

DRONE_Z = 1800.  # FY1飞行高度

# ======================== 平滑约束惩罚 ========================
def constraint_penalty(x):
    t_det, A_x, A_y, A_z = x
    pen = 0.0
    if A_z < 50:   pen += (50 - A_z) / 50
    if A_z > 1795: pen += (A_z - 1795) / 100
    dx = A_x - FY1_0[0]; dy = A_y - FY1_0[1]
    dist_h = math.sqrt(dx*dx + dy*dy)
    if t_det > 1e-6:
        v_needed = dist_h / t_det
        if v_needed < V_MIN: pen += (V_MIN - v_needed) / V_MIN
        if v_needed > V_MAX: pen += (v_needed - V_MAX) / V_MAX
        t_fall_val = math.sqrt(max(0, 2*(DRONE_Z - A_z)/G))
        t_rel = t_det - t_fall_val
        if t_rel < 0: pen += abs(t_rel) / max(1.0, t_det)
    return pen

def objective(x):
    """BO目标函数: 返回 T_eff"""
    t_det = x[0]; Ad = np.array([x[1], x[2], x[3]])
    feas = drone_feasible(Ad, t_det)
    if feas is None:
        return -constraint_penalty(x)
    try:
        return max(0.0, compute_T_eff(Ad, t_det))
    except Exception:
        return -constraint_penalty(x)

# ======================== 高斯过程 ========================
def rbf_kernel_batch(X1, X2, ls, sigma2):
    X1_s = X1 / ls[None, :]; X2_s = X2 / ls[None, :]
    sqdist = (np.sum(X1_s**2, axis=1)[:, None]
              + np.sum(X2_s**2, axis=1)[None, :]
              - 2 * X1_s @ X2_s.T)
    return sigma2 * np.exp(-0.5 * np.clip(sqdist, 0, None))

class GP:
    def __init__(self, ls, sigma2, noise=1e-6):
        self.ls = np.asarray(ls); self.sigma2 = sigma2; self.noise = noise
        self.X = None; self.y = None; self.L = None; self.alpha = None
        self._ym = 0.0; self._ys = 1.0

    def fit(self, X, y):
        self.X = np.asarray(X); raw = np.asarray(y)
        self._ym = np.mean(raw); self._ys = max(np.std(raw), 1e-8)
        self.y = (raw - self._ym) / self._ys
        n = len(self.X)
        K = rbf_kernel_batch(self.X, self.X, self.ls, self.sigma2)
        K += np.eye(n) * self.noise
        for attempt in range(3):
            try:
                self.L = cholesky(K, lower=True)
                break
            except np.linalg.LinAlgError:
                K += np.eye(n) * (10 ** (-6 + attempt*2))
        self.alpha = solve_triangular(self.L.T,
                        solve_triangular(self.L, self.y, lower=True))

    def predict_batch(self, X_pred, batch_size=2000):
        X_pred = np.asarray(X_pred); n_pred = len(X_pred)
        mu = np.zeros(n_pred); sigma_sq = np.zeros(n_pred)
        for start in range(0, n_pred, batch_size):
            end = min(start+batch_size, n_pred); Xb = X_pred[start:end]
            K_s = rbf_kernel_batch(Xb, self.X, self.ls, self.sigma2)
            mu[start:end] = K_s @ self.alpha
            v = solve_triangular(self.L, K_s.T, lower=True)
            sigma_sq[start:end] = self.sigma2 - np.sum(v**2, axis=0)
        return mu*self._ys+self._ym, np.sqrt(np.maximum(sigma_sq,1e-12))*self._ys

def maximize_ei(gp, y_best, lb, ub, n_candidates=30000):
    d = len(lb)
    X_cand = np.random.uniform(lb, ub, (n_candidates, d))
    mu, sigma = gp.predict_batch(X_cand, batch_size=3000)
    sigma = np.maximum(sigma, 1e-12)
    delta = mu - y_best - 0.01
    Z = delta / sigma
    ei = delta * norm.cdf(Z) + sigma * norm.pdf(Z)
    best = np.argmax(ei)
    return X_cand[best], ei[best]

def bayesian_optimize(obj_fn, lb, ub, n_init=30, n_iter=120, seed=42,
                       known_points=None, verbose=True):
    np.random.seed(seed)
    d = len(lb)
    X = np.zeros((n_init, d))
    for j in range(d):
        perm = np.random.permutation(n_init)
        X[:, j] = lb[j] + (ub[j]-lb[j]) * (perm + np.random.uniform(size=n_init)) / n_init
    if known_points:
        for i, (xp, _) in enumerate(known_points):
            if i < n_init: X[i] = np.array(xp)
    y = np.array([obj_fn(x) for x in X])
    best_idx = np.argmax(y); y_best = y[best_idx]; x_best = X[best_idx].copy()
    n_feas_init = np.sum(y > 0)
    if verbose:
        print(f"  初始{n_init}点 | 可行={n_feas_init}/{n_init} | 最优={y_best:.4f}s")
    gp = GP(ls=(ub-lb)*0.2, sigma2=2.0, noise=1e-3)
    t0 = time.time()
    for it in range(1, n_iter+1):
        gp.fit(X, y)
        x_next, ei_val = maximize_ei(gp, y_best, lb, ub)
        y_next = obj_fn(x_next)
        X = np.vstack([X, x_next]); y = np.append(y, y_next)
        if y_next > y_best: y_best = y_next; x_best = x_next.copy()
        if verbose and (it % 25 == 0 or it == 1 or it == n_iter):
            elapsed = time.time() - t0
            n_feas = np.sum(y > 0)
            print(f"  iter {it:3d}/{n_iter} | best={y_best:.4f}s | "
                  f"feas={n_feas}/{len(y)} | {elapsed:.1f}s")
    elapsed = time.time() - t0
    if verbose: print(f"\n  BO完成: {len(X)}次评估 / {elapsed:.1f}s")
    return x_best, y_best

# ======================== 锚点搜索 ========================
def grid_search_anchor(n_grid=6, verbose=True):
    """粗粒度4D网格搜索寻找可行锚点。"""
    td_vals = np.linspace(2.0, 4.3, n_grid)
    ax_vals = np.linspace(17550., 17700., n_grid)
    ay_vals = np.linspace(-15., 15., n_grid)
    az_vals = np.linspace(1740., 1790., n_grid)

    best_T = -1.0; best_x = None; n_feas = 0
    for td in td_vals:
        for Ax in ax_vals:
            for Ay in ay_vals:
                for Az in az_vals:
                    Ad = np.array([Ax, Ay, Az])
                    feas = drone_feasible(Ad, td)
                    if feas is None: continue
                    n_feas += 1
                    try: T = compute_T_eff(Ad, td)
                    except Exception: continue
                    if T > best_T:
                        best_T = T; best_x = np.array([td, Ax, Ay, Az])

    if verbose:
        print(f"  4D网格({n_grid}^4={n_grid**4}点): 可行={n_feas} | 最优T_eff={best_T:.4f}s")
        if best_x is not None:
            print(f"    锚点: td={best_x[0]:.3f}s A=({best_x[1]:.0f},{best_x[2]:.0f},{best_x[3]:.0f})")
    return best_x, best_T


def local_refine(x0, verbose=True):
    """纯Python坐标下降局部精化 — 避免scipy NM触发的segfault"""
    x = x0.copy().astype(float)
    t_det = x[0]; Ad = np.array([x[1], x[2], x[3]])
    feas0 = drone_feasible(Ad, t_det)
    best_T = compute_T_eff(Ad, t_det) if feas0 else 0.0
    init_T = best_T

    # 步长: [td, Ax, Ay, Az]
    steps = np.array([[0.05, 0, 0, 0], [0, 50, 0, 0], [0, 0, 5, 0], [0, 0, 0, 10]])
    for round_idx in range(3):  # 3轮, 每轮步长减半
        improved = True
        while improved:
            improved = False
            for direction in [1, -1]:
                for i in range(4):
                    candidate = x + direction * steps[i]
                    td_c = candidate[0]; Ad_c = np.array([candidate[1], candidate[2], candidate[3]])
                    feas = drone_feasible(Ad_c, td_c)
                    if feas is None: continue
                    try:
                        T_c = compute_T_eff(Ad_c, td_c)
                    except Exception:
                        continue
                    if T_c > best_T:
                        best_T = T_c; x = candidate.copy()
                        improved = True
        steps *= 0.5

    td = x[0]; Ad = np.array([x[1], x[2], x[3]])
    feas = drone_feasible(Ad, td)
    T_opt = best_T

    if verbose:
        print(f"  局部精化: T_eff={T_opt:.4f}s (+{T_opt - init_T:.4f}s vs 网格)")
        print(f"    精化解: td={td:.3f}s A=({Ad[0]:.0f},{Ad[1]:.0f},{Ad[2]:.0f})")
        if feas:
            print(f"    v={feas['v']:.1f}m/s θ={np.degrees(feas['theta']):.1f}° "
                  f"t_rel={feas['t_release']:.2f}s t_fall={feas['t_fall']:.2f}s")
    return x, T_opt


def find_anchor(verbose=True):
    """完整锚点搜索流水线: 网格搜索 → Nelder-Mead精化"""
    if verbose:
        print(f"\n{'─'*70}")
        print("Phase 0: 锚点搜索 — 网格搜索 + Nelder-Mead")
        print(f"{'─'*70}")
        print("  [1/2] 4D网格搜索...")

    grid_x, grid_T = grid_search_anchor(verbose=verbose)

    if grid_x is None:
        if verbose: print("  [FALLBACK] 网格未找到可行点")
        return None, 0.0, None, 0.0

    if verbose: print("  [2/2] 坐标下降局部精化...")
    nm_x, nm_T = local_refine(grid_x, verbose=verbose)

    # 验证NM结果合理性 (允许提升, 仅拒绝明显异常的数值溢出)
    if nm_T > TL or nm_T < 0:
        if verbose:
            print(f"  [ADJUST] NM结果异常({nm_T:.3f}s), 回退至网格锚点")
        nm_x = grid_x.copy()
        nm_T = grid_T

    return nm_x, nm_T, grid_x, grid_T


# ======================== 主程序 ========================
if __name__ == "__main__":
    import json as _json
    t_start = time.time()
    print("="*70)
    print("问题2: 贝叶斯优化 v7 — 锚点搜索+GP+EI (common.py)")
    print("="*70)

    # ===== Phase 0: 锚点搜索 (必须成功) =====
    nm_x, nm_T, grid_x, grid_T = find_anchor(verbose=True)
    if nm_x is None:
        print("  [FATAL] 锚点搜索失败!"); exit(1)

    # 保存锚点结果 (防后续BO崩溃丢失数据)
    anchor_data = {
        'td': float(nm_x[0]), 'Ax': float(nm_x[1]), 'Ay': float(nm_x[2]), 'Az': float(nm_x[3]),
        'T_eff': float(nm_T), 'grid_T_eff': float(grid_T),
        'grid_x': [float(grid_x[0]), float(grid_x[1]), float(grid_x[2]), float(grid_x[3])],
    }
    feas_a = drone_feasible(np.array([nm_x[1], nm_x[2], nm_x[3]]), nm_x[0])
    if feas_a:
        anchor_data['v'] = float(feas_a['v'])
        anchor_data['theta_deg'] = float(np.degrees(feas_a['theta']) % 360)
        anchor_data['t_release'] = float(feas_a['t_release'])
        anchor_data['t_fall'] = float(feas_a['t_fall'])

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'anchor_result.json')
    with open(out_path, 'w') as f:
        _json.dump(anchor_data, f, indent=2)
    print(f"\n  [OK] 锚点已保存到 {out_path}")
    print(f"  T_eff={nm_T:.4f}s  td={nm_x[0]:.3f}s  A=({nm_x[1]:.0f},{nm_x[2]:.0f},{nm_x[3]:.0f})")

    anchor_tuple = (nm_x.tolist(), nm_T)

    # ===== Phase 1: BO实验 (Python 3.14可能segfault, 用轻量参数) =====
    print(f"\n{'─'*70}")
    print("Phase 1: BO实验 (轻量参数, Python 3.14 segfault保护)")
    print(f"{'─'*70}")

    center = nm_x.copy()
    delta  = np.array([3.0, 300.0, 200.0, 300.0])
    lb1 = np.maximum([0.5, 15700, -500, 200], center - delta)
    ub1 = np.minimum([15.0, 17800,  500, 1790], center + delta)

    known = [anchor_tuple, (grid_x.tolist(), grid_T)]

    # 用轻量参数避免内存压力触发 segfault
    n_init_b = 15; n_iter_b = 60; n_cand_b = 10000
    bo_success = False
    try:
        print(f"  实验1: 局部BO (n_init={n_init_b}, n_iter={n_iter_b}, n_cand={n_cand_b})")
        x1, y1 = bayesian_optimize(
            objective, lb1, ub1, n_init=n_init_b, n_iter=n_iter_b, seed=42,
            known_points=known, verbose=True)
        bo_success = True
        t1 = x1[0]; Ad1 = np.array([x1[1], x1[2], x1[3]])
        feas1 = drone_feasible(Ad1, t1)
        print(f"\n  局部BO最优: T_eff={y1:.4f}s  td={t1:.3f}s  A=({Ad1[0]:.0f},{Ad1[1]:.0f},{Ad1[2]:.0f})")
    except Exception as e:
        print(f"  [SKIP] BO实验1异常: {e}")
        y1 = nm_T; x1 = nm_x

    print(f"\n{'='*70}")
    print(f"最终结果")
    print(f"{'='*70}")
    print(f"  {'方法':<40s} {'T_eff':>8s}")
    print(f"  {'─'*60}")
    print(f"  {'Grid搜索 (锚点)':<40s} {grid_T:>8.4f}s")
    print(f"  {'Grid+局部精化 (锚点)':<40s} {nm_T:>8.4f}s")
    if bo_success:
        print(f"  {'BO (GP+EI)':<40s} {y1:>8.4f}s")

    best_dur = max(nm_T, y1) if bo_success else nm_T
    best_x = x1 if (bo_success and y1 > nm_T) else nm_x
    t_best = best_x[0]; Ad_best = np.array([best_x[1], best_x[2], best_x[3]])
    feas_best = drone_feasible(Ad_best, t_best)

    print(f"\n  ★ 全局最优解: T_eff = {best_dur:.4f}s")
    print(f"    t_det={t_best:.3f}s  A=({Ad_best[0]:.0f},{Ad_best[1]:.0f},{Ad_best[2]:.0f})")
    if feas_best:
        print(f"    v={feas_best['v']:.1f}m/s  θ={np.degrees(feas_best['theta']):.1f}°  "
              f"t_rel={feas_best['t_release']:.2f}s  t_fall={feas_best['t_fall']:.2f}s")
        # 更新保存最优结果
        anchor_data['td'] = float(t_best)
        anchor_data['Ax'] = float(Ad_best[0]); anchor_data['Ay'] = float(Ad_best[1]); anchor_data['Az'] = float(Ad_best[2])
        anchor_data['T_eff'] = float(best_dur)
        anchor_data['v'] = float(feas_best['v'])
        anchor_data['theta_deg'] = float(np.degrees(feas_best['theta']) % 360)
        anchor_data['t_release'] = float(feas_best['t_release'])
        anchor_data['t_fall'] = float(feas_best['t_fall'])
        with open(out_path, 'w') as f:
            _json.dump(anchor_data, f, indent=2)

    print(f"\n  总耗时: {time.time()-t_start:.1f}s")

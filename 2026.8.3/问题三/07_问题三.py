"""
问题3: FY1投放3枚烟幕弹干扰M1
==============================
复用问题二的 GP+EI 贝叶斯优化框架，扩展到8D联合优化。

核心约束:
  - v, θ 是优化变量，3枚弹共享（无人机方向和速度一经确定不可变）
  - 单枚弹 T_eff ≤ 4.5064s (P2单弹最优是理论上界)
  - 投放时序: t_det1 < t_det2 < t_det3, 投放间隔 ≥ 1s

锚点策略:
  - 使用P2精确最优参数 v=70.38, θ=178.55°
  - Bomb1 = P2最优 (t_det=2.80, t_delay=2.673) → T_eff=4.5064s
  - Bomb2/3 在Bomb1时间窗口附近合理放置
"""

import sys, os, time, gc, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *
import numpy as np
from scipy.optimize import bisect
from scipy.linalg import cholesky, solve_triangular
from scipy.stats import norm

# ======================== P2最优解 (理论上界) ========================
# 验证: drone_feasible(Ad=[17603,5,1765], td=2.80, drone_start=FY1_0)
#   → v=70.3798, θ=178.5461°, t_delay=2.6726, T_eff=4.5064s
T_MAX_SINGLE = 4.5064  # 单弹遮蔽时长理论上界

# ======================== 目标函数 ========================

def obj_q3(x):
    """x = [v, theta, t_det1, t_delay1, t_det2, t_delay2, t_det3, t_delay3] (8D)

    返回: 总遮蔽时长 - 单弹超界软约束惩罚 (最大化)
    """
    v, theta = float(x[0]), float(x[1])

    # v 边界检查
    if v < V_MIN or v > V_MAX:
        return -abs(v - np.clip(v, V_MIN, V_MAX)) / V_MIN - 0.1

    intervals_list = []
    T_effs = []
    t_rels = []

    for i in range(3):
        t_det = float(x[2 + 2*i])
        t_delay = float(x[3 + 2*i])

        if t_det <= 0.1 or t_delay < 0.001:
            return -1.0

        # Az 计算和边界检查
        Az = 1800. - 0.5 * G * t_delay**2
        if Az < 50 or Az > 1798:
            return -(min(abs(Az - 50), abs(Az - 1798)) / 100 + 0.1)

        t_rel = t_det - t_delay
        if t_rel < 0:
            return -abs(t_rel) - 0.1
        t_rels.append(t_rel)

        # 起爆点坐标 (无人机投放后, 炸弹自由落体 t_delay 秒后在 Ad 起爆)
        Ax = FY1_0[0] + v * np.cos(theta) * t_det
        Ay = FY1_0[1] + v * np.sin(theta) * t_det
        Ad = np.array([Ax, Ay, Az])

        # 计算单弹遮蔽区间和时长
        ivs = compute_T_eff_intervals(Ad, t_det, 'M1')
        T_single = sum(e - s for s, e in ivs)
        if T_single < 1e-6:
            # fallback: 区间法可能遗漏, 用步进法补算
            T_single = compute_T_eff(Ad, t_det, 'M1')

        T_effs.append(T_single)
        intervals_list.append(ivs)

    # 投放时序约束: t_det递增
    if not (x[2] < x[4] < x[6]):
        return -0.5

    # 投放间隔 ≥ 1s (基于释放时间 t_rel)
    for i in range(1, 3):
        gap = t_rels[i] - t_rels[i-1]
        if gap < 1.0 - 1e-4:
            return -(1.0 - gap) - 0.1

    # 软约束惩罚: 任意单枚弹 T_eff > P2最优上界
    # 物理上3枚弹共享v,θ → 单弹不可能超越P2最优
    penalty = 0.0
    for i, T_s in enumerate(T_effs):
        if T_s > T_MAX_SINGLE:
            excess = T_s - T_MAX_SINGLE
            penalty += 10.0 * excess

    total_T = union_duration(intervals_list)
    return total_T - penalty


# ======================== BO 组件 (从问题二复用, 适配8D) ========================

def rbf_kernel_batch(X1, X2, ls, sigma2):
    X1_s = X1 / ls[None, :]
    X2_s = X2 / ls[None, :]
    sqdist = (np.sum(X1_s**2, axis=1)[:, None]
              + np.sum(X2_s**2, axis=1)[None, :]
              - 2 * X1_s @ X2_s.T)
    return sigma2 * np.exp(-0.5 * np.clip(sqdist, 0, None))


class GP:
    """轻量级高斯过程回归 (P2同款)"""
    def __init__(self, ls, sigma2, noise=1e-6):
        self.ls = np.asarray(ls)
        self.sigma2 = sigma2
        self.noise = noise
        self.X = None
        self.y = None
        self.L = None
        self.alpha = None
        self._ym = 0.0
        self._ys = 1.0

    def fit(self, X, y):
        self.X = np.asarray(X)
        raw = np.asarray(y)
        self._ym = np.mean(raw)
        self._ys = max(np.std(raw), 1e-8)
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
        X_pred = np.asarray(X_pred)
        n_pred = len(X_pred)
        mu = np.zeros(n_pred)
        sigma = np.zeros(n_pred)
        for start in range(0, n_pred, batch_size):
            end = min(start + batch_size, n_pred)
            Xb = X_pred[start:end]
            K_s = rbf_kernel_batch(Xb, self.X, self.ls, self.sigma2)
            v = solve_triangular(self.L, K_s.T, lower=True)
            mu[start:end] = K_s @ self.alpha
            sigma[start:end] = np.sqrt(np.maximum(
                self.sigma2 - np.sum(v**2, axis=0), 1e-12))
        return mu * self._ys + self._ym, sigma * self._ys


def maximize_ei(gp, y_best, lb, ub, n_candidates=40000):
    """随机搜索最大化 Expected Improvement (8D候选点)"""
    d = len(lb)
    X_cand = np.random.uniform(lb, ub, (n_candidates, d))
    mu, sigma = gp.predict_batch(X_cand, batch_size=3000)
    sigma = np.maximum(sigma, 1e-12)
    delta = mu - y_best - 0.01
    Z = delta / sigma
    ei = delta * norm.cdf(Z) + sigma * norm.pdf(Z)
    best = np.argmax(ei)
    return X_cand[best], ei[best]


def bayesian_optimize(obj_fn, lb, ub, n_init=40, n_iter=100, seed=42,
                       known_points=None, verbose=True):
    """贝叶斯优化主循环 (8D版, Windows segfault保护: 降低迭代数)"""
    np.random.seed(seed)
    d = len(lb)

    # --- LHS 初始化 ---
    X = np.zeros((n_init, d))
    for j in range(d):
        perm = np.random.permutation(n_init)
        X[:, j] = lb[j] + (ub[j] - lb[j]) * (perm + np.random.uniform(size=n_init)) / n_init

    # 注入已知锚点
    if known_points:
        for i, (xp, _) in enumerate(known_points):
            if i < n_init:
                X[i] = xp

    # 后处理: 强制 t_det1 < t_det2 < t_det3 以提高可行率
    for i in range(n_init):
        td = sorted([X[i, 2], X[i, 4], X[i, 6]])
        X[i, 2], X[i, 4], X[i, 6] = td[0], td[1], td[2]

    y = np.array([obj_fn(x) for x in X])
    best_idx = np.argmax(y)
    y_best = y[best_idx]
    x_best = X[best_idx].copy()

    n_feas_init = np.sum(y > 0)
    if verbose:
        print(f"  LHS {n_init}pts | feasible={n_feas_init}/{n_init} | best={y_best:.4f}s")

    gp = GP(ls=(ub - lb) * 0.2, sigma2=2.0, noise=1e-3)

    for it in range(1, n_iter + 1):
        gp.fit(X, y)
        x_next, ei_val = maximize_ei(gp, y_best, lb, ub)
        y_next = obj_fn(x_next)

        X = np.vstack([X, x_next])
        y = np.append(y, y_next)

        if y_next > y_best:
            y_best = y_next
            x_best = x_next.copy()

        if verbose and (it % 40 == 0 or it == 1 or it == n_iter):
            n_feas = np.sum(y > 0)
            print(f"  iter {it:3d}/{n_iter} | EI={ei_val:.4f} | "
                  f"best={y_best:.4f}s | feas={n_feas}/{len(y)}")

        # Windows Python 3.14 + scipy 可能内存碎片化, 分批gc
        if it % 30 == 0:
            gc.collect()

    if verbose:
        print(f"  BO done: {len(X)} evals | best={y_best:.4f}s")
    return x_best, y_best


# ======================== 辅助函数: 提取bombs_info ========================
def extract_bombs_info(x_opt, v_opt, th_opt):
    """从优化变量提取炸弹信息列表"""
    bombs_info = []
    for i in range(3):
        td = float(x_opt[2 + 2*i])
        tdl = float(x_opt[3 + 2*i])
        Ax = FY1_0[0] + v_opt * np.cos(th_opt) * td
        Ay = FY1_0[1] + v_opt * np.sin(th_opt) * td
        Az = 1800. - 0.5 * G * tdl**2
        Ad = np.array([Ax, Ay, Az])
        tr = td - tdl
        T_s = compute_T_eff(Ad, td, 'M1')
        ivs = compute_T_eff_intervals(Ad, td, 'M1')
        bombs_info.append({
            'Ad': Ad, 't_det': td, 't_delay': tdl,
            't_release': tr, 'T_eff': T_s, 'intervals': ivs
        })
    return bombs_info


def save_to_xlsx(bombs_info, v_opt, th_opt):
    """填入结果到 附件/result1.xlsx (保留原模板中文格式)"""
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xlsx_path = os.path.join(proj_root, '附件', 'result1.xlsx')
    import openpyxl
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb['Sheet1']

    for i, bi in enumerate(bombs_info):
        row = 2 + i
        t_rel = bi['t_release']
        rx = FY1_0[0] + v_opt * np.cos(th_opt) * t_rel
        ry = FY1_0[1] + v_opt * np.sin(th_opt) * t_rel
        rz = 1800.0
        dx, dy, dz = bi['Ad'][0], bi['Ad'][1], bi['Ad'][2]

        ws.cell(row=row, column=2,  value=round(np.degrees(th_opt), 4))
        ws.cell(row=row, column=3,  value=round(v_opt, 2))
        ws.cell(row=row, column=5,  value=round(rx, 1))
        ws.cell(row=row, column=6,  value=round(ry, 1))
        ws.cell(row=row, column=7,  value=round(rz, 1))
        ws.cell(row=row, column=8,  value=round(dx, 1))
        ws.cell(row=row, column=9,  value=round(dy, 1))
        ws.cell(row=row, column=10, value=round(dz, 1))
        ws.cell(row=row, column=11, value=round(bi['T_eff'], 4))
        ws.cell(row=row, column=12, value='M1')

    wb.save(xlsx_path)
    return xlsx_path


def print_results(bombs_info, v_opt, th_opt, label="RESULT"):
    """打印结果表格"""
    all_intervals = [bi['intervals'] for bi in bombs_info]
    total_u = union_duration(all_intervals)
    flight_time = np.linalg.norm(M1_0 - O_DECOY) / VM

    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'='*70}")
    print(f"  Total T_eff (union): {total_u:.4f}s")
    print(f"  Drone: v={v_opt:.2f} m/s  theta={np.degrees(th_opt):.4f} deg")
    print(f"")
    print(f"  {'Bomb':>5s} {'t_rel':>8s} {'t_det':>8s} {'t_delay':>8s} "
          f"{'Ax':>9s} {'Ay':>8s} {'Az':>7s} {'T_single':>9s}  Intervals")
    print(f"  {'-'*85}")
    for i, bi in enumerate(bombs_info):
        iv_str = '; '.join([f"[{s:.3f},{e:.3f}]" for s, e in bi['intervals']])
        print(f"  {i+1:5d} {bi['t_release']:8.4f} {bi['t_det']:8.4f} "
              f"{bi['t_delay']:8.4f} {bi['Ad'][0]:9.1f} {bi['Ad'][1]:8.1f} "
              f"{bi['Ad'][2]:7.1f} {bi['T_eff']:9.4f}  {iv_str}")

    print(f"\n  Union total: {total_u:.4f}s")
    print(f"  M1 flight: {flight_time:.1f}s | Coverage: {total_u/flight_time*100:.1f}%")

    # 物理约束
    print(f"\n  Physics Check:")
    for i, bi in enumerate(bombs_info):
        ok = "OK" if bi['T_eff'] <= T_MAX_SINGLE + 1e-6 else "EXCEEDS!"
        print(f"    Bomb{i+1} T_single={bi['T_eff']:.4f}s <= {T_MAX_SINGLE}s [{ok}]")

    print(f"\n  Comparison:")
    print(f"    P1 (fixed, 1 bomb):     1.3900s")
    print(f"    P2 (optimal, 1 bomb):    {T_MAX_SINGLE:.4f}s")
    print(f"    P3 (3 bombs, union):     {total_u:.4f}s  ({total_u/T_MAX_SINGLE:.2f}x P2)")

    return total_u


# ======================== 主程序 ========================
if __name__ == "__main__":
    t_start = time.time()
    print("=" * 70)
    print("Problem 3: FY1 x 3 bombs -> M1 | BO (GP+EI, 8D)")
    print("=" * 70)

    # ---- Phase 0: 锚点计算 (先保存防segfault) ----
    print("\n[Phase 0] Computing anchor from P2 optimal...")

    v_p2 = 70.3798
    th_p2 = np.radians(178.5461)

    x_anchor = np.array([
        v_p2, th_p2,
        2.80, 2.6726,   # Bomb1 = P2最优
        4.50, 3.20,     # Bomb2
        6.50, 3.80,     # Bomb3
    ])

    y_anchor = obj_q3(x_anchor)
    bombs_anchor = extract_bombs_info(x_anchor, v_p2, th_p2)
    total_anchor = print_results(bombs_anchor, v_p2, th_p2, "PHASE 0: ANCHOR (P2-based)")

    # 先保存锚点结果 (即使后续BO segfault也有结果)
    saved_path = save_to_xlsx(bombs_anchor, v_p2, th_p2)
    print(f"\n  [OK] Anchor saved to {saved_path}")

    # ---- Phase 1: BO优化 (可选, 可能segfault) ----
    print(f"\n{'─'*70}")
    print("[Phase 1] Bayesian Optimization (8D) — may segfault on scipy 1.17.1")
    print(f"{'─'*70}")

    xc = x_anchor.copy()
    delta_bo = np.array([
        5.0, np.radians(5.0),
        1.5, 1.0,
        3.0, 2.0,
        5.0, 2.5,
    ])
    lb_bo = np.maximum(xc - delta_bo * 3, [
        70.0, np.radians(160.0), 0.5, 0.05, 2.0, 0.5, 3.0, 0.5,
    ])
    ub_bo = np.minimum(xc + delta_bo * 3, [
        140.0, np.radians(200.0), 5.0, 5.0, 10.0, 5.5, 15.0, 6.0,
    ])

    print(f"  Bounds: v[{lb_bo[0]:.0f},{ub_bo[0]:.0f}] "
          f"theta[{np.degrees(lb_bo[1]):.0f},{np.degrees(ub_bo[1]):.0f}]deg "
          f"t_det1[{lb_bo[2]:.1f},{ub_bo[2]:.1f}] "
          f"t_det2[{lb_bo[4]:.1f},{ub_bo[4]:.1f}] "
          f"t_det3[{lb_bo[6]:.1f},{ub_bo[6]:.1f}]")

    gc.collect()
    bo_improved = False
    try:
        x_bo, y_bo = bayesian_optimize(
            obj_q3, lb_bo, ub_bo, n_init=40, n_iter=100, seed=42,
            known_points=[(x_anchor, y_anchor)], verbose=True
        )
        gc.collect()

        if y_bo > y_anchor:
            bo_improved = True
            v_bo, th_bo = float(x_bo[0]), float(x_bo[1])
            bombs_bo = extract_bombs_info(x_bo, v_bo, th_bo)
            total_bo = print_results(bombs_bo, v_bo, th_bo, "PHASE 1: BO OPTIMAL")
            save_to_xlsx(bombs_bo, v_bo, th_bo)
            print(f"\n  [OK] BO improved: {y_bo:.4f}s > {y_anchor:.4f}s (+{(y_bo-y_anchor):.4f}s)")
        else:
            print(f"\n  [--] BO no improvement: best={y_bo:.4f}s <= anchor={y_anchor:.4f}s")
    except Exception as e:
        print(f"\n  [WARN] BO crashed: {e}")
        print(f"  [INFO] Anchor results already saved — no data lost.")

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"Problem 3 complete! ({elapsed:.1f}s total, BO={'improved' if bo_improved else 'anchor-used'})")
    print(f"{'='*70}")

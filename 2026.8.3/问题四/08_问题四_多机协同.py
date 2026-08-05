"""
问题四：3架无人机(FY1, FY2, FY3)各投放1枚烟幕干扰弹，协同干扰M1
==============================================================
v2: 使用 common.py (numpy物理模型, 与问题二一致)

策略:
  阶段0: 物理引导锚点搜索 — 在导弹-目标连线附近采样
  阶段1: BO(GP+EI) 独立优化FY2/FY3单弹
          (FY1沿用问题二最优解: T_eff≈4.51s)
  阶段2: CMA-ES(cma库)联合优化3架无人机12D协同策略

关键约束: M1飞行时间 T_flight≈67.0s, 烟幕在导弹到达假目标后失效
输出: 附件/result2.xlsx
"""

import sys, os, time, gc, math, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import *
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.stats import norm
import subprocess, json, tempfile

T_FLIGHT = MISSILE_T_FLIGHT['M1']  # ≈ 67.0s

DRONE_CONFIGS = [
    ('FY1', FY1_0, 1800.0),
    ('FY2', FY2_0, 1400.0),
    ('FY3', FY3_0, 700.0),
]

# ====================================================================
# FY1已知最优解 (问题二 v7: C2区间修复后)
# ====================================================================
FY1_OPT_X = (3.007, 17525.0, 9.0, 1760.0)  # td, Ax, Ay, Az
FY1_OPT_TEFF = compute_T_eff(np.array([17525.0, 9.0, 1760.0]), 3.007, missile='M1')

# ====================================================================
# ====================================================================
# 约束惩罚与目标函数
# ====================================================================
def constraint_penalty(x, drone_start, drone_z):
    t_det, A_x, A_y, A_z = float(x[0]), float(x[1]), float(x[2]), float(x[3])
    pen = 0.0
    if A_z < 50:   pen += (50 - A_z) / 50
    if A_z > drone_z - 5: pen += (A_z - (drone_z - 5)) / 100
    dx = A_x - drone_start[0]; dy = A_y - drone_start[1]
    dist_h = math.sqrt(dx*dx + dy*dy)
    if t_det > 1e-6:
        v_needed = dist_h / t_det
        if v_needed < V_MIN: pen += (V_MIN - v_needed) / V_MIN
        if v_needed > V_MAX: pen += (v_needed - V_MAX) / V_MAX
        t_fall_val = math.sqrt(max(0, 2 * (drone_z - A_z) / G))
        t_rel = t_det - t_fall_val
        if t_rel < 0: pen += abs(t_rel) / max(1.0, t_det)
    return pen

def objective_single(x, drone_start, drone_z):
    t_det, Ax, Ay, Az = float(x[0]), float(x[1]), float(x[2]), float(x[3])
    Ad = np.array([Ax, Ay, Az])
    feas = drone_feasible(Ad, t_det, drone_start=drone_start, drone_z=drone_z)
    if feas is None:
        return -constraint_penalty(x, drone_start, drone_z)
    dur = compute_T_eff(Ad, t_det, missile='M1')
    return max(0.0, dur)

# ====================================================================
# 独立进程锚点搜索 (subprocess.run完全隔离segfault)
# ====================================================================
def run_anchor_search(drone_name):
    """
    调用 _run_search.py 在独立进程中运行物理引导搜索.
    conda Python 3.13的numpy segfault只会杀死子进程, 主进程不受影响.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(script_dir, '_run_search.py')

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        out_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, script, drone_name, out_path],
            capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and os.path.exists(out_path):
            with open(out_path, 'r') as f:
                data = json.load(f)
            T = data['best_T']
            bx = data['best_x']
            if T > 0 and bx:
                feas = drone_feasible(np.array([bx['Ax'], bx['Ay'], bx['Az']]),
                                      bx['td'],
                                      drone_start=np.array(data['drone_start']),
                                      drone_z=data['drone_z'])
                x_tuple = (bx['td'], bx['Ax'], bx['Ay'], bx['Az'], feas)
                print(f"  {drone_name}: {data['n_feas']} feas, {data['n_eff']} eff | "
                      f"T_eff={T:.4f}s")
                print(f"       td={bx['td']:.2f}s A=({bx['Ax']:.0f},{bx['Ay']:.0f},{bx['Az']:.0f}) "
                      f"v={bx['v']:.1f}m/s th={bx['theta']:.0f}°")
                return x_tuple, T, data['n_feas'], data['n_eff']
        else:
            stderr_tail = result.stderr[-200:] if result.stderr else '(none)'
            print(f"  [{drone_name} FAIL] rc={result.returncode} stderr={stderr_tail}")
    except subprocess.TimeoutExpired:
        print(f"  [{drone_name} TIMEOUT] 子进程超时(300s)")
    except Exception as e:
        print(f"  [{drone_name} ERROR] {e}")
    finally:
        try: os.unlink(out_path)
        except: pass
        tmp = out_path + '.tmp'
        if os.path.exists(tmp):
            try:
                with open(tmp, 'r') as f:
                    tmp_data = json.load(f)
                print(f"  [{drone_name} PARTIAL] 从中间文件恢复: T_eff={tmp_data.get('best_T',0):.4f}s")
            except: pass
            try: os.unlink(tmp)
            except: pass

    return None, -1.0, 0, 0

# ====================================================================
# GP + BO (与问题二相同)
# ====================================================================
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
                K += np.eye(n) * (10 ** (-6 + attempt * 2))
        self.alpha = solve_triangular(self.L.T,
                                      solve_triangular(self.L, self.y, lower=True))

    def predict_batch(self, X_pred, batch_size=2000):
        X_pred = np.asarray(X_pred); n_pred = len(X_pred)
        mu = np.zeros(n_pred); sigma_sq = np.zeros(n_pred)
        for start in range(0, n_pred, batch_size):
            end = min(start + batch_size, n_pred); Xb = X_pred[start:end]
            K_s = rbf_kernel_batch(Xb, self.X, self.ls, self.sigma2)
            mu[start:end] = K_s @ self.alpha
            v = solve_triangular(self.L, K_s.T, lower=True)
            sigma_sq[start:end] = self.sigma2 - np.sum(v**2, axis=0)
        return (mu * self._ys + self._ym,
                np.sqrt(np.maximum(sigma_sq, 1e-12)) * self._ys)

def maximize_ei(gp, y_best, lb, ub, n_candidates=30000):
    d = len(lb)
    X_cand = np.random.uniform(lb, ub, (n_candidates, d))
    mu, sigma = gp.predict_batch(X_cand, batch_size=5000)
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
        X[:, j] = lb[j] + (ub[j] - lb[j]) * (perm + np.random.uniform(size=n_init)) / n_init
    if known_points:
        for i, (xp, _) in enumerate(known_points):
            if i < n_init: X[i] = np.array(xp)

    y = np.array([obj_fn(x) for x in X])
    best_idx = np.argmax(y); y_best = y[best_idx]; x_best = X[best_idx].copy()
    n_feas_init = np.sum(y > 0)
    if verbose:
        print(f"  LHS {n_init}pts | feas={n_feas_init}/{n_init} | best={y_best:.4f}s")

    gp = GP(ls=(ub - lb) * 0.2, sigma2=2.0, noise=1e-3)
    t0 = time.time()
    for it in range(1, n_iter + 1):
        gp.fit(X, y)
        x_next, ei_val = maximize_ei(gp, y_best, lb, ub)
        y_next = obj_fn(x_next)
        X = np.vstack([X, x_next]); y = np.append(y, y_next)
        if y_next > y_best: y_best = y_next; x_best = x_next.copy()
        if verbose and (it % 30 == 0 or it == 1 or it == n_iter):
            elapsed = time.time() - t0
            n_feas = np.sum(y > 0)
            print(f"  iter {it:3d}/{n_iter} | best={y_best:.4f}s | "
                  f"feas={n_feas}/{len(y)} | {elapsed:.1f}s")
        if it % 50 == 0: gc.collect()
    return x_best, y_best

# ====================================================================
# 联合目标函数 (12D) — 使用common.py的multi_bomb_T_eff
# ====================================================================
def joint_T_eff(x):
    bombs = []
    total_penalty = 0.0
    for i, (name, drone_start, drone_z) in enumerate(DRONE_CONFIGS):
        base = 4 * i
        td = x[base]; Ad = np.array([x[base+1], x[base+2], x[base+3]])
        feas = drone_feasible(Ad, td, drone_start=drone_start, drone_z=drone_z)
        if feas is None:
            total_penalty += constraint_penalty(x[base:base+4], drone_start, drone_z)
        else:
            bombs.append((Ad, td))
    if total_penalty > 0: return -1000.0 - total_penalty
    if len(bombs) < 3: return -1000.0 - (3 - len(bombs))
    try:
        return multi_bomb_T_eff(bombs, 'M1')
    except Exception:
        return -1000.0

def cma_objective(x):
    val = joint_T_eff(x)
    if val <= 0: return 1000.0 - val
    return -val

# ====================================================================
# 结果构建 & 保存
# ====================================================================
def build_result(x_opt):
    bombs_info = []
    for i, (name, drone_start, drone_z) in enumerate(DRONE_CONFIGS):
        base = 4 * i
        td = x_opt[base]; Ad = np.array([x_opt[base+1], x_opt[base+2], x_opt[base+3]])
        feas = drone_feasible(Ad, td, drone_start=drone_start, drone_z=drone_z)
        T_s = compute_T_eff(Ad, td, missile='M1')
        ivs = compute_T_eff_intervals(Ad, td, missile='M1') if T_s > 0 else []
        if feas:
            v = feas['v']; theta_deg = math.degrees(feas['theta']) % 360
            t_rel = feas['t_release']; t_fall = feas['t_fall']
            rx = drone_start[0] + v * math.cos(feas['theta']) * t_rel
            ry = drone_start[1] + v * math.sin(feas['theta']) * t_rel
        else:
            v = theta_deg = t_rel = t_fall = 0.0; rx = ry = 0.0
        bombs_info.append({
            'drone': name, 't_release': t_rel, 't_det': td,
            'release_x': rx, 'release_y': ry, 'release_z': drone_z,
            'A_x': Ad[0], 'A_y': Ad[1], 'A_z': Ad[2],
            'T_eff': T_s, 'v': v, 'theta': theta_deg, 'intervals': ivs,
        })

    all_ivs = [bi['intervals'] for bi in bombs_info]
    total_T = union_duration(all_ivs)
    merged = []
    for ivs in all_ivs: merged.extend(ivs)
    merged.sort(key=lambda x: x[0])
    merged_ivs = []
    for s, e in merged:
        if merged_ivs and s <= merged_ivs[-1][1] + 1e-6:
            merged_ivs[-1] = (merged_ivs[-1][0], max(merged_ivs[-1][1], e))
        else:
            merged_ivs.append((s, e))

    coverage = total_T / T_FLIGHT * 100
    return {'summary': {
        'Problem': '问题4: FY1+FY2+FY3 各1枚弹 → M1',
        'Total_T_eff(s)': round(total_T, 4),
        'M1_flight_time(s)': round(T_FLIGHT, 1),
        'Coverage(%)': round(coverage, 2),
    }, 'bombs': bombs_info, 'shielding': {'M1': {'duration': total_T, 'intervals': merged_ivs}}}

def save_to_result2(result):
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xlsx_path = os.path.join(proj_root, '附件', 'result2.xlsx')
    try:
        import openpyxl
        wb = openpyxl.load_workbook(xlsx_path)
        ws = wb['Sheet1']
        for i, bi in enumerate(result['bombs']):
            row = 2 + i
            for col, key in enumerate(['drone', 'theta', 'v', 'release_x', 'release_y',
                                        'release_z', 'A_x', 'A_y', 'A_z', 'T_eff'], 1):
                ws.cell(row=row, column=col, value=round(bi[key], 4) if isinstance(bi[key], float) else bi[key])
        wb.save(xlsx_path)
        print(f"  [OK] Saved to {xlsx_path}")
        return xlsx_path
    except Exception as e:
        print(f"  [WARN] xlsx write failed: {e}")
        from common import save_result_xlsx
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result2_out.xlsx')
        save_result_xlsx({'summary': result['summary'],
                          'bombs': [{k: bi[k] for k in ['drone', 't_release', 't_det', 'A_x', 'A_y', 'A_z', 'T_eff', 'v', 'theta']} for bi in result['bombs']],
                          'shielding': result['shielding']}, out_path)
        return out_path


# ====================================================================
# 主程序
# ====================================================================
if __name__ == "__main__":
    t_start = time.time()
    print("=" * 70)
    print("问题4 v2: FY1 + FY2 + FY3 各投放1枚烟幕干扰弹 → M1")
    print(f"M1飞行时间: {T_FLIGHT:.1f}s | 策略: 物理引导+BO+CMA-ES (common.py)")
    print("=" * 70)

    # ================================================================
    # Phase 0: 锚点搜索
    # ================================================================
    print(f"\n{'─'*70}")
    print("Phase 0: 锚点搜索 (独立子进程隔离, segfault不波及主进程)")
    print(f"{'─'*70}")

    print(f"  FY1: 问题二锚点 T_eff={FY1_OPT_TEFF:.4f}s "
          f"x=[{FY1_OPT_X[0]:.2f},{FY1_OPT_X[1]:.0f},{FY1_OPT_X[2]:.0f},{FY1_OPT_X[3]:.0f}]")

    # FY2: 独立子进程搜索
    print(f"\n  FY2 物理引导搜索 (独立进程, 12000点)...")
    t0 = time.time()
    fy2_x, fy2_T, fy2_feas, fy2_eff = run_anchor_search('FY2')
    print(f"  FY2 done ({time.time()-t0:.1f}s)")
    if fy2_x is None or fy2_T <= 0:
        print(f"  [FALLBACK] FY2 搜索失败, 使用默认锚点")
        fy2_x = (20.0, 9000.0, 50.0, 900.0, {'v': 100.0, 'theta': math.atan2(50-1400, 9000-12000)})
        fy2_T = 0.0

    gc.collect()

    # FY3: 独立子进程搜索
    print(f"\n  FY3 物理引导搜索 (独立进程, 8000点)...")
    t0 = time.time()
    fy3_x, fy3_T, fy3_feas, fy3_eff = run_anchor_search('FY3')
    print(f"  FY3 done ({time.time()-t0:.1f}s)")
    if fy3_x is None or fy3_T <= 0:
        print(f"  [FALLBACK] FY3 搜索失败, 使用默认锚点")
        fy3_x = (35.0, 2500.0, -500.0, 400.0, {'v': 100.0, 'theta': math.atan2(2500, -3500)})
        fy3_T = 0.0

    gc.collect()

    # Phase 0 汇总
    phase0 = [
        ('FY1', np.array(FY1_OPT_X), FY1_OPT_TEFF),
        ('FY2', np.array([fy2_x[0], fy2_x[1], fy2_x[2], fy2_x[3]]), fy2_T),
        ('FY3', np.array([fy3_x[0], fy3_x[1], fy3_x[2], fy3_x[3]]), fy3_T),
    ]
    print(f"\n  Phase 0 汇总:")
    for nm, xo, To in phase0:
        print(f"    {nm}: T_eff={To:.4f}s td={xo[0]:.2f}s A=({xo[1]:.0f},{xo[2]:.0f},{xo[3]:.0f})")
    indep_sum = sum(max(0, t) for _, _, t in phase0)
    print(f"    独立求和 (理论上界): {indep_sum:.4f}s")

    # ================================================================
    # Phase 1: BO单独优化 FY2, FY3
    # ================================================================
    print(f"\n{'─'*70}")
    print("Phase 1: BO单无人机优化")
    print(f"{'─'*70}")

    # FY2 BO
    if fy2_T > 0:
        c2 = (fy2_x[0], fy2_x[1], fy2_x[2], fy2_x[3])
        lb2 = np.array([max(0.5, c2[0]-10), max(500, c2[1]-5000),
                        max(-5500, c2[2]-2000), max(50, c2[3]-500)])
        ub2 = np.array([min(T_FLIGHT-5, c2[0]+10), min(17800, c2[1]+4000),
                        min(5000, c2[2]+2000), min(1390, c2[3]+300)])
        print(f"  FY2 BO: {30}+{100} eval")
        obj2 = lambda x: objective_single(x, FY2_0, 1400.0)
        known2 = [(np.array(c2), fy2_T)]
        fy2_bo_x, fy2_bo_T = bayesian_optimize(
            obj2, lb2, ub2, n_init=30, n_iter=100, seed=42,
            known_points=known2, verbose=True)
        gc.collect()
    else:
        fy2_bo_x, fy2_bo_T = np.array(fy2_x[:4]), fy2_T

    # FY3 BO
    if fy3_T > 0:
        c3 = (fy3_x[0], fy3_x[1], fy3_x[2], fy3_x[3])
        lb3 = np.array([max(0.5, c3[0]-15), max(500, c3[1]-4000),
                        max(-5500, c3[2]-2000), max(50, c3[3]-300)])
        ub3 = np.array([min(T_FLIGHT-5, c3[0]+15), min(17800, c3[1]+4000),
                        min(5000, c3[2]+2000), min(690, c3[3]+200)])
        print(f"  FY3 BO: {30}+{100} eval")
        obj3 = lambda x: objective_single(x, FY3_0, 700.0)
        known3 = [(np.array(c3), fy3_T)]
        fy3_bo_x, fy3_bo_T = bayesian_optimize(
            obj3, lb3, ub3, n_init=30, n_iter=100, seed=123,
            known_points=known3, verbose=True)
        gc.collect()
    else:
        fy3_bo_x, fy3_bo_T = np.array(fy3_x[:4]), fy3_T

    # Phase 1 汇总
    print(f"\n  Phase 1 汇总:")
    phase1 = [
        ('FY1', np.array(FY1_OPT_X), FY1_OPT_TEFF),
        ('FY2', fy2_bo_x, fy2_bo_T),
        ('FY3', fy3_bo_x, fy3_bo_T),
    ]
    for nm, xo, To in phase1:
        print(f"    {nm}: T_eff={To:.4f}s td={xo[0]:.3f}s A=({xo[1]:.0f},{xo[2]:.0f},{xo[3]:.0f})")

    # ================================================================
    # Phase 2: CMA-ES联合优化
    # ================================================================
    print(f"\n{'='*70}")
    print("Phase 2: CMA-ES联合优化 (12D)")
    print(f"{'='*70}")

    x0_joint = np.concatenate([p[1] for p in phase1])
    init_val = joint_T_eff(x0_joint)
    print(f"  初始解 (拼接并集): T_eff = {init_val:.4f}s")

    def make_bounds(cx, dz, tight=False):
        td_c, Ax_c, Ay_c, Az_c = float(cx[0]), float(cx[1]), float(cx[2]), float(cx[3])
        s = 0.1 if tight else 0.5
        return [
            (max(0.5, td_c*(1-s*0.5)), min(T_FLIGHT-2, td_c*(1+s*0.5))),
            (max(500.0, Ax_c - max(2000, abs(Ax_c)*s)), min(17800.0, Ax_c + max(2000, abs(Ax_c)*s))),
            (max(-5500.0, Ay_c - max(1500, abs(Ay_c)*s)), min(5000.0, Ay_c + max(1500, abs(Ay_c)*s))),
            (max(50.0, Az_c - 400), min(dz-10, Az_c + 200)),
        ]

    all_bounds = (make_bounds(phase1[0][1], 1800, tight=True) +
                  make_bounds(phase1[1][1], 1400) +
                  make_bounds(phase1[2][1], 700))

    lb_array = np.array([b[0] for b in all_bounds])
    ub_array = np.array([b[1] for b in all_bounds])
    x0_clipped = np.clip(x0_joint, lb_array, ub_array)

    import cma; gc.collect()
    sigma0 = 0.5
    print(f"  CMA-ES: sigma0={sigma0}, maxfevals=300")

    try:
        es = cma.CMAEvolutionStrategy(x0_clipped, sigma0, {
            'bounds': [lb_array.tolist(), ub_array.tolist()],
            'maxfevals': 300, 'seed': 42, 'verbose': -1,
            'CMA_diagonal': True, 'popsize': 6})
        best_f_cma = float('inf'); best_x_cma = x0_clipped.copy()
        gen_count = 0
        while not es.stop():
            solutions = es.ask()
            fitnesses = [cma_objective(x) for x in solutions]
            es.tell(solutions, fitnesses)
            idx = np.argmin(fitnesses)
            if fitnesses[idx] < best_f_cma:
                best_f_cma = fitnesses[idx]; best_x_cma = solutions[idx].copy()
            gen_count += 1
            gc.collect()
            if gen_count % 5 == 0 or gen_count == 1:
                print(f"  CMA-ES gen {gen_count:4d} | sigma={es.sigma:.4f} | "
                      f"best={-best_f_cma:.4f} | evals={es.result.evaluations}")
        print(f"  CMA-ES done: {es.result.evaluations} evals | best={-best_f_cma:.4f}")
        cma_teff = joint_T_eff(best_x_cma)
    except Exception as e:
        print(f"  [WARN] CMA-ES error: {e}, using Phase 1")
        best_x_cma = x0_joint; cma_teff = init_val

    # ================================================================
    # 最终结果
    # ================================================================
    if cma_teff > init_val:
        final_x, final_teff = best_x_cma, cma_teff
        print(f"\n  ✓ CMA-ES改进: {cma_teff:.4f}s > {init_val:.4f}s (+{cma_teff-init_val:.4f}s)")
    else:
        final_x, final_teff = x0_joint, init_val
        print(f"\n  → 使用Phase 1拼接: {init_val:.4f}s")

    result = build_result(final_x)

    print(f"\n  {'─'*80}")
    print(f"  {'Drone':>6s} {'t_rel':>8s} {'t_det':>8s} {'v':>7s} {'th':>7s} "
          f"{'RelX':>9s} {'RelY':>8s} {'RelZ':>7s} "
          f"{'Ax':>9s} {'Ay':>8s} {'Az':>7s} {'T_single':>9s}")
    print(f"  {'─'*80}")
    for bi in result['bombs']:
        print(f"  {bi['drone']:>6s} {bi['t_release']:8.3f} {bi['t_det']:8.3f} "
              f"{bi['v']:7.1f} {bi['theta']:7.1f} "
              f"{bi['release_x']:9.1f} {bi['release_y']:8.1f} {bi['release_z']:7.1f} "
              f"{bi['A_x']:9.1f} {bi['A_y']:8.1f} {bi['A_z']:7.1f} {bi['T_eff']:9.4f}")

    print(f"\n  遮蔽区间 (并集):")
    for mn, info in result['shielding'].items():
        iv_str = '; '.join([f"[{s:.3f}, {e:.3f}]" for s, e in info['intervals']])
        print(f"    {mn}: {info['duration']:.4f}s | {iv_str}")

    print(f"\n  对比分析:")
    print(f"    M1飞行时间: {T_FLIGHT:.1f}s")
    print(f"    FY1单弹 (问题二): {FY1_OPT_TEFF:.4f}s")
    print(f"    独立求和:          {indep_sum:.4f}s")
    print(f"    协同并集:          {final_teff:.4f}s")
    print(f"    覆盖率:            {result['summary']['Coverage(%)']:.2f}%")
    if FY1_OPT_TEFF > 0:
        print(f"    提升 (vs FY1单弹): {final_teff/FY1_OPT_TEFF:.2f}x")

    print(f"\n  保存到 result2.xlsx...")
    save_to_result2(result)

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"问题4 v2 完成! (总耗时 {elapsed:.1f}s)")
    print(f"协同遮蔽时长: {final_teff:.4f}s | 覆盖率: {result['summary']['Coverage(%)']:.2f}%")
    print(f"{'='*70}")

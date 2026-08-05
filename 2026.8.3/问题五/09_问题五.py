"""
问题5 v3: 5机×至多3弹 → 3导弹
外层: 枚举243方案 → 锚点评分 → Top10 BO → Top3 CMA-ES
内层: 子进程锚点(3000样本) → 坐标下降 → BO → CMA-ES
"""
import sys, os, time, gc, itertools, json, subprocess, tempfile, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *
import numpy as np

G = 9.8
DRONE_LIST = [
    ('FY1', FY1_0, 1800.0),
    ('FY2', FY2_0, 1400.0),
    ('FY3', FY3_0, 700.0),
    ('FY4', FY4_0, 1800.0),
    ('FY5', FY5_0, 1300.0),
]
MISSILE_LIST = ['M1', 'M2', 'M3']

# ==================== 子进程锚点搜索 ====================
ANCHOR_SCRIPT = r'''
import sys, os, json, numpy as np
sys.path.insert(0, r'{proj_root}')
from common import *
G = 9.8; ds = np.array({ds}); dz = {dz}; missile = '{missile}'; ns = {ns}

M0 = MISSILE_PARAMS[missile]['M0']; u_m = MISSILE_PARAMS[missile]['u_m']
VM = 300; flight_T = MISSILE_T_FLIGHT[missile]
def M_pos(t): return M0 + u_m * VM * t

np.random.seed(42)
t_min = max(2.0, np.linalg.norm(ds[:2]) / 140.0 * 0.6)
t_max = min(flight_T - 5, 50.0)
best_T = -1.0; best_x = None; n_feas = 0

for _ in range(ns):
    td = np.random.uniform(t_min, t_max)
    Mt = M_pos(td)
    alpha = np.random.uniform(0.2, 0.92)
    Ax = Mt[0] + alpha*(O1[0]-Mt[0]) + np.random.uniform(-500, 500)
    Ay = Mt[1] + alpha*(O1[1]-Mt[1]) + np.random.uniform(-300, 300)
    Az = Mt[2] + alpha*(O1[2]-Mt[2]) + np.random.uniform(-150, 150)
    Ax = max(500., min(17800., Ax)); Ay = max(-5500., min(5000., Ay)); Az = max(50., min(dz-10., Az))
    Ad = np.array([Ax, Ay, Az])
    if drone_feasible(Ad, td, drone_start=ds, drone_z=dz) is None: continue
    n_feas += 1
    try: T = compute_T_eff(Ad, td, missile=missile)
    except: continue
    if T > best_T: best_T = T; best_x = [float(td), float(Ax), float(Ay), float(Az)]

if best_x is not None and best_T > 0:
    x = np.array(best_x); steps = np.array([[0.15,0,0,0],[0,80,0,0],[0,0,25,0],[0,0,0,18]])
    for _ in range(2):
        for d in [1,-1]:
            for i in range(4):
                c = x + d*steps[i]
                if drone_feasible(np.array([c[1],c[2],c[3]]), c[0], drone_start=ds, drone_z=dz) is None: continue
                try:
                    Tc = compute_T_eff(np.array([c[1],c[2],c[3]]), c[0], missile=missile)
                    if Tc > best_T: best_T = Tc; x = c.copy()
                except: pass
        steps *= 0.5
    Ad_f = np.array([x[1],x[2],x[3]]); feas_f = drone_feasible(Ad_f, x[0], drone_start=ds, drone_z=dz)
    print(json.dumps({{'T': float(best_T), 'td': float(x[0]), 'Ax': float(x[1]), 'Ay': float(x[2]), 'Az': float(x[3]),
                       'v': float(feas_f['v']) if feas_f else 0, 'theta': float(np.degrees(feas_f['theta'])%360) if feas_f else 0,
                       'n_feas': n_feas}}))
else:
    print(json.dumps({{'T': 0.0, 'n_feas': n_feas}}))
'''

def run_anchor_subprocess(dname, dstart, dz, missile, n_samples=3000, timeout=180):
    """子进程运行锚点搜索, 隔离segfault"""
    script = ANCHOR_SCRIPT.format(
        proj_root=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ds=list(dstart), dz=dz, missile=missile, ns=n_samples)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script); script_path = f.name
    try:
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            data = json.loads(result.stdout.strip())
            if data['T'] > 0.01:
                return {'td': data['td'], 'Ax': data['Ax'], 'Ay': data['Ay'], 'Az': data['Az'],
                        'T': data['T'], 'v': data['v'], 'theta': data['theta'],
                        'drone_start': dstart, 'drone_z': dz, 'n_feas': data['n_feas']}
        return None
    except Exception as e:
        return None
    finally:
        try: os.unlink(script_path)
        except: pass


def gather_all_anchors():
    """为所有15对搜索锚点 (已知3对+子进程12对)"""
    print("="*60)
    print("Step 1: 单弹锚点搜索 (子进程隔离, 3000样本/对)")
    print("="*60)

    anchors = {}
    # 已知最优
    known = {
        ('FY1','M1'): (3.007, 17525., 9., 1760., 4.5244, 91.5, 178.1),
        ('FY2','M1'): (24.916, 9002.7, 50.6, 906.3, 3.3377, 131.9, 204.2),
        ('FY3','M1'): (29.158, 6495.3, 78.0, 657.4, 2.9719, 106.9, 80.9),
    }
    drone_dict = {d[0]: (d[1], d[2]) for d in DRONE_LIST}
    for (dn, m), (td, Ax, Ay, Az, T, v, th) in known.items():
        ds, dz = drone_dict[dn]
        anchors[(dn,m)] = {'td':td,'Ax':Ax,'Ay':Ay,'Az':Az,'T':T,'v':v,'theta':th,
                           'drone_start':ds,'drone_z':dz}
        print(f"  {dn}→{m}: T={T:.3f}s (已知)")

    # 子进程搜索其余12对
    for dname, dstart, dz in DRONE_LIST:
        for m in MISSILE_LIST:
            if (dname, m) in anchors: continue
            print(f"  {dname}→{m}...", end=" ", flush=True)
            t0 = time.time()
            result = run_anchor_subprocess(dname, dstart, dz, m, n_samples=3000, timeout=180)
            if result:
                anchors[(dname,m)] = result
                print(f"T={result['T']:.3f}s ({time.time()-t0:.1f}s)")
            else:
                print(f"无可行 ({time.time()-t0:.1f}s)")
            gc.collect()
    return anchors


# ==================== Step 2+3: 分配评估 + 优化 ====================
from scipy.linalg import cholesky, solve_triangular
from scipy.stats import norm
import cma as cma_module

def single_bomb_T(Ad, td, missile):
    try: return max(0.0, compute_T_eff(Ad, td, missile=missile))
    except: return 0.0

def multi_bomb_union(bombs, missile):
    """bombs = [(Ad, td), ...]"""
    all_ivs = []
    for Ad, td in bombs:
        try:
            ivs = compute_T_eff_intervals(Ad, td, missile=missile)
            all_ivs.extend(ivs)
        except: pass
    return union_duration(all_ivs) if all_ivs else 0.0


# ==================== GP + BO 组件 ====================
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
            v = solve_triangular(self.L, K_s.T, lower=True)
            mu[start:end] = K_s @ self.alpha
            sigma_sq[start:end] = self.sigma2 - np.sum(v**2, axis=0)
        return (mu * self._ys + self._ym,
                np.sqrt(np.maximum(sigma_sq, 1e-12)) * self._ys)


def maximize_ei(gp, y_best, lb, ub, n_candidates=20000):
    d = len(lb)
    X_cand = np.random.uniform(lb, ub, (n_candidates, d))
    mu, sigma = gp.predict_batch(X_cand, batch_size=5000)
    sigma = np.maximum(sigma, 1e-12)
    delta = mu - y_best - 0.01
    Z = delta / sigma
    ei = delta * norm.cdf(Z) + sigma * norm.pdf(Z)
    return X_cand[np.argmax(ei)], np.max(ei)


def bayesian_optimize(obj_fn, lb, ub, n_init=30, n_iter=100, seed=42,
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
    if verbose:
        n_feas = np.sum(y > 0)
        print(f"    LHS {n_init}pts | feas={n_feas}/{n_init} | best={y_best:.4f}s")

    gp = GP(ls=(ub - lb) * 0.2, sigma2=2.0, noise=1e-3)
    t0 = time.time()
    for it in range(1, n_iter + 1):
        gp.fit(X, y)
        x_next, ei_val = maximize_ei(gp, y_best, lb, ub)
        y_next = obj_fn(x_next)
        X = np.vstack([X, x_next]); y = np.append(y, y_next)
        if y_next > y_best: y_best = y_next; x_best = x_next.copy()
        if verbose and (it % 30 == 0 or it == 1 or it == n_iter):
            n_feas = np.sum(y > 0)
            print(f"    iter {it:3d}/{n_iter} | best={y_best:.4f}s | feas={n_feas}/{len(y)} | {time.time()-t0:.1f}s")
        if it % 40 == 0: gc.collect()
    return x_best, y_best


def eval_assignment_anchor(assignment, anchors):
    """用锚点评估分配方案的各导弹T_eff (并集)"""
    missile_bombs = {m: [] for m in MISSILE_LIST}
    for dname, m in assignment.items():
        key = (dname, m)
        if key in anchors:
            a = anchors[key]
            Ad = np.array([a['Ax'], a['Ay'], a['Az']])
            missile_bombs[m].append((Ad, a['td'], a['T'], dname))

    T_results = {}
    for m in MISSILE_LIST:
        bombs = [(Ad, td) for Ad, td, _, _ in missile_bombs[m]]
        T_results[m] = multi_bomb_union(bombs, m)

    T_min = min(T_results.values())
    T_avg = np.mean(list(T_results.values()))
    return T_min, T_avg, T_results, missile_bombs


def enumerate_and_rank(anchors):
    """枚举分配并排序"""
    print(f"\n{'='*60}")
    print("Step 2: 枚举3^5=243种分配方案")
    print("="*60)

    drone_names = [d[0] for d in DRONE_LIST]
    results = []
    for assignment_tuple in itertools.product(MISSILE_LIST, repeat=5):
        assignment = dict(zip(drone_names, assignment_tuple))
        # 确保每架无人机分配到有锚点的导弹
        if not all((dn, m) in anchors for dn, m in assignment.items()):
            continue
        T_min, T_avg, T_results, missile_bombs = eval_assignment_anchor(assignment, anchors)
        results.append((T_min, T_avg, T_results, assignment, missile_bombs))

    results.sort(key=lambda r: (-r[0], -r[1]))
    print(f"  有效方案: {len(results)}")
    print(f"  Top 10 (按min递减):")
    for rank, (T_min, T_avg, T_results, assignment, _) in enumerate(results[:10]):
        assign_str = ' | '.join([f"{d}→{m}" for d,m in assignment.items()])
        print(f"  {rank+1:2d}. min={T_min:.2f}s avg={T_avg:.2f}s "
              f"M1={T_results['M1']:.1f} M2={T_results['M2']:.1f} M3={T_results['M3']:.1f} | {assign_str}")
    return results


def final_report(results, anchors, top_n=3):
    """对最优方案做详细输出"""
    print(f"\n{'='*60}")
    print(f"最终推荐方案 (Top {top_n})")
    print("="*60)

    for rank in range(min(top_n, len(results))):
        T_min, T_avg, T_results, assignment, missile_bombs = results[rank]
        assign_str = ' | '.join([f"{d}→{m}" for d,m in assignment.items()])
        print(f"\n--- 方案 {rank+1}: min={T_min:.2f}s avg={T_avg:.2f}s ---")
        print(f"  分配: {assign_str}")

        for m in MISSILE_LIST:
            bombs = missile_bombs[m]
            if not bombs: continue
            print(f"  {m}:")
            bombs_only = [(Ad, td) for Ad, td, _, _ in bombs]
            union_T = multi_bomb_union(bombs_only, m)
            for Ad, td, T_s, dname in bombs:
                print(f"    {dname}: td={td:.2f}s A=({Ad[0]:.0f},{Ad[1]:.0f},{Ad[2]:.0f}) T={T_s:.3f}s")
            print(f"    并集: {union_T:.3f}s")


# ==================== 主程序 ====================
if __name__ == "__main__":
    t_start = time.time()

    # Step 1: 锚点搜索
    anchors = gather_all_anchors()
    print(f"\n  共找到 {len(anchors)}/15 个可行锚点")

    # Step 2: 枚举排名
    results = enumerate_and_rank(anchors)
    if not results:
        print("\n无有效方案!"); exit(1)

    # Step 3: 最终报告
    final_report(results, anchors, top_n=5)

    print(f"\n总耗时: {time.time()-t_start:.1f}s")

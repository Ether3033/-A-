"""
问题四辅助脚本: 物理引导锚点搜索 (独立进程运行)
用法: python _run_search.py <drone_name> <output_json_path>
  drone_name: FY2 或 FY3
"""
import sys, os, json, time, gc, math, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *
import numpy as np

T_FLIGHT = MISSILE_T_FLIGHT['M1']

def physics_guided_search(drone_start, drone_z, n_samples=10000, seed=42):
    random.seed(seed); np.random.seed(seed)
    dr_y = float(drone_start[1])
    best_T = -1.0; best_x = None; n_feas = 0; n_eff = 0

    t_min = max(5.0, abs(dr_y) / V_MAX * 1.2)
    t_max = min(T_FLIGHT - 10, 55.0)

    for k in range(n_samples):
        td = random.uniform(t_min, t_max)
        Mt = M_fn(td, 'M1')
        alpha = random.uniform(0.3, 0.95)
        Ax = Mt[0] + alpha * (O1[0] - Mt[0]) + random.uniform(-300, 300)
        Ay = Mt[1] + alpha * (O1[1] - Mt[1]) + random.uniform(-150, 150)
        Az = Mt[2] + alpha * (O1[2] - Mt[2]) + random.uniform(-100, 100)
        Ax = max(500.0, min(17800.0, Ax))
        Ay = max(-5500.0, min(5000.0, Ay))
        Az = max(50.0, min(drone_z - 10.0, Az))

        Ad = np.array([Ax, Ay, Az])
        feas = drone_feasible(Ad, td, drone_start=drone_start, drone_z=drone_z)
        if feas is None: continue
        n_feas += 1

        try:
            T = compute_T_eff(Ad, td, missile='M1')
        except Exception:
            continue
        if T > 0: n_eff += 1
        if T > best_T:
            best_T = T
            best_x = {
                'td': td, 'Ax': Ax, 'Ay': Ay, 'Az': Az,
                'v': feas['v'], 'theta': math.degrees(feas['theta']) % 360,
                't_release': feas['t_release'], 't_fall': feas['t_fall']
            }

        if (k + 1) % 500 == 0:
            gc.collect()
            if best_x:
                with open(sys.argv[2] + '.tmp', 'w') as f:
                    json.dump({'best_T': best_T, 'best_x': best_x,
                               'n_feas': n_feas, 'n_eff': n_eff}, f)

    return best_x, best_T, n_feas, n_eff


def grid_search_fy3(drone_start, drone_z):
    """FY3确定性网格搜索 — 极少量评估,避免numpy segfault"""
    # FY3起点(6000,-3000,700), 导弹路径Y≈0
    best_T = -1.0; best_x = None; n_feas = 0; n_eff = 0

    # td: 10-50s, 每5s取一个
    td_vals = np.linspace(10, 50, 9)
    # alpha: 沿M(td)→O1连线的位置
    alpha_vals = [0.4, 0.55, 0.7, 0.85]
    # 小扰动
    eps_vals = [(0, 0, 0), (100, 50, 30), (-100, -50, -30)]

    for td in td_vals:
        Mt = M_fn(td, 'M1')
        for alpha in alpha_vals:
            base_Ax = Mt[0] + alpha * (O1[0] - Mt[0])
            base_Ay = Mt[1] + alpha * (O1[1] - Mt[1])
            base_Az = Mt[2] + alpha * (O1[2] - Mt[2])
            for (ex, ey, ez) in eps_vals:
                Ax = max(500.0, min(17800.0, base_Ax + ex))
                Ay = max(-5500.0, min(5000.0, base_Ay + ey))
                Az = max(50.0, min(drone_z - 10.0, base_Az + ez))
                Ad = np.array([Ax, Ay, Az])
                feas = drone_feasible(Ad, td, drone_start=drone_start, drone_z=drone_z)
                if feas is None: continue
                n_feas += 1
                try:
                    T = compute_T_eff(Ad, td, missile='M1')
                except Exception:
                    continue
                if T > 0: n_eff += 1
                if T > best_T:
                    best_T = T
                    best_x = {
                        'td': float(td), 'Ax': float(Ax), 'Ay': float(Ay), 'Az': float(Az),
                        'v': feas['v'], 'theta': math.degrees(feas['theta']) % 360,
                        't_release': feas['t_release'], 't_fall': feas['t_fall']
                    }
            gc.collect()  # 每次外层循环后回收
    return best_x, best_T, n_feas, n_eff


if __name__ == "__main__":
    drone_name = sys.argv[1]
    out_path = sys.argv[2]

    configs = {'FY2': (FY2_0, 1400.0), 'FY3': (FY3_0, 700.0)}
    drone_start, drone_z = configs[drone_name]

    if drone_name == 'FY3':
        # FY3用确定性网格(9*4*3=108点),避免随机搜索触发segfault
        best_x, best_T, n_feas, n_eff = grid_search_fy3(drone_start, drone_z)
    else:
        n_samples = 12000
        seed = 42
        best_x, best_T, n_feas, n_eff = physics_guided_search(drone_start, drone_z, n_samples, seed)

    output = {'drone': drone_name, 'best_T': best_T,
              'best_x': best_x, 'n_feas': n_feas, 'n_eff': n_eff,
              'drone_start': list(drone_start), 'drone_z': drone_z}
    with open(out_path, 'w') as f:
        json.dump(output, f)
    print(f"OK {drone_name}: T_eff={best_T:.4f}s feas={n_feas} eff={n_eff}")

"""
问题三辅助脚本: 轻量网格搜索3弹最优配置 (独立进程运行, 防segfault)
用法: python _run_search_p3.py <output_json_path>
"""
import sys, os, json, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *
import numpy as np

G = 9.8

# 粗网格: v(3) × θ(3) × td1(2) × delay1(2) × td2(3) × delay2(2) × td3(3) × delay3(2)
# = 3*3*2*2*3*2*3*2 = 1296 配置
v_vals = [85, 95, 105]
th_vals = [176, 179, 182]
td1_vals = [2.8, 3.2]
dl1_vals = [2.5, 2.9]
td2_vals = [4.5, 5.5, 6.5]
dl2_vals = [2.5, 3.2]
td3_vals = [7.0, 8.5, 10.0]
dl3_vals = [2.5, 3.5]

best_union = 0.0; best_config = None
n_tested = 0; n_feas = 0

for v in v_vals:
    for th_deg in th_vals:
        th = np.radians(th_deg)
        for td1 in td1_vals:
            for tdl1 in dl1_vals:
                tr1 = td1 - tdl1
                if tr1 < 0: continue
                Az1 = 1800 - 0.5*G*tdl1**2
                Ax1 = FY1_0[0] + v*np.cos(th)*td1
                Ay1 = FY1_0[1] + v*np.sin(th)*td1
                Ad1 = np.array([Ax1, Ay1, Az1])
                if drone_feasible(Ad1, td1) is None: continue

                for td2 in td2_vals:
                    if td2 <= td1 + 0.3: continue
                    for tdl2 in dl2_vals:
                        tr2 = td2 - tdl2
                        if tr2 < tr1 + 1.0: continue
                        Az2 = 1800 - 0.5*G*tdl2**2
                        if Az2 < 50: continue
                        Ax2 = FY1_0[0] + v*np.cos(th)*td2
                        Ay2 = FY1_0[1] + v*np.sin(th)*td2
                        Ad2 = np.array([Ax2, Ay2, Az2])
                        if drone_feasible(Ad2, td2) is None: continue

                        for td3 in td3_vals:
                            if td3 <= td2 + 0.3: continue
                            for tdl3 in dl3_vals:
                                tr3 = td3 - tdl3
                                if tr3 < tr2 + 1.0: continue
                                Az3 = 1800 - 0.5*G*tdl3**2
                                if Az3 < 50: continue
                                Ax3 = FY1_0[0] + v*np.cos(th)*td3
                                Ay3 = FY1_0[1] + v*np.sin(th)*td3
                                Ad3 = np.array([Ax3, Ay3, Az3])
                                if drone_feasible(Ad3, td3) is None: continue

                                n_tested += 1
                                try:
                                    ivs1 = compute_T_eff_intervals(Ad1, td1, 'M1')
                                    ivs2 = compute_T_eff_intervals(Ad2, td2, 'M1')
                                    ivs3 = compute_T_eff_intervals(Ad3, td3, 'M1')
                                    total = union_duration([ivs1, ivs2, ivs3])
                                    if total > 0: n_feas += 1
                                    if total > best_union:
                                        best_union = total
                                        best_config = {
                                            'v': v, 'theta_deg': th_deg,
                                            'bombs': [
                                                {'td': td1, 'tdl': tdl1, 'Ad': [float(Ad1[0]), float(Ad1[1]), float(Ad1[2])],
                                                 'ivs': [(float(s), float(e)) for s,e in ivs1],
                                                 'T_single': float(sum(e-s for s,e in ivs1))},
                                                {'td': td2, 'tdl': tdl2, 'Ad': [float(Ad2[0]), float(Ad2[1]), float(Ad2[2])],
                                                 'ivs': [(float(s), float(e)) for s,e in ivs2],
                                                 'T_single': float(sum(e-s for s,e in ivs2))},
                                                {'td': td3, 'tdl': tdl3, 'Ad': [float(Ad3[0]), float(Ad3[1]), float(Ad3[2])],
                                                 'ivs': [(float(s), float(e)) for s,e in ivs3],
                                                 'T_single': float(sum(e-s for s,e in ivs3))},
                                            ],
                                            'union': float(total),
                                        }
                                except Exception:
                                    pass

                                if n_tested % 50 == 0:
                                    gc.collect()

output = {'best_union': best_union, 'best_config': best_config,
          'n_tested': n_tested, 'n_feas': n_feas}

# Also try the P2 anchor as baseline
try:
    Ad_p2 = np.array([17525., 9., 1760.])
    td_p2 = 3.007
    ivs_p2 = compute_T_eff_intervals(Ad_p2, td_p2, 'M1')
    T_p2 = sum(e-s for s,e in ivs_p2)
    output['p2_anchor'] = {'T_eff': float(T_p2), 'ivs': [(float(s), float(e)) for s,e in ivs_p2]}
except Exception:
    output['p2_anchor'] = None

with open(sys.argv[1], 'w') as f:
    json.dump(output, f)
print(f"OK P3: union={best_union:.4f}s tested={n_tested} feas={n_feas}")

"""Test anchor search for one drone-missile pair"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

if len(sys.argv) < 5:
    print("Usage: _test_anchor.py <drone_name> <missile> <n_samples> <out_json>")
    exit(1)

dname = sys.argv[1]; missile = sys.argv[2]
ns = int(sys.argv[3]); out_path = sys.argv[4]

DRONE_MAP = {'FY1': (FY1_0, 1800.), 'FY2': (FY2_0, 1400.), 'FY3': (FY3_0, 700.),
             'FY4': (FY4_0, 1800.), 'FY5': (FY5_0, 1300.)}
ds, dz = DRONE_MAP[dname]

M0 = MISSILE_PARAMS[missile]['M0']; u_m = MISSILE_PARAMS[missile]['u_m']
flight_T = MISSILE_T_FLIGHT[missile]

def M_pos(t): return M0 + u_m * VM * t

np.random.seed(42)
t_min = max(2.0, abs(ds[1]) / V_MAX * 0.6)
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
    try:
        T = compute_T_eff(Ad, td, missile=missile)
        if T > best_T: best_T = T; best_x = [float(td), float(Ax), float(Ay), float(Az)]
    except: pass

result = {'T': 0.0, 'n_feas': n_feas}
if best_x is not None and best_T > 0.01:
    x = np.array(best_x); s = np.array([[0.15,0,0,0],[0,80,0,0],[0,0,25,0],[0,0,0,18]])
    for _ in range(2):
        for d in [1,-1]:
            for i in range(4):
                c = x + d*s[i]
                if drone_feasible(np.array([c[1],c[2],c[3]]), c[0], drone_start=ds, drone_z=dz) is None: continue
                try:
                    Tc = compute_T_eff(np.array([c[1],c[2],c[3]]), c[0], missile=missile)
                    if Tc > best_T: best_T = Tc; x = c.copy()
                except: pass
        s *= 0.5
    Ad_f = np.array([x[1],x[2],x[3]]); feas_f = drone_feasible(Ad_f, x[0], drone_start=ds, drone_z=dz)
    result = {'T': float(best_T), 'td': float(x[0]), 'Ax': float(x[1]), 'Ay': float(x[2]), 'Az': float(x[3]),
              'v': float(feas_f['v']) if feas_f else 0,
              'theta': float(np.degrees(feas_f['theta'])%360) if feas_f else 0,
              'n_feas': n_feas}

with open(out_path, 'w') as f:
    json.dump(result, f)

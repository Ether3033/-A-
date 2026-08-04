"""
问题1 - 文件2: 完整计算 (求根法)
=================================
FY1投放1枚烟幕弹干扰M1:
  FY1以120m/s朝向假目标, 1.5s投弹, 3.6s后起爆
  计算有效遮蔽时长

方法:
  1. 解析: M(t), A(t) 关于t的显式函数
  2. 圆柱→单位球面投影: 215个边界采样点
  3. 二分求根: θ_max(t) = α(t)  → C2起始时刻
  4. 二次方程: |MA|=10, |MO1|=|AO1| → C1起止, 几何失效
  5. 取交后取并 → 精确遮蔽时长
"""

import numpy as np
from scipy.optimize import bisect

# ======================== 场景参数 ========================
M1_0    = np.array([20000., 0., 2000.])    # M1初始位置
FY1_0   = np.array([17800., 0., 1800.])    # FY1初始位置
O_DECOY = np.array([0., 0., 0.])           # 假目标(原点,导弹瞄准点)
O1      = np.array([0., 200., 0.])         # 真目标底面圆心
Rc, Hc  = 7., 10.                          # 圆柱半径,高度
R       = 10.                              # 烟幕球半径
V_SINK  = 3.                               # 烟幕下沉速度 m/s
T_LIFE  = 20.                              # 烟幕有效时间 s
VM      = 300.                             # 导弹速度 m/s
G       = 9.8                              # 重力加速度

# 问题1给定参数
V_DRONE  = 120.        # 无人机速度
T_REL    = 1.5          # 投放时刻
T_DELAY  = 3.6          # 投放→起爆延时
T_DET    = T_REL + T_DELAY  # 起爆时刻 = 5.1s

# ======================== 运动函数 ========================
u_m = (O_DECOY - M1_0) / np.linalg.norm(O_DECOY - M1_0)

def M(t):
    """导弹M1在t时刻的位置"""
    return M1_0 + u_m * VM * t

def A(t):
    """烟幕球心在t时刻的位置 (t≥T_DET时有效, 匀速下沉)"""
    age = t - T_DET
    if age < 0 or age > T_LIFE:
        return None
    return A_det - np.array([0., 0., V_SINK * age])

# 投放点和起爆点 (正向计算)
u_f = np.array([-1., 0., 0.])  # FY1朝向假目标(正西)
R_pt = FY1_0 + V_DRONE * u_f * T_REL
A_det = R_pt + V_DRONE * u_f * T_DELAY + np.array([0., 0., -0.5*G*T_DELAY**2])

# ======================== 圆柱投影采样 (215点) ========================
def sample_boundary(M_pos):
    """圆柱可见边界 → 215个单位方向向量"""
    Mx, My, Mz = M_pos
    Ox, Oy, Oz = O1
    d_xy = np.array([Mx-Ox, My-Oy]); dist_xy = np.linalg.norm(d_xy)
    u_xy = d_xy / dist_xy
    psi = np.arcsin(np.clip(Rc/dist_xy, 0, 1)); cp, sp = np.cos(psi), np.sin(psi)
    tR = np.array([u_xy[0]*cp-u_xy[1]*sp, u_xy[0]*sp+u_xy[1]*cp])
    tL = np.array([u_xy[0]*cp+u_xy[1]*sp, -u_xy[0]*sp+u_xy[1]*cp])
    TR = np.array([Ox+Rc*tR[0], Oy+Rc*tR[1]])
    TL = np.array([Ox+Rc*tL[0], Oy+Rc*tL[1]])

    pts = []; ns, ne = 20, 48
    for i in range(ns+1):
        z = Oz + Hc*i/ns
        pts.append(np.array([TL[0],TL[1],z]))
        pts.append(np.array([TR[0],TR[1],z]))
    for i in range(ne+1):
        a = 2*np.pi*i/ne
        pts.append(np.array([Ox+Rc*np.cos(a), Oy+Rc*np.sin(a), Oz+Hc]))
    aM = np.arctan2(My-Oy, Mx-Ox)
    for i in range(ne//2+1):
        a = aM + np.pi/2 + np.pi*i/(ne//2)
        pts.append(np.array([Ox+Rc*np.cos(a), Oy+Rc*np.sin(a), Oz]))
    for i in range(1, 10):
        f = i/11.; Tm = TL + f*(TR-TL)
        v = Tm - np.array([Ox,Oy]); v = v*(Rc/np.linalg.norm(v))
        Tm = np.array([Ox,Oy])+v
        for j in range(ns//2+1):
            z = Oz + Hc*j/(ns//2)
            pts.append(np.array([Tm[0], Tm[1], z]))

    vecs = []
    for p in pts:
        v = p - M_pos; d = np.linalg.norm(v)
        if d > 1e-10: vecs.append(v/d)
    return np.array(vecs)

# ======================== 不等式组 ========================
def theta_max_and_alpha(t):
    """返回 θ_max(t), α(t)"""
    Mt = M(t); At = A(t)
    if At is None: return None, None
    MA = np.linalg.norm(Mt - At)
    if MA <= R: return 0.0, np.pi/2
    u_A = (At - Mt) / MA
    alpha = np.arcsin(R / MA)
    proj = sample_boundary(Mt)
    dots = np.clip(proj @ u_A, -1, 1)
    return np.max(np.arccos(dots)), alpha

def shielded(t):
    """判定t时刻是否遮蔽"""
    Mt = M(t); At = A(t)
    if At is None: return False

    MA = np.linalg.norm(Mt - At)
    # 条件1: 导弹在烟幕球内
    if MA <= R: return True

    # 条件2: 烟幕遮挡视线
    MO1 = np.linalg.norm(Mt - O1)
    AO1 = np.linalg.norm(At - O1)
    if MO1 <= AO1: return False  # 烟幕在目标后方

    th, alpha = theta_max_and_alpha(t)
    return th <= alpha

# ======================== 二分求根: θ_max(t) = α(t) ========================
def f_C2(t):
    """f(t) = α(t) - θ_max(t)"""
    th, alpha = theta_max_and_alpha(t)
    if th is None: return -1
    return alpha - th

# ======================== 解析求根: |MA|² = R² ========================
dx0 = M1_0[0] - A_det[0]
dy0 = M1_0[1] - A_det[1]
dz0 = M1_0[2] - A_det[2] - V_SINK * T_DET
vx = u_m[0]*VM; vy = u_m[1]*VM; vz = u_m[2]*VM + V_SINK
a2_ma = vx**2+vy**2+vz**2
a1_ma = 2*(dx0*vx+dy0*vy+dz0*vz)
a0_ma = dx0**2+dy0**2+dz0**2 - R**2
disc_ma = a1_ma**2 - 4*a2_ma*a0_ma
t_c1_enter = (-a1_ma - np.sqrt(disc_ma)) / (2*a2_ma)
t_c1_exit  = (-a1_ma + np.sqrt(disc_ma)) / (2*a2_ma)

# ======================== 解析求根: |MO1|² = |AO1|² ========================
v_mx, v_my, v_mz = u_m[0]*VM, u_m[1]*VM, u_m[2]*VM
Az_c = A_det[2] + V_SINK * T_DET
a2_g = (v_mx**2+v_my**2+v_mz**2) - V_SINK**2
a1_g = 2*(M1_0[0]*v_mx + (M1_0[1]-200)*v_my + M1_0[2]*v_mz + Az_c*V_SINK)
a0_g = (M1_0[0]**2+(M1_0[1]-200)**2+M1_0[2]**2) - (A_det[0]**2+(A_det[1]-200)**2+Az_c**2)
disc_g = a1_g**2 - 4*a2_g*a0_g
t_geom_roots = [(-a1_g - np.sqrt(disc_g))/(2*a2_g), (-a1_g + np.sqrt(disc_g))/(2*a2_g)]
t_geom = next((tg for tg in t_geom_roots if T_DET < tg < T_DET+T_LIFE), T_DET+T_LIFE)

# ======================== 主计算 ========================
if __name__ == "__main__":
    print("="*70)
    print("问题1: FY1投放1枚烟幕弹干扰M1  求根法精确计算")
    print("="*70)

    # 投放参数
    print(f"\n[给定参数]")
    print(f"  FY1速度={V_DRONE}m/s 方向=朝向假目标(正西)")
    print(f"  投放时刻={T_REL}s  投放点=({R_pt[0]:.0f},{R_pt[1]:.0f},{R_pt[2]:.0f})")
    print(f"  起爆延时={T_DELAY}s  起爆时刻={T_DET}s")
    print(f"  起爆点=({A_det[0]:.0f},{A_det[1]:.0f},{A_det[2]:.1f})")

    # 二分求C2起始
    print(f"\n[二分求根] θ_max(t) = α(t)")
    print(f"  扫描: f(5.1)={f_C2(5.1):.6f}  f(8.5)={f_C2(8.5):.6f}")
    t_c2_start = bisect(f_C2, 5.1, 8.5, xtol=1e-8, maxiter=100)
    th_s, al_s = theta_max_and_alpha(t_c2_start)
    print(f"  → t={t_c2_start:.6f}s  θ_max={np.degrees(th_s):.6f}°  α={np.degrees(al_s):.6f}°")

    # C1起止
    print(f"\n[解析求根] |MA|² = 100")
    print(f"  二次方程: {a2_ma:.2f}t² + {a1_ma:.2f}t + {a0_ma:.2f} = 0")
    print(f"  → t_enter={t_c1_enter:.6f}s  t_exit={t_c1_exit:.6f}s")

    # 几何失效
    print(f"\n[解析求根] |MO1|² = |AO1|²")
    print(f"  → t_geom={t_geom:.6f}s")

    # 遮蔽时长
    t_end = min(t_c1_exit, t_geom, T_DET+T_LIFE)
    T_C2 = t_c1_enter - t_c2_start
    T_C1 = t_end - t_c1_enter
    T_total = T_C2 + T_C1

    print(f"\n{'='*70}")
    print(f"结果")
    print(f"{'='*70}")
    print(f"  时间线:")
    print(f"    t={T_DET:.1f}s              起爆")
    print(f"    t={t_c2_start:.4f}s          C2开始 (θ_max≤α)")
    print(f"    t={t_c1_enter:.4f}s          C1开始 (M入球)")
    print(f"    t={t_geom:.4f}s              几何失效")
    print(f"    t={t_c1_exit:.4f}s           C1结束 (M出球)")
    print(f"")
    print(f"  C2遮蔽: {T_C2:.4f}s")
    print(f"  C1遮蔽: {T_C1:.4f}s")
    print(f"  ═══════════════════")
    print(f"  有效遮蔽时长 T_eff = {T_total:.4f}s")
    print(f"  占导弹飞行的 {T_total/67*100:.1f}%")

    # 验证
    print(f"\n[验证] 切换点处的状态:")
    for t_tag, desc in [(t_c2_start-0.001,"C2开始前"), (t_c2_start,"C2开始"),
                         (t_c1_enter-0.001,"入球前"), (t_c1_enter,"入球"),
                         (t_geom-0.001,"失效前"), (t_geom+0.001,"失效后")]:
        s = shielded(t_tag)
        Mt = M(t_tag); At = A(t_tag)
        MA_t = np.linalg.norm(Mt-At) if At is not None else 0
        print(f"    {desc:12s} t={t_tag:.4f}s  shielded={s}  |MA|={MA_t:.1f}m")

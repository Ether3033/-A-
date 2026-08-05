"""
问题A 空间场景示意图
====================
3枚导弹M1/M2/M3 + 5架无人机FY1-FY5 + 真假目标

导弹: 300m/s, 直指假目标(原点)
无人机: 各自位置标注
真目标: 圆柱 (0,200,0), r=7m, h=10m
假目标: 原点 (0,0,0)
"""

import numpy as np
import plotly.graph_objects as go

# ======================== 数据 ========================
# 导弹初始位置 (3枚)
missiles = {
    'M1': np.array([20000., 0., 2000.]),
    'M2': np.array([19000., 600., 2100.]),
    'M3': np.array([18000., -600., 1900.]),
}

# 无人机初始位置 (5架)
drones = {
    'FY1': np.array([17800., 0., 1800.]),
    'FY2': np.array([12000., 1400., 1400.]),
    'FY3': np.array([6000., -3000., 700.]),
    'FY4': np.array([11000., 2000., 1800.]),
    'FY5': np.array([13000., -2000., 1300.]),
}

# 目标
O_DECOY = np.array([0., 0., 0.])      # 假目标(原点, 导弹瞄准点)
O_REAL  = np.array([0., 200., 0.])     # 真目标底面圆心
Rc, Hc = 7., 10.                       # 圆柱半径, 高度
VM = 300.                              # 导弹速度 m/s

# ======================== 颜色方案 ========================
C_M = {'M1': '#e74c3c', 'M2': '#e67e22', 'M3': '#c0392b'}   # 红/橙/深红
C_D = {'FY1': '#2980b9', 'FY2': '#27ae60', 'FY3': '#8e44ad',
       'FY4': '#16a085', 'FY5': '#d35400'}                    # 蓝/绿/紫/青/橙

# ======================== 创建图形 ========================
fig = go.Figure()

# ── 图例占位 ──
fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='lines',
    line=dict(color='red', width=3, dash='dash'), name='导弹飞行方向 (→假目标)'))
fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers',
    marker=dict(size=6, color='cyan', symbol='circle', line=dict(width=1, color='black')),
    name='无人机 (FY1-FY5)'))
fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers',
    marker=dict(size=7, color='orange', symbol='circle', line=dict(width=1, color='darkorange')),
    name='假目标 (导弹瞄准点)'))
fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers',
    marker=dict(size=6, color='limegreen', symbol='circle', line=dict(width=1, color='darkgreen')),
    name='真目标 (圆柱)'))

# ======================== 1. 导弹 ========================
for name, pos in missiles.items():
    # 导弹位置
    fig.add_trace(go.Scatter3d(
        x=[pos[0]], y=[pos[1]], z=[pos[2]],
        mode='markers+text',
        marker=dict(size=7, color=C_M[name], symbol='circle',
                     line=dict(width=1.5, color='black')),
        text=[f'<b>{name}</b>'],
        textposition='top center',
        textfont=dict(size=17, color=C_M[name]),
        showlegend=False, hoverinfo='skip'
    ))

    # 飞行方向箭头 (指向假目标)
    u_dir = (O_DECOY - pos) / np.linalg.norm(O_DECOY - pos)
    arrow_len = 2500  # 箭头长度
    end = pos + u_dir * arrow_len
    # 箭头主体
    fig.add_trace(go.Scatter3d(
        x=[pos[0], end[0]], y=[pos[1], end[1]], z=[pos[2], end[2]],
        mode='lines',
        line=dict(color=C_M[name], width=2, dash='dash'),
        showlegend=False, hoverinfo='skip'
    ))
    # 箭头尖端 (锥形示意: 用大号标记)
    fig.add_trace(go.Scatter3d(
        x=[end[0]], y=[end[1]], z=[end[2]],
        mode='markers',
        marker=dict(size=5, color=C_M[name], symbol='diamond-open',
                     line=dict(width=1, color=C_M[name])),
        showlegend=False, hoverinfo='skip'
    ))

    # 导弹弹道延长线 (虚线延长至更远处, 展现俯冲轨迹)
    far = pos + u_dir * 6000
    fig.add_trace(go.Scatter3d(
        x=[pos[0], far[0]], y=[pos[1], far[1]], z=[pos[2], far[2]],
        mode='lines',
        line=dict(color=C_M[name], width=1.5, dash='dot'),
        opacity=0.5, showlegend=False, hoverinfo='skip'
    ))

# ── 导弹速度标签 ──
# 用 M1 标注速度
mid_m1 = (missiles['M1'] + O_DECOY) / 2 + np.array([0, 800, 500])
fig.add_trace(go.Scatter3d(
    x=[mid_m1[0]], y=[mid_m1[1]], z=[mid_m1[2]],
    mode='text',
    text=['<b>v<sub>M</sub> = 300 m/s<br>直指假目标</b>'],
    textfont=dict(size=16, color='darkred'),
    showlegend=False, hoverinfo='skip'
))

# ======================== 2. 无人机 ========================
for name, pos in drones.items():
    # 无人机位置
    fig.add_trace(go.Scatter3d(
        x=[pos[0]], y=[pos[1]], z=[pos[2]],
        mode='markers+text',
        marker=dict(size=6, color=C_D[name], symbol='circle',
                     line=dict(width=1.5, color='black')),
        text=[f'<b>{name}</b>'],
        textposition='top center',
        textfont=dict(size=16, color=C_D[name]),
        showlegend=False, hoverinfo='skip'
    ))

    # 高度参考线 (垂直虚线到地面)
    fig.add_trace(go.Scatter3d(
        x=[pos[0], pos[0]], y=[pos[1], pos[1]], z=[0, pos[2]],
        mode='lines',
        line=dict(color=C_D[name], width=1, dash='dot'),
        opacity=0.3, showlegend=False, hoverinfo='skip'
    ))

# ── 无人机参数标注 ──
fig.add_trace(go.Scatter3d(
    x=[14000], y=[3500], z=[2200],
    mode='text',
    text=['<b>FY1–FY5: 无人机</b><br>v ∈ [70, 140] m/s<br>高度: 700–1800 m<br>各携带烟幕弹'],
    textfont=dict(size=15, color='darkblue'),
    showlegend=False, hoverinfo='skip'
))

# ======================== 3. 真假目标 ========================
# 假目标 (原点) — 导弹瞄准点
fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0],
    mode='markers+text',
    marker=dict(size=7, color='orange', symbol='circle',
                 line=dict(width=3, color='darkorange')),
    text=['<b>假目标 (原点)</b>'],
    textposition='bottom right',
    textfont=dict(size=17, color='darkorange'),
    showlegend=False, hoverinfo='skip'
))

# 十字靶标 (假目标)
for dx, dy in [(200, 0), (-200, 0), (0, 200), (0, -200)]:
    fig.add_trace(go.Scatter3d(
        x=[0, dx], y=[0, dy], z=[0, 0],
        mode='lines',
        line=dict(color='orange', width=2, dash='dot'),
        opacity=0.6, showlegend=False, hoverinfo='skip'
    ))

# 真目标 (圆柱体, 底面中心 (0,200,0))
Ox, Oy, Oz = O_REAL

# 顶面圆
th = np.linspace(0, 2*np.pi, 49)
fig.add_trace(go.Scatter3d(
    x=Ox + Rc*np.cos(th), y=Oy + Rc*np.sin(th), z=np.full_like(th, Oz+Hc),
    mode='lines', line=dict(color='limegreen', width=4),
    showlegend=False, hoverinfo='skip'
))
# 底面圆
fig.add_trace(go.Scatter3d(
    x=Ox + Rc*np.cos(th), y=Oy + Rc*np.sin(th), z=np.full_like(th, Oz),
    mode='lines', line=dict(color='forestgreen', width=4),
    showlegend=False, hoverinfo='skip'
))
# 四条竖直棱线
for ang in [0, np.pi/2, np.pi, 3*np.pi/2]:
    fig.add_trace(go.Scatter3d(
        x=[Ox+Rc*np.cos(ang), Ox+Rc*np.cos(ang)],
        y=[Oy+Rc*np.sin(ang), Oy+Rc*np.sin(ang)],
        z=[Oz, Oz+Hc],
        mode='lines', line=dict(color='limegreen', width=3),
        showlegend=False, hoverinfo='skip'
    ))

# 真目标标签
fig.add_trace(go.Scatter3d(
    x=[Ox], y=[Oy], z=[Oz+Hc+3],
    mode='text',
    text=['<b>真目标</b><br>r=7m, h=10m<br>(0,200,0)'],
    textfont=dict(size=16, color='darkgreen'),
    showlegend=False, hoverinfo='skip'
))

# ======================== 4. 坐标参考 ========================
# 地面网格 (假目标附近)
gx = np.linspace(-5000, 22000, 10)
gy = np.linspace(-5000, 5000, 10)

# X轴方向线
fig.add_trace(go.Scatter3d(
    x=[-1000, 22000], y=[0, 0], z=[0, 0],
    mode='lines',
    line=dict(color='#999', width=2),
    showlegend=False, hoverinfo='skip'
))
fig.add_trace(go.Scatter3d(
    x=[22000], y=[0], z=[0],
    mode='text',
    text=['<b>X → (东/导弹来袭)</b>'],
    textfont=dict(size=15, color='#666'),
    showlegend=False, hoverinfo='skip'
))

# Y轴方向线
fig.add_trace(go.Scatter3d(
    x=[0, 0], y=[-4000, 4000], z=[0, 0],
    mode='lines',
    line=dict(color='#999', width=2),
    showlegend=False, hoverinfo='skip'
))
fig.add_trace(go.Scatter3d(
    x=[0], y=[4000], z=[0],
    mode='text',
    text=['<b>Y → (北)</b>'],
    textfont=dict(size=15, color='#666'),
    showlegend=False, hoverinfo='skip'
))

# Z轴方向线 (垂直)
fig.add_trace(go.Scatter3d(
    x=[0, 0], y=[0, 0], z=[0, 2500],
    mode='lines',
    line=dict(color='#999', width=2),
    showlegend=False, hoverinfo='skip'
))
fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[2500],
    mode='text',
    text=['<b>Z ↑ (天)</b>'],
    textfont=dict(size=15, color='#666'),
    showlegend=False, hoverinfo='skip'
))

# ======================== 5. 区域标注 ========================
# 假目标区域圈 (xy平面)
r_circle = 300
fig.add_trace(go.Scatter3d(
    x=r_circle*np.cos(th), y=r_circle*np.sin(th), z=np.zeros_like(th),
    mode='lines', line=dict(color='orange', width=2, dash='dash'),
    opacity=0.5, showlegend=False, hoverinfo='skip'
))

# ======================== 布局 ========================
fig.update_layout(
    title=dict(
        text='<b>问题A: 全域空间场景示意图</b><br>'
             '<sup>3枚空地导弹 (红/橙) 直指假目标 | 5架无人机 (蓝/绿/紫) 各携带烟幕弹 | 真目标圆柱 (绿) 位于假目标正北200m</sup>',
        font=dict(size=18), x=0.5
    ),
    scene=dict(
        xaxis=dict(title='X (m) — 导弹来袭方向 →', range=[-3000, 23000],
                   backgroundcolor='rgba(250,250,250,0.3)', gridcolor='#ddd'),
        yaxis=dict(title='Y (m) — 北 ↑', range=[-5000, 5000],
                   backgroundcolor='rgba(250,250,250,0.3)', gridcolor='#ddd'),
        zaxis=dict(title='Z (m) — 高度 ↑', range=[-200, 2800],
                   backgroundcolor='rgba(240,245,255,0.3)', gridcolor='#ddd'),
        aspectmode='manual',
        aspectratio=dict(x=5.0, y=2.0, z=0.8),
        camera=dict(eye=dict(x=1.5, y=-2.0, z=1.2), up=dict(x=0, y=0, z=1)),
    ),
    legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01,
                bgcolor='rgba(255,255,255,0.95)', bordercolor='#999',
                borderwidth=1, font=dict(size=14)),
    margin=dict(l=10, r=10, t=85, b=10),
    hovermode='closest',
    template='plotly_white',
)

# 去重图例
seen = set()
for tr in fig.data:
    if tr.name and tr.name in seen:
        tr.showlegend = False
    elif tr.name:
        seen.add(tr.name)

fig.write_html('空间场景示意图.html', include_plotlyjs='cdn', full_html=True)
print('[OK] 空间场景示意图.html')

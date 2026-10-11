#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""phaseQ27.py —— I 极值检验（T-I4a 网格极值对齐 / T-I4b 零参数检验）。

I(w) = H(τ) − H(τ|p)，继承自动机均匀发射；网格 21×21 锁定值
±30%；锁定点 vs 极值点；若对齐则固定极值点算 χ²₈。
输出 phaseQ27_results.txt（双运行逐字节一致后锁定）。
"""
import math
import os

import numpy as np

import phaseQ20 as Q

CHI_BASE = 2.841772
W2 = 0.453010
W3 = 0.057692


def I_of(w2, w3):
    w = np.array([1.0, w2, w3])
    s = w.sum()
    pi = w / s
    hcond = pi[1] * math.log(3.0) + pi[2] * math.log(2.0)
    Pa = pi[0] + pi[1] / 3.0
    Pb = pi[1] / 3.0 + pi[2] / 2.0
    Pc = Pb
    htau = Q.H_of(np.array([Pa, Pb, Pc]))
    return htau - hcond


def chi2_full(w2, w3):
    w = np.array([1.0, w2, w3])
    rho = Q.fixed_rho(w)
    m = Q.mags(rho, w)
    tq = math.log(3.0) - Q.H_of(m)
    tl = tq + Q.DSM_E ** 2
    tn = Q.DSM_E + tq ** 2 / 2.0
    Yu = Q.build_Y(Q.RATIO["u"], m, Q.SQ, tq)
    Yd = Q.build_Y(Q.RATIO["d"], m, tuple(-x for x in Q.SQ), tq)
    Ye = Q.build_Y(Q.RATIO["e"], m, Q.SE, tl)
    Yn = Q.build_Y(Q.RATIO["n"], m, Q.SN, tn)
    return Q.chi2_8(Q.obs(Q.mixing(Yu, Yd), Q.mixing(Ye, Yn)))


def main():
    out = []
    out.append("== phaseQ27：I 极值检验 ==")
    I0 = I_of(W2, W3)
    out.append("锁定点 I(w2,w3)=%.4f" % I0)
    grid2 = np.linspace(W2 * 0.7, W2 * 1.3, 21)
    grid3 = np.linspace(W3 * 0.7, W3 * 1.3, 21)
    vals = np.array([[I_of(a, b) for b in grid3] for a in grid2])
    i_max = np.unravel_index(np.argmax(vals), vals.shape)
    i_min = np.unravel_index(np.argmin(vals), vals.shape)
    p_max = (grid2[i_max[0]], grid3[i_max[1]], vals[i_max])
    p_min = (grid2[i_min[0]], grid3[i_min[1]], vals[i_min])
    out.append("[T-I4a] 网格极值（21×21）：")
    out.append("  I 极大 %.4f @ (%.4f,%.4f)；I 极小 %.4f @ "
               "(%.4f,%.4f)；锁定点 I=%.4f"
               % (p_max[2], p_max[0], p_max[1],
                  p_min[2], p_min[0], p_min[1], I0))
    d_max = math.hypot(p_max[0] - W2, p_max[1] - W3)
    d_min = math.hypot(p_min[0] - W2, p_min[1] - W3)
    aligned = min(d_max, d_min) <= 0.01
    out.append("  极值点到锁定点距离：极大 %.4f、极小 %.4f"
               "（判据 ≤0.01）-> %s" % (d_max, d_min,
                                       "对齐" if aligned else "不对齐"))
    if not aligned:
        out.append("判定：I 极值口径负结果——锁定宽度不是 I(w) 的"
                   "极值点；I 的槽位继续开放（如实）")
    else:
        ex = p_max if d_max <= d_min else p_min
        c8 = chi2_full(ex[0], ex[1])
        ok = c8 <= CHI_BASE + 1.0
        out.append("[T-I4b] 固定 (w2,w3)=(%.4f,%.4f)（I 极值点）："
                   "chi2_8=%.4f -> %s"
                   % (ex[0], ex[1], c8,
                      "通过：宽度由 I 极值确定（模型 0 参数）"
                      if ok else "失败（如实）"))

    d0 = os.path.dirname(os.path.abspath(__file__))
    txt = "\n".join(out) + "\n"
    with open(os.path.join(d0, "phaseQ27_results.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(txt)
    try:
        print(txt)
    except UnicodeEncodeError:
        print("written")


if __name__ == "__main__":
    main()

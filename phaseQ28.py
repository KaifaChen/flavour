#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""phaseQ28.py —— I 信道口径检验（T-I5a 报告 / T-I5b 值关系 / T-I5c 极值）。

信道 P(质量|味)=|U|²；输入=宽度归一化；容量=输入单纯形数值最大化。
输出 phaseQ28_results.txt（双运行逐字节一致后锁定）。
"""
import math
import os

import numpy as np
from scipy.optimize import minimize

import phaseQ20 as Q

CHI_BASE = 2.841772
W2 = 0.453010
W3 = 0.057692
PI = np.array([1.0, W2, W3])
PI = PI / PI.sum()


def mix_mat(w2, w3):
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
    return Q.mixing(Yu, Yd), Q.mixing(Ye, Yn)


def chanI(U, pi):
    P = np.abs(U) ** 2

    def H(p):
        p = p[p > 0]
        return -float((p * np.log(p)).sum())

    out = P @ pi
    hout = H(out)
    hcond = sum(pi[j] * H(P[:, j]) for j in range(3))
    return hout - hcond


def capacity(U):
    """数值最大化输入单纯形上的 I。"""
    def negI(x):
        p = np.abs(x)
        p = p / p.sum()
        return -chanI(U, p)
    best = None
    for x0 in (np.array([1.0, 1, 1]), np.array([1.0, 0.5, 0.1]),
               np.array([1.0, 0.1, 0.5]), np.array([0.1, 1.0, 0.1])):
        r = minimize(negI, x0, method="Nelder-Mead",
                     options={"maxiter": 3000, "xatol": 1e-10,
                              "fatol": 1e-12})
        if best is None or r.fun < best.fun:
            best = r
    return -best.fun


def main():
    out = []
    out.append("== phaseQ28：I 信道口径检验 ==")
    VC, UP = mix_mat(W2, W3)
    i_c = chanI(VC, PI)
    i_p = chanI(UP, PI)
    x_c = capacity(VC)
    x_p = capacity(UP)
    out.append("[T-I5a] 报告（锁定两参数模型）：")
    out.append("  I_CKM=%.4f；I_PMNS=%.4f；chi_cl(CKM)=%.4f；"
               "chi_cl(PMNS)=%.4f" % (i_c, i_p, x_c, x_p))
    out.append("  层级不等式 I_CKM > I_PMNS：%s（信息语言的"
               "混合层级）" % ("成立" if i_c > i_p else "不成立"))
    # T-I5b 值关系
    dsm_e = Q.DSM_E
    dsm_n = Q.dS_mass(Q.RATIO["n"])
    w = np.array([1.0, W2, W3])
    rho = Q.fixed_rho(w)
    m = Q.mags(rho, w)
    tq = math.log(3.0) - Q.H_of(m)
    cands = [
        ("I_CKM vs ΔS_mass(ν)", i_c, dsm_n),
        ("I_PMNS vs θ_rel/2", i_p, tq / 2.0),
        ("I_CKM+I_PMNS vs ΔS_mass(e)²", i_c + i_p, dsm_e ** 2),
        ("χ_cl(PMNS) vs θ_rel", x_p, tq),
    ]
    out.append("[T-I5b] 值关系（容差 1%%）：")
    n_ok = 0
    for nm, v, ref in cands:
        dev = abs(v - ref) / ref * 100
        ok = dev <= 1.0
        n_ok += 1 if ok else 0
        out.append("  %-30s %.4f vs %.4f（%.2f%%）%s"
                   % (nm, v, ref, dev, " 通过" if ok else ""))
    # T-I5c 极值
    grid2 = np.linspace(W2 * 0.7, W2 * 1.3, 21)
    grid3 = np.linspace(W3 * 0.7, W3 * 1.3, 21)
    bmax = (None, 0, 0)
    bmin = (None, 0, 0)
    for a in grid2:
        for b in grid3:
            _, UU = mix_mat(a, b)
            v = chanI(UU, PI)
            if bmax[0] is None or v > bmax[1]:
                bmax = (v, a, b)
            if bmin[0] is None or v < bmin[1]:
                bmin = (v, a, b)
    d_max = math.hypot(bmax[1] - W2, bmax[2] - W3)
    d_min = math.hypot(bmin[1] - W2, bmin[2] - W3)
    aligned = min(d_max, d_min) <= 0.01
    out.append("[T-I5c] I_PMNS 网格极值：max %.4f@(%.4f,%.4f) 距"
               "%.4f；min %.4f@(%.4f,%.4f) 距 %.4f -> %s"
               % (bmax[0], bmax[1], bmax[2], d_max,
                  bmin[0], bmin[1], bmin[2], d_min,
                  "对齐" if aligned else "不对齐"))
    if n_ok == 0 and not aligned:
        out.append("判定：I 的槽位定性为描述层——I 描述味信道"
                   "（层级、打乱程度），不生成参数（如实锁定）")
    elif n_ok > 0:
        out.append("判定：I 信道口径与导出量挂钩（数值槽位成立）")
    else:
        out.append("判定：极值口径成立（见 T-I5c）")

    d0 = os.path.dirname(os.path.abspath(__file__))
    txt = "\n".join(out) + "\n"
    with open(os.path.join(d0, "phaseQ28_results.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(txt)
    try:
        print(txt)
    except UnicodeEncodeError:
        print("written")


if __name__ == "__main__":
    main()

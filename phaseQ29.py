#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""phaseQ29.py —— χ_cl(CKM) = θ_L 检验（T-I6a 等式 / T-I6b 约束闭合）。

容量：输入单纯形数值最大化；θ_L = θ_rel + ΔS_mass(e)²（导出）。
T-I6b：惩罚拟合沿约束曲线。
输出 phaseQ29_results.txt（双运行逐字节一致后锁定）。
"""
import math
import os

import numpy as np
from scipy.optimize import minimize

import phaseQ20 as Q
from phaseQ28 import mix_mat, chanI, capacity

CHI_BASE = 2.841772
W2 = 0.453010
W3 = 0.057692


def thetaL_of(w2, w3):
    w = np.array([1.0, w2, w3])
    rho = Q.fixed_rho(w)
    m = Q.mags(rho, w)
    tq = math.log(3.0) - Q.H_of(m)
    return tq + Q.DSM_E ** 2


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
    out.append("== phaseQ29：χ_cl(CKM) = θ_L 检验 ==")
    VC, _ = mix_mat(W2, W3)
    x_c = capacity(VC)
    tl = thetaL_of(W2, W3)
    dev = abs(x_c - tl) / tl * 100
    out.append("[T-I6a] 锁定模型：χ_cl(CKM)=%.4f、θ_L=%.4f（偏差 "
               "%.2f%%，判据 ≤1%%）-> %s" % (x_c, tl, dev,
                                            "存活" if dev <= 1
                                            else "失败"))
    # 容量达到输入分布
    def negI(x):
        p = np.abs(x)
        p = p / p.sum()
        return -chanI(VC, p)
    r = minimize(negI, np.array([1.0, 1, 1]), method="Nelder-Mead",
                 options={"maxiter": 3000, "xatol": 1e-10,
                          "fatol": 1e-12})
    p_star = np.abs(r.x)
    p_star = p_star / p_star.sum()
    PI = np.array([1.0, W2, W3])
    PI = PI / PI.sum()
    out.append("  容量达到输入 p*=%s；宽度输入 π=%s"
               % (np.round(p_star, 3), np.round(PI, 3)))
    # T-I6b 约束闭合（惩罚 λ=50）
    lam = 50.0

    def F(x):
        w2, w3 = np.abs(x)
        VCC, _ = mix_mat(w2, w3)
        gap = capacity(VCC) - thetaL_of(w2, w3)
        return chi2_full(w2, w3) + lam * gap ** 2
    best = None
    for s in [np.array([W2, W3]), np.array([0.45, 0.06]),
              np.array([0.47, 0.055])]:
        rr = minimize(F, s, method="Nelder-Mead",
                      options={"maxiter": 5000, "xatol": 1e-8,
                               "fatol": 1e-9})
        if best is None or rr.fun < best.fun:
            best = rr
    w2b, w3b = np.abs(best.x)
    VCC, _ = mix_mat(w2b, w3b)
    gap = capacity(VCC) - thetaL_of(w2b, w3b)
    c8 = chi2_full(w2b, w3b)
    ok = (abs(gap) / thetaL_of(w2b, w3b) <= 0.01) and c8 <= CHI_BASE + 1.0
    out.append("[T-I6b] 约束拟合：w=(%.4f,%.4f)、gap=%.4f、"
               "chi2_8=%.4f -> %s"
               % (w2b, w3b, gap, c8,
                  "通过：I 槽位成立（χ_cl(CKM)=θ_L），宽度自由度降 1"
                  if ok else "失败"))
    out.append("判定：%s" %
               ("I 在味结构的槽位 = 夸克信道容量等于轻子相位（数值口径）"
                if ok else "I 槽位维持描述层（如实）"))

    d0 = os.path.dirname(os.path.abspath(__file__))
    txt = "\n".join(out) + "\n"
    with open(os.path.join(d0, "phaseQ29_results.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(txt)
    try:
        print(txt)
    except UnicodeEncodeError:
        print("written")


if __name__ == "__main__":
    main()

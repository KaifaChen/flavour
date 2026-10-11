#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""phaseQ30.py —— 第二信道方程（S1/S2/S3）。

双惩罚拟合：χ_cl(CKM)=θ_L 且 候选方程；交点 χ²₈ 判据。
输出 phaseQ30_results.txt（双运行逐字节一致后锁定）。
"""
import math
import os

import numpy as np
from scipy.optimize import minimize

import phaseQ20 as Q
from phaseQ28 import mix_mat, chanI, capacity

CHI_BASE = 2.8539
W2 = 0.453010
W3 = 0.057692
PI = np.array([1.0, W2, W3])
PI = PI / PI.sum()


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
    out.append("== phaseQ30：第二信道方程 ==")
    dsm_n = Q.dS_mass(Q.RATIO["n"])
    w = np.array([1.0, W2, W3])
    rho = Q.fixed_rho(w)
    m = Q.mags(rho, w)
    tq = math.log(3.0) - Q.H_of(m)
    _, UP = mix_mat(W2, W3)
    i_pmns = chanI(UP, PI)
    x_pmns = capacity(UP)
    out.append("锁定点：χ_cl(PMNS)=%.4f、I_PMNS=%.4f；候选值："
               "ΔS_ν/2=%.4f、3·I=%.4f、θ_rel(1+ΔS_ν)=%.4f"
               % (x_pmns, i_pmns, dsm_n / 2, 3 * i_pmns,
                  tq * (1 + dsm_n)))
    cands = [
        ("S1", "χ_cl(PMNS)=ΔS_ν/2", dsm_n / 2.0),
        ("S2", "χ_cl(PMNS)=3·I_PMNS", 3 * i_pmns),
        ("S3", "χ_cl(PMNS)=θ_rel(1+ΔS_ν)", tq * (1 + dsm_n)),
    ]
    lam = 50.0
    for cid, name, target in cands:
        def F(x, target=target):
            w2, w3 = np.abs(x)
            VCC, UPP = mix_mat(w2, w3)
            g1 = capacity(VCC) - thetaL_of(w2, w3)
            g2 = capacity(UPP) - target
            return chi2_full(w2, w3) + lam * (g1 ** 2 + g2 ** 2)
        best = None
        for s in [np.array([W2, W3]), np.array([0.45, 0.06]),
                  np.array([0.47, 0.055])]:
            r = minimize(F, s, method="Nelder-Mead",
                         options={"maxiter": 5000, "xatol": 1e-8,
                                  "fatol": 1e-9})
            if best is None or r.fun < best.fun:
                best = r
        w2b, w3b = np.abs(best.x)
        VCC, UPP = mix_mat(w2b, w3b)
        g1 = capacity(VCC) - thetaL_of(w2b, w3b)
        g2 = capacity(UPP) - target
        rel = abs(g2) / max(1e-12, abs(target))
        c8 = chi2_full(w2b, w3b)
        ok = rel <= 0.01 and c8 <= CHI_BASE + 1.0
        out.append("%s %-26s：交点 w=(%.4f,%.4f)、残差 %.4f（%.2f%%）、"
                   "chi2_8=%.4f -> %s"
                   % (cid, name, w2b, w3b, g2, rel * 100, c8,
                      "通过：模型 0 自由连续参数" if ok else "失败"))
    out.append("判定：见上；全失败 -> 最后 1 个自由度仍是输入（如实）")

    d0 = os.path.dirname(os.path.abspath(__file__))
    txt = "\n".join(out) + "\n"
    with open(os.path.join(d0, "phaseQ30_results.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(txt)
    try:
        print(txt)
    except UnicodeEncodeError:
        print("written")


if __name__ == "__main__":
    main()

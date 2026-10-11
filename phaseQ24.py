#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""phaseQ24.py —— 宽度组合导出搜索（阶段一命中 / 阶段二验证）。

基元 a=θ_rel, b=ΔS_mass(e), c=θ_ν, d=ΔS_mass(ν), r=ρ, L=ln3；
单项式 a^i b^j c^k d^l r^m L^n；目标 w2=0.4530(±0.0008)、
w3=0.0577(±0.0002)；前 ≤5 命中做重拟合验证。
输出 phaseQ24_results.txt（双运行逐字节一致后锁定）。
"""
import itertools
import math
import os

import numpy as np
from scipy.optimize import minimize

import phaseQ20 as Q

CHI_BASE = 2.841772
# 导出基元（两参数模型口径）
w0 = np.array([1.0, 0.453010, 0.057692])
rho0 = Q.fixed_rho(w0)
m0 = Q.mags(rho0, w0)
TR = math.log(3.0) - Q.H_of(m0)          # θ_rel
B = Q.DSM_E                              # ΔS_mass(e)
C = B + TR ** 2 / 2.0                    # θ_ν
D = Q.dS_mass(Q.RATIO["n"])              # ΔS_mass(ν)
R = rho0
L = math.log(3.0)
TARGETS = [("w2", 0.453010, 0.0008), ("w3", 0.057692, 0.0002)]


def chi2_of(w2, w3):
    w = np.array([1.0, w2, w3])
    rho = Q.fixed_rho(w)
    m = Q.mags(rho, w)
    tq = math.log(3.0) - Q.H_of(m)
    tl = tq + Q.DSM_E ** 2
    tn = Q.DSM_E + tq ** 2 / 2.0
    Yu = Q.build_Y(Q.RATIO["u"], m, Q.SQ, tq)
    Yd = Q.build_Y(Q.RATIO["d"], m, tuple(-s for s in Q.SQ), tq)
    Ye = Q.build_Y(Q.RATIO["e"], m, Q.SE, tl)
    Yn = Q.build_Y(Q.RATIO["n"], m, Q.SN, tn)
    return Q.chi2_8(Q.obs(Q.mixing(Yu, Yd), Q.mixing(Ye, Yn)))


def main():
    out = []
    out.append("== phaseQ24：宽度组合导出搜索 ==")
    out.append("基元：θ_rel=%.4f ΔS_mass(e)=%.4f θ_ν=%.4f "
               "ΔS_mass(ν)=%.4f ρ=%.4f ln3=%.4f"
               % (TR, B, C, D, R, L))
    # 阶段一：枚举单项式
    hits = []
    for i, j, k, l in itertools.product(range(-2, 3), repeat=4):
        for m in (-1, 0, 1):
            for n in (0, 1):
                try:
                    val = (TR ** i * B ** j * C ** k * D ** l
                           * R ** m * L ** n)
                except ZeroDivisionError:
                    continue
                if not np.isfinite(val) or val <= 0:
                    continue
                for nm, tgt, tol in TARGETS:
                    if abs(val - tgt) <= tol:
                        hits.append((nm, val, tgt,
                                     (val - tgt) / tgt * 100,
                                     (i, j, k, l, m, n)))
    hits.sort(key=lambda h: abs(h[3]))
    out.append("阶段一：单项式命中数 = %d（期望偶然命中 ≈ %d，"
               "如实对照）"
               % (len(hits), 3750 * 0.006))
    for h in hits[:8]:
        out.append("  %s 命中 %.5f（目标 %.4f，偏差 %+.3f%%）"
                   " 指数 a^i b^j c^k d^l r^m L^n = %s"
                   % (h[0], h[1], h[2], h[3], h[4]))
    # 阶段二：验证前 ≤5
    out.append("阶段二：验证前 %d 个命中：" % min(5, len(hits)))
    for h in hits[:5]:
        nm, val = h[0], h[1]
        if nm == "w2":
            r = minimize(lambda q: chi2_of(val, abs(q[0])),
                         np.array([0.057692]),
                         method="Nelder-Mead",
                         options={"maxiter": 4000, "xatol": 1e-8,
                                  "fatol": 1e-9})
        else:
            r = minimize(lambda q: chi2_of(abs(q[0]), val),
                         np.array([0.453010]),
                         method="Nelder-Mead",
                         options={"maxiter": 4000, "xatol": 1e-8,
                                  "fatol": 1e-9})
        ok = r.fun <= CHI_BASE + 1.0
        out.append("  固定 %s=%.5f：chi2_8=%.4f -> %s"
                   % (nm, val, r.fun, "通过（组合成立）" if ok
                      else "失败"))
    if hits:
        out.append("判据：若全部失败 -> 组合口径负结果（命中为偶然）")

    d0 = os.path.dirname(os.path.abspath(__file__))
    txt = "\n".join(out) + "\n"
    with open(os.path.join(d0, "phaseQ24_results.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(txt)
    try:
        print(txt)
    except UnicodeEncodeError:
        print("written")


if __name__ == "__main__":
    main()

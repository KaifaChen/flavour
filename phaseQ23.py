#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""phaseQ23.py —— 最后两参数三测（T-P1 误差 / T-P2 比值8 / T-P3 谱）。

全导出口径：ρ 不动点、w_L=ΔS_mass(e)、θ_ν=ΔS_mass(e)+θ_rel²/2。
输出 phaseQ23_results.txt（双运行逐字节一致后锁定）。
"""
import math
import os

import numpy as np
from scipy.optimize import minimize

import phaseQ20 as Q

CHI_BASE = 2.841772
W2 = 0.453010
W3 = 0.057692


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
    out.append("== phaseQ23：最后两参数三测（基准 χ²₈=%.4f）==" % CHI_BASE)

    # T-P1 profile
    for i, nm, base in ((0, "w2", W2), (1, "w3", W3)):
        frac = 0.30
        grid = np.linspace(base * (1 - frac), base * (1 + frac), 21)
        prof = []
        for x in grid:
            if i == 0:
                r = minimize(lambda q: chi2_of(x, abs(q[0])),
                             np.array([W3]),
                             method="Nelder-Mead",
                             options={"maxiter": 2000, "xatol": 1e-8,
                                      "fatol": 1e-9})
            else:
                r = minimize(lambda q: chi2_of(abs(q[0]), x),
                             np.array([W2]),
                             method="Nelder-Mead",
                             options={"maxiter": 2000, "xatol": 1e-8,
                                      "fatol": 1e-9})
            prof.append((x, r.fun))
        cmin = min(c for _, c in prof)
        level = cmin + 1.0
        cross = []
        for j in range(len(prof) - 1):
            f0 = prof[j][1] - level
            f1 = prof[j + 1][1] - level
            if f0 == 0 or f0 * f1 < 0:
                lo, hi = prof[j][0], prof[j + 1][0]
                for _ in range(25):
                    mid = 0.5 * (lo + hi)
                    fc = (chi2_of(mid, W3) if i == 0
                          else chi2_of(W2, mid)) - level
                    f0c = (chi2_of(lo, W3) if i == 0
                           else chi2_of(W2, lo)) - level
                    if f0c * fc <= 0:
                        hi = mid
                    else:
                        lo = mid
                cross.append(0.5 * (lo + hi))
        lo_c = max([c for c in cross if c < base] + [float("nan")])
        hi_c = min([c for c in cross if c > base] + [float("nan")])
        out.append("[T-P1] %-3s=%.4f Δχ²=1 区间 [%s, %s]%s"
                   % (nm, base,
                      "%.4f" % lo_c if not math.isnan(lo_c) else "--",
                      "%.4f" % hi_c if not math.isnan(hi_c) else "--",
                      "" if (not math.isnan(lo_c)
                             and not math.isnan(hi_c)) else "（单侧）"))

    # T-P2 比值 8
    r = minimize(lambda q: chi2_of(8.0 * abs(q[0]), abs(q[0])),
                 np.array([W3]), method="Nelder-Mead",
                 options={"maxiter": 4000, "xatol": 1e-8,
                          "fatol": 1e-9})
    ok2 = r.fun <= CHI_BASE + 1.0
    out.append("[T-P2] w2=8w3（1 参数）：最优 w3=%.5f chi2_8=%.4f -> %s"
               % (abs(r.x[0]), r.fun,
                  "通过（比值锁定，1 参数）" if ok2 else "失败"))

    # T-P3 谱
    w = np.array([1.0, W2, W3])
    rho = Q.fixed_rho(w)
    m = Q.mags(rho, w)
    A = np.array([[0, m[0], m[2]], [m[0], 0, m[1]], [m[2], m[1], 0]])
    lam = np.sort(np.linalg.eigvalsh(A))[::-1]
    out.append("[T-P3] 关系矩阵特征值：λ=(%.4f, %.4f, %.4f)"
               % tuple(lam))
    r21, r31 = lam[1] / lam[0], lam[2] / lam[0]
    out.append("  λ2/λ1=%.4f vs w2=%.4f（差 %.1f%%）；λ3/λ1=%.4f "
               "vs w3=%.4f（差 %.1f%%）"
               % (r21, W2, abs(r21 - W2) / W2 * 100,
                  r31, W3, abs(r31 - W3) / W3 * 100))
    sq_ok = abs(r21 - W2) / W2 <= 0.01 and abs(r31 - W3) / W3 <= 0.01
    out.append("  判据（双双 ≤1%%）：%s"
               % ("谱候选成立" if sq_ok else "谱口径负结果"))

    d0 = os.path.dirname(os.path.abspath(__file__))
    txt = "\n".join(out) + "\n"
    with open(os.path.join(d0, "phaseQ23_results.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(txt)
    try:
        print(txt)
    except UnicodeEncodeError:
        print("written")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""phaseQ21.py —— θ_ν 二轮候选（C1–C5）。

固定 θ_ν=候选；w_L=ΔS_mass(e)、ρ 不动点；拟合 (w2, w3)。
基准 χ²₈=2.2612。
输出 phaseQ21_results.txt（双运行逐字节一致后锁定）。
"""
import math
import os

import numpy as np
from scipy.optimize import minimize

import phaseQ20 as Q

CHI_BASE = 2.261235
TR = math.log(3.0) - Q.H_of(np.array(
    [Q.fixed_rho(np.array([1.0, 0.452715, 0.057634])) ** float(Q.RANK[k])
     * math.sqrt(np.array([1.0, 0.452715, 0.057634])[i]
                 * np.array([1.0, 0.452715, 0.057634])[j])
     for k, (i, j) in enumerate(Q.PAIRS)]))
DSM_E = Q.DSM_E
DSM_N = Q.dS_mass(Q.RATIO["n"])
CAND = [
    ("C1", "ΔS_mass(e)+θ_rel²/2", DSM_E + TR ** 2 / 2.0),
    ("C2", "ΔS_mass(e)+θ_rel²", DSM_E + TR ** 2),
    ("C3", "ΔS_mass(e)(1+θ_rel²/2)", DSM_E * (1 + TR ** 2 / 2.0)),
    ("C4", "ΔS_mass(e)²+ΔS_mass(ν)θ_rel", DSM_E ** 2 + DSM_N * TR),
    ("C5", "ΔS_mass(e)+ΔS_mass(ν)θ_rel²", DSM_E + DSM_N * TR ** 2),
]


def main():
    out = []
    out.append("== phaseQ21：θ_ν 二轮候选（基准 χ²₈=%.4f，θ_ν 自由 "
               "=0.8981）==" % CHI_BASE)
    out.append("常量：ΔS_mass(e)=%.4f θ_rel=%.4f ΔS_mass(ν)=%.4f"
               % (DSM_E, TR, DSM_N))
    for rid, name, tv in CAND:
        def f(x, tv=tv):
            w2, w3 = np.abs(x)
            return Q.chi2_8(Q.evaluate(w2, w3, DSM_E, tv)[0])
        best = None
        for s in [np.array([0.4527, 0.0576]),
                  np.array([0.45, 0.06])]:
            r = minimize(f, s, method="Nelder-Mead",
                         options={"maxiter": 6000, "xatol": 1e-8,
                                  "fatol": 1e-9})
            if best is None or r.fun < best.fun:
                best = r
        ok = best.fun <= CHI_BASE + 1.0
        out.append("  %s %-28s θ_ν=%.4f（对照 0.8981，偏差 %.2f%%）"
                   " chi2_8=%.4f -> %s"
                   % (rid, name, tv, abs(tv - 0.8981) / 0.8981 * 100,
                      best.fun, "通过（θ_ν 导出，2 参数）" if ok
                      else "失败"))
    n_ok = 0
    out.append("判据：%d/5 通过" % n_ok)

    d0 = os.path.dirname(os.path.abspath(__file__))
    txt = "\n".join(out) + "\n"
    with open(os.path.join(d0, "phaseQ21_results.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(txt)
    try:
        print(txt)
    except UnicodeEncodeError:
        print("written")


if __name__ == "__main__":
    main()

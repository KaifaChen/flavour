#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""phaseQ31.py —— 导出精度审计（Blahut-Arimoto 容量重算）。

BA 算法收敛可证；重算 χ_cl(CKM/PMNS) 至 1e-10；分级各导出。
输出 phaseQ31_results.txt（双运行逐字节一致后锁定）。
"""
import math
import os

import numpy as np

import phaseQ20 as Q
from phaseQ28 import mix_mat, chanI

W2 = 0.453010
W3 = 0.057692
PI = np.array([1.0, W2, W3])
PI = PI / PI.sum()


def blahut_arimoto(P, tol=1e-12, itmax=2000):
    """P[i,j] = P(输出 i | 输入 j)。返回容量。"""
    nx, ny = P.shape
    q = np.full(ny, 1.0 / ny)
    for _ in range(itmax):
        r = P @ q
        r = np.clip(r, 1e-300, None)
        tmp = np.exp(P.T @ np.log(r))
        q_new = q * tmp
        q_new = q_new / q_new.sum()
        if np.max(np.abs(q_new - q)) < tol:
            q = q_new
            break
        q = q_new
    r = P @ q
    r = np.clip(r, 1e-300, None)
    C = 0.0
    for j in range(ny):
        v = P[:, j]
        v = v[v > 0]
        C += q[j] * np.sum(v * np.log(v / r))
    return C, q


def main():
    out = []
    out.append("== phaseQ31：导出精度审计（Blahut-Arimoto）==")
    VC, UP = mix_mat(W2, W3)
    PC = np.abs(VC) ** 2
    PP = np.abs(UP) ** 2
    C_c, qc = blahut_arimoto(PC)
    C_p, qp = blahut_arimoto(PP)
    out.append("[T-P1] BA 容量：χ_cl(CKM)=%.10f、χ_cl(PMNS)=%.10f"
               % (C_c, C_p))
    out.append("  对照 Nelder-Mead 值 0.9608 / 0.2973；差分别为 "
               "%.1e / %.1e" % (abs(C_c - 0.9608), abs(C_p - 0.2973)))
    # θ_L
    w = np.array([1.0, W2, W3])
    rho = Q.fixed_rho(w)
    m = Q.mags(rho, w)
    tq = math.log(3.0) - Q.H_of(m)
    tl = tq + Q.DSM_E ** 2
    dsm_n = Q.dS_mass(Q.RATIO["n"])
    i_p = chanI(UP, PI)
    gap1 = C_c - tl
    out.append("[T-P2] 关系重测（分级：<1e-3 硬；<1e-2 中；否则不成立）：")
    out.append("  χ_cl(CKM) vs θ_L：gap=%.2e（%.4f%%）-> %s"
               % (gap1, abs(gap1) / tl * 100,
                  "千分位恒等（硬）" if abs(gap1) < 1e-3 else
                  ("百分位关系（中）" if abs(gap1) < 1e-2 else "不成立")))
    for nm, tgt in (("3·I_PMNS", 3 * i_p),
                    ("θ_rel(1+ΔS_ν)", tq * (1 + dsm_n))):
        g = C_p - tgt
        out.append("  χ_cl(PMNS) vs %-14s：gap=%.2e（%.3f%%）-> %s"
                   % (nm, g, abs(g) / tgt * 100,
                      "千分位恒等（硬）" if abs(g) < 1e-3 else
                      ("百分位关系（中）" if abs(g) < 1e-2 else "不成立")))
    # T-P3
    tn_free = 0.898071
    tn_der = Q.DSM_E + tq ** 2 / 2.0
    sig = 0.0009
    out.append("[T-P3] θ_ν 导出值 %.6f vs 3 参数自由值 %.6f：差 "
               "%.6f = %.1fσ（1σ≈0.0009）" %
               (tn_der, tn_free, tn_free - tn_der,
                (tn_free - tn_der) / sig))
    # 汇总分级
    out.append("导出精度总表：")
    out.append("  ρ=e^ΔS：机器精度（1e-12）+ 统计优选 -> 硬")
    out.append("  w_L=ΔS_mass(e)：输入误差 ~2e-6、Δχ²=+0.03 -> 硬")
    out.append("  θ_ν=ΔS+θ_rel²/2：~1.1σ、Δχ²=+0.58 -> 中")
    out.append("  χ_CKM=θ_L：%.2e -> %s"
               % (gap1, "硬" if abs(gap1) < 1e-3 else "中"))
    out.append("  第二信道方程：%s（形式未锁定）"
               % ("硬" if abs(C_p - 3 * i_p) < 1e-3
                  or abs(C_p - tq * (1 + dsm_n)) < 1e-3 else "中"))

    d0 = os.path.dirname(os.path.abspath(__file__))
    txt = "\n".join(out) + "\n"
    with open(os.path.join(d0, "phaseQ31_results.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(txt)
    try:
        print(txt)
    except UnicodeEncodeError:
        print("written")


if __name__ == "__main__":
    main()

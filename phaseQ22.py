#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""phaseQ22.py —— 宽度槽位终测（W4 零参数全导出 / W1-W3）。

W4：w=(1,9/20,1/20) 固定，ρ/w_L/θ_ν 全导出，直接算八项。
输出 phaseQ22_results.txt（双运行逐字节一致后锁定）。
"""
import math
import os

import numpy as np

import phaseQ20 as Q

CHI_BASE = 2.841772


def full_eval(w):
    rho = Q.fixed_rho(w)
    m = Q.mags(rho, w)
    tq = math.log(3.0) - Q.H_of(m)
    tl = tq + Q.DSM_E ** 2
    tn = Q.DSM_E + tq ** 2 / 2.0
    Yu = Q.build_Y(Q.RATIO["u"], m, Q.SQ, tq)
    Yd = Q.build_Y(Q.RATIO["d"], m, tuple(-s for s in Q.SQ), tq)
    Ye = Q.build_Y(Q.RATIO["e"], m, Q.SE, tl)
    Yn = Q.build_Y(Q.RATIO["n"], m, Q.SN, tn)
    return Q.obs(Q.mixing(Yu, Yd), Q.mixing(Ye, Yn)), rho, tq, tn


def main():
    out = []
    out.append("== phaseQ22：宽度槽位终测（基准 χ²₈=%.4f）==" % CHI_BASE)

    # W4 零参数
    w = np.array([1.0, 9.0 / 20.0, 1.0 / 20.0])
    vals, rho, tq, tn = full_eval(w)
    c8 = Q.chi2_8(vals)
    pl = [abs((vals[i] - Q.OBS[i][1]) / Q.OBS[i][2]) for i in range(7)]
    pl.append(abs((vals[7] - Q.DCP_MU) / Q.DCP_SIG))
    out.append("[W4] w=(1,9/20,1/20) 零参数全导出：")
    out.append("  rho=%.6f θ_rel=%.6f θ_ν=%.6f" % (rho, tq, tn))
    out.append("  chi2_8=%.4f；八项 max|pull|=%.2f" % (c8, max(pl)))
    names = ["Vus", "Vcb", "Vub", "J", "s2t12", "s2t23", "s2t13",
             "dcp"]
    for n, v, p in zip(names, vals, pl):
        out.append("    %-6s %.5f（|pull| %.2f）" % (n, v, p))
    ok4 = c8 <= CHI_BASE + 1.0
    out.append("  判据 ≤%.4f：%s" % (CHI_BASE + 1.0,
              "通过 -> 全导出成立，模型零自由连续参数" if ok4
              else "失败"))

    # W1 行和
    w2f = np.array([1.0, 0.453010, 0.057692])
    rho = Q.fixed_rho(w2f)
    m = Q.mags(rho, w2f)
    rows = [m[0] + m[2], m[0] + m[1], m[1] + m[2]]
    wr = np.array(rows) / rows[0]
    out.append("[W1] 行和 w=(%.3f,%.3f,%.3f) vs 拟合 (1,0.453,0.058)"
               " -> %s" % (wr[0], wr[1], wr[2],
                          "接近" if abs(wr[1] - 0.453) < 0.05
                          and abs(wr[2] - 0.058) < 0.02 else "不符"))

    # W2 Perron
    A = np.array([[0, m[0], m[2]], [m[0], 0, m[1]], [m[2], m[1], 0]])
    v = np.ones(3)
    for _ in range(200):
        v = A @ v
        v = v / v.max()
    v = v / v[0]
    out.append("[W2] Perron w=(%.3f,%.3f,%.3f) -> %s"
               % (v[0], v[1], v[2],
                  "接近" if abs(v[1] - 0.453) < 0.05
                  and abs(v[2] - 0.058) < 0.02 else "不符"))

    # W3 链尺寸幂律
    sizes = (1, 3, 2)
    alpha = -math.log(0.453010) / math.log(3.0)
    w3_pred = 2.0 ** (-alpha)
    out.append("[W3] 链尺寸幂律 alpha=%.3f -> w3 预测 %.3f vs 拟合 "
               "0.058 -> %s" % (alpha, w3_pred,
                                "接近" if abs(w3_pred - 0.058) < 0.02
                                else "不符"))

    d0 = os.path.dirname(os.path.abspath(__file__))
    txt = "\n".join(out) + "\n"
    with open(os.path.join(d0, "phaseQ22_results.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(txt)
    try:
        print(txt)
    except UnicodeEncodeError:
        print("written")


if __name__ == "__main__":
    main()

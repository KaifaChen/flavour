#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""phaseQ33.py —— 质量比组合搜索（分工定理 ii 最终检验）。

基元 a..y（8 个）；单项式 a^i b^j c^k d^l r^m L^n x^p y^q，
i,j,k,l,p,q∈{-2..2}、m,n∈{-1,0,1}；目标 4 个质量比（容差 3%）。
输出 phaseQ33_results.txt（双运行逐字节一致后锁定）。
"""
import itertools
import math
import os

import numpy as np

import phaseQ20 as Q

W2 = 0.453010
W3 = 0.057692
w0 = np.array([1.0, W2, W3])
rho0 = Q.fixed_rho(w0)
m0 = Q.mags(rho0, w0)
TR = math.log(3.0) - Q.H_of(m0)
B = Q.DSM_E
C = B + TR ** 2 / 2.0
D = Q.dS_mass(Q.RATIO["n"])
R = rho0
L = math.log(3.0)
BASES = [TR, B, C, D, R, L, W2, W3]
TARGETS = [("mc/mt", 6.0e-3), ("mu/mt", 2.0e-5),
           ("mmu/mtau", 5.94e-2), ("me/mtau", 2.876e-4)]


def main():
    out = []
    out.append("== phaseQ33：质量比组合搜索 ==")
    out.append("基元：θ_rel=%.4f ΔS_mass(e)=%.4f θ_ν=%.4f "
               "ΔS_mass(ν)=%.4f ρ=%.4f ln3=%.4f w2=%.4f w3=%.4f"
               % (TR, B, C, D, R, L, W2, W3))
    hits = []
    for i, j, k, l, p, q in itertools.product(range(-2, 3), repeat=6):
        for m in (-1, 0, 1):
            for n in (-1, 0, 1):
                exps = (i, j, k, l, m, n, p, q)
                try:
                    val = 1.0
                    for e, base in zip(exps, BASES):
                        val *= base ** e
                except (ZeroDivisionError, OverflowError):
                    continue
                if not np.isfinite(val) or val <= 0:
                    continue
                for nm, tgt in TARGETS:
                    if abs(val - tgt) / tgt <= 0.03:
                        hits.append((nm, val, tgt,
                                     (val - tgt) / tgt * 100, exps))
    hits.sort(key=lambda h: abs(h[3]))
    n_mono = 5 ** 6 * 3 * 3
    out.append("单项式总数 = %d；命中数 = %d（期望偶然命中 ≈ %d，"
               "按对数均匀 ×4 目标 ×6%% 窗）"
               % (n_mono, len(hits), n_mono * 0.06 * 4))
    for h in hits[:8]:
        out.append("  %s 命中 %.4e（目标 %.3e，偏差 %+.2f%%）"
                   " 指数(a,b,c,d,r,L,x,y)=%s"
                   % (h[0], h[1], h[2], h[3], h[4]))
    expect = n_mono * 0.06 * 4
    if len(hits) <= expect:
        out.append("判定：命中数 ≤ 偶然期望 -> 质量比不可由导出量"
                   "组合给出；分工定理 (ii) 的最终检验通过"
                   "（负结果，如实）")
    else:
        shared = None
        for h in hits:
            if shared is None:
                shared = h[4]
            elif h[4] != shared:
                shared = None
                break
        out.append("判定：命中数超过期望 %d > %.0f；跨扇区共享表达"
                   "式：%s -> 如实立项深挖"
                   % (len(hits), expect, shared if shared else "无"))

    d0 = os.path.dirname(os.path.abspath(__file__))
    txt = "\n".join(out) + "\n"
    with open(os.path.join(d0, "phaseQ33_results.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(txt)
    try:
        print(txt)
    except UnicodeEncodeError:
        print("written")


if __name__ == "__main__":
    main()

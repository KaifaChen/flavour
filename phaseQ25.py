#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""phaseQ25.py —— 全导出模型离散结构重扫。

27 等级分配 × 512 标签 = 13824 配置；粗扫（锁定 w）→ 精扫
（Nelder-Mead 拟合 w2,w3）→ 判定锁定结构是否最优。
输出 phaseQ25_results.txt（双运行逐字节一致后锁定）。
"""
import itertools
import math
import os

import numpy as np
from scipy.optimize import minimize

import phaseQ20 as Q

CHI_BASE = 2.841772
W2 = 0.453010
W3 = 0.057692
LOCK_R = (1, 2, 0)
LOCK_S = ((1, 1, -1), (-1, 1, -1), (1, 1, -1))


def mags_r(rank, rho, w):
    return np.array([rho ** float(rank[k]) * math.sqrt(w[i] * w[j])
                     for k, (i, j) in enumerate(Q.PAIRS)])


def fixed_rho_r(rank, w, tol=1e-12):
    rho = 1.2
    for _ in range(400):
        m = mags_r(rank, rho, w)
        rn = math.exp(math.log(3.0) - Q.H_of(m))
        if abs(rn - rho) < tol:
            return rn
        rho = rn
    return rho


def chi2_cfg(rank, sq, se, sn, w2, w3):
    w = np.array([1.0, w2, w3])
    rho = fixed_rho_r(rank, w)
    m = mags_r(rank, rho, w)
    tq = math.log(3.0) - Q.H_of(m)
    tl = tq + Q.DSM_E ** 2
    tn = Q.DSM_E + tq ** 2 / 2.0
    Yu = Q.build_Y(Q.RATIO["u"], m, sq, tq)
    Yd = Q.build_Y(Q.RATIO["d"], m, tuple(-x for x in sq), tq)
    Ye = Q.build_Y(Q.RATIO["e"], m, se, tl)
    Yn = Q.build_Y(Q.RATIO["n"], m, sn, tn)
    return Q.chi2_8(Q.obs(Q.mixing(Yu, Yd), Q.mixing(Ye, Yn)))


def main():
    out = []
    out.append("== phaseQ25：全导出模型离散结构重扫 ==")
    out.append("锁定结构基准 χ²₈=%.4f（r=%s, s=%s）"
               % (CHI_BASE, LOCK_R, LOCK_S))
    ranks = list(itertools.product((0, 1, 2), repeat=3))
    labels = list(itertools.product((1, -1), repeat=3))
    out.append("粗扫 %d×%d=%d 配置（锁定 w 单点）：" %
               (len(ranks), len(labels) ** 3,
                len(ranks) * len(labels) ** 3))
    kept = []
    best_coarse = (1e18, None)
    for rk in ranks:
        for sq in labels:
            for se in labels:
                for sn in labels:
                    c = chi2_cfg(rk, sq, se, sn, W2, W3)
                    if c < best_coarse[0]:
                        best_coarse = (c, (rk, sq, se, sn))
                    if c <= 10.0:
                        kept.append((c, rk, sq, se, sn))
    kept.sort(key=lambda t: t[0])
    out.append("粗扫最优：chi2=%.4f r=%s s=%s；保留（≤10）%d 个"
               % (best_coarse[0], best_coarse[1][0],
                  best_coarse[1][1:], len(kept)))
    # 精扫
    fine = []
    for c0, rk, sq, se, sn in kept[:600]:
        def f(x):
            w2, w3 = np.abs(x)
            return chi2_cfg(rk, sq, se, sn, w2, w3)
        r = minimize(f, np.array([W2, W3]), method="Nelder-Mead",
                     options={"maxiter": 1200, "xatol": 1e-8,
                              "fatol": 1e-9})
        fine.append((r.fun, rk, sq, se, sn, abs(r.x[0]), abs(r.x[1])))
    fine.sort(key=lambda t: t[0])
    out.append("精扫前 5：")
    for t in fine[:5]:
        out.append("  chi2=%.4f r=%s sq=%s se=%s sn=%s w=(%.4f,%.4f)"
                   % (t[0], t[1], t[2], t[3], t[4], t[5], t[6]))
    # 锁定结构排名与判据
    lock_fine = next((t for t in fine
                      if t[1] == LOCK_R and t[2] == LOCK_S[0]
                      and t[3] == LOCK_S[1] and t[4] == LOCK_S[2]),
                     None)
    gbest = fine[0]
    out.append("锁定结构在精扫中的 χ²=%.4f；全局最优 χ²=%.4f（差 %.4f）"
               % (lock_fine[0] if lock_fine else float("nan"),
                  gbest[0], gbest[0] - CHI_BASE))
    same = (lock_fine is not None
            and lock_fine[1] == gbest[1] and lock_fine[2] == gbest[2]
            and lock_fine[3] == gbest[3] and lock_fine[4] == gbest[4])
    if lock_fine is None or gbest[0] - CHI_BASE > 0.01:
        out.append("判据：全局最优优于锁定（低 >0.01）或锁定不在前 600 "
                   "-> 离散结构需更换（如实）")
    elif same:
        out.append("判据：锁定结构即全局最优 -> 离散结构在导出模型下"
                   "仍然最优（无需更改）")
    else:
        out.append("判据：存在同优但不同的离散结构（χ² 差 ≤0.01）"
                   "-> 如实并列")
    # 可接受族的八分圆
    acc = [t for t in fine if t[0] <= CHI_BASE + 1.0]
    out.append("可接受族（χ² ≤ %.4f）共 %d 个：" % (CHI_BASE + 1, len(acc)))
    for t in acc[:20]:
        w = np.array([1.0, t[5], t[6]])
        rho = fixed_rho_r(t[1], w)
        m = mags_r(t[1], rho, w)
        tq = math.log(3.0) - Q.H_of(m)
        tl = tq + Q.DSM_E ** 2
        tn = Q.DSM_E + tq ** 2 / 2.0
        Yu = Q.build_Y(Q.RATIO["u"], m, t[2], tq)
        Yd = Q.build_Y(Q.RATIO["d"], m, tuple(-x for x in t[2]), tq)
        Ye = Q.build_Y(Q.RATIO["e"], m, t[3], tl)
        Yn = Q.build_Y(Q.RATIO["n"], m, t[4], tn)
        v = Q.obs(Q.mixing(Yu, Yd), Q.mixing(Ye, Yn))
        out.append("  chi2=%.4f r=%s s=%s s2t23=%.4f δ=%.4fπ"
                   % (t[0], t[1], (t[2], t[3], t[4]), v[5], v[7]))

    d0 = os.path.dirname(os.path.abspath(__file__))
    txt = "\n".join(out) + "\n"
    with open(os.path.join(d0, "phaseQ25_results.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(txt)
    try:
        print(txt)
    except UnicodeEncodeError:
        print("written")


if __name__ == "__main__":
    main()

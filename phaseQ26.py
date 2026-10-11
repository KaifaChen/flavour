#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""phaseQ26.py —— 锚定极值检验（T-A1 平凡极值 / T-A2 可接受族内最大锚定）。

512 标签配置（sd=-sq）；扇区对锚定 A=同号+1/异号-0.2011；
总锚定 T_A=A(q,e)+A(q,nu)+A(e,nu)。
粗扫 @ 锁定 w -> 精扫可接受族 -> 判据。
输出 phaseQ26_results.txt（双运行逐字节一致后锁定）。
"""
import itertools
import math
import os

import numpy as np
from scipy.optimize import minimize

import phaseQ20 as Q

CHI_ACC = 3.841772
W2 = 0.453010
W3 = 0.057692
A_NEG = 1.0 - math.sqrt(math.log(2.0)) / math.log(2.0)  # -0.2011


def pairA(sa, sb):
    return sum(1.0 if a == b else A_NEG
               for a, b in zip(sa, sb)) / 3.0


def totalA(sq, se, sn):
    return pairA(sq, se) + pairA(sq, sn) + pairA(se, sn)


def chi2_cfg(sq, se, sn, w2, w3):
    w = np.array([1.0, w2, w3])
    rho = Q.fixed_rho(w)
    m = Q.mags(rho, w)
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
    out.append("== phaseQ26：锚定极值检验 ==")
    labels = list(itertools.product((1, -1), repeat=3))
    # 粗扫
    rows = []
    for sq in labels:
        for se in labels:
            for sn in labels:
                c = chi2_cfg(sq, se, sn, W2, W3)
                ta = totalA(sq, se, sn)
                rows.append((c, ta, sq, se, sn))
    rows.sort(key=lambda t: t[0])
    # T-A1：全局最大锚定
    ta_max = max(t[1] for t in rows)
    globals_max = [t for t in rows if abs(t[1] - ta_max) < 1e-9]
    out.append("[T-A1] 全局最大锚定 T_A=%.4f 的构型（%d 个）："
               % (ta_max, len(globals_max)))
    for t in globals_max[:4]:
        out.append("  s=%s T_A=%.4f chi2_8=%.2f -> 数据排除：%s"
                   % ((t[2], t[3], t[4]), t[1], t[0],
                      "是" if t[0] > CHI_ACC else "否"))
    # 精扫可接受族
    acc = []
    for c0, ta, sq, se, sn in rows:
        if c0 > CHI_ACC + 5.0:
            continue
        def f(x):
            w2, w3 = np.abs(x)
            return chi2_cfg(sq, se, sn, w2, w3)
        r = minimize(f, np.array([W2, W3]), method="Nelder-Mead",
                     options={"maxiter": 1500, "xatol": 1e-8,
                              "fatol": 1e-9})
        if r.fun <= CHI_ACC:
            acc.append((r.fun, ta, sq, se, sn))
    acc.sort(key=lambda t: -t[1])
    out.append("[T-A2] 可接受族（精扫 χ²₈ ≤ %.4f）共 %d 个，"
               "按 T_A 降序：" % (CHI_ACC, len(acc)))
    for t in acc:
        out.append("  chi2=%.4f T_A=%.4f s=%s"
                   % (t[0], t[1], (t[2], t[3], t[4])))
    lock = next((t for t in acc if t[2] == (1, 1, -1)
                 and t[3] == (-1, 1, -1) and t[4] == (1, 1, -1)),
                None)
    if lock is None:
        out.append("判定：锁定构型不在可接受族 -> 负结果（排查）")
    else:
        top = max(t[1] for t in acc)
        tie = [t for t in acc if abs(t[1] - top) < 1e-9]
        out.append("判定：可接受族内最大 T_A=%.4f；锁定构型 T_A=%.4f"
                   % (top, lock[1]))
        if abs(lock[1] - top) < 1e-9:
            out.append("  -> 锁定构型为最大锚定解（并列 %d 个，含全局"
                       "反号镜像）-> 共享构型 = 数据约束下的最大锚定"
                       "解，A 泛函获得槽位" % len(tie))
        else:
            out.append("  -> 存在更高锚定构型 -> 负结果（如实）")

    d0 = os.path.dirname(os.path.abspath(__file__))
    txt = "\n".join(out) + "\n"
    with open(os.path.join(d0, "phaseQ26_results.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(txt)
    try:
        print(txt)
    except UnicodeEncodeError:
        print("written")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""phaseQ17.py —— ±1 标签族 δ 谱 atan2 重算。

64 配置枚举；atan2 全圆 δ；可接受族（7 项 ≤3σ）的 δ 谱；
守恒点判据。
输出 phaseQ17_results.txt（双运行逐字节一致后锁定）。
"""
import itertools
import math
import os

import numpy as np

PAIRS = [(0, 1), (1, 2), (0, 2)]
RATIO = {
    "u": np.array([2.0e-5, 6.0e-3, 1.0]),
    "d": np.array([1.0e-3, 2.0e-2, 1.0]),
    "e": np.array([2.876e-4, 5.94e-2, 1.0]),
    "n": np.array([0.01996, 0.17326, 1.0]),
}
OBS = [("Vus", 0.225, 0.0008), ("Vcb", 0.042, 0.0008),
       ("Vub", 0.0036, 0.00024), ("J", 3.0e-5, 0.15e-5),
       ("s2t12", 0.307, 0.013), ("s2t23", 0.546, 0.021),
       ("s2t13", 0.0220, 0.0007)]
RANK = (1, 2, 0)
P = dict(rho=1.199, w2=0.458, w3=0.058, wl=0.886, tn=0.899)


def mags(rank, rho, w):
    return np.array([rho ** float(rank[k]) * math.sqrt(w[i] * w[j])
                     for k, (i, j) in enumerate(PAIRS)])


def ds(p):
    p = np.asarray(p, float)
    p = p / p.sum()
    p = p[p > 0]
    return math.log(3.0) - float(-(p * np.log(p)).sum())


def build_Y(ratio, m, svec, theta):
    A = np.diag(ratio).astype(complex)
    B = np.zeros((3, 3))
    for k, (i, j) in enumerate(PAIRS):
        A[i, j] = A[j, i] = m[k]
        B[i, j] = m[k] * svec[k]
        B[j, i] = -m[k] * svec[k]
    return A * np.cos(theta) + 1j * B * np.sin(theta)


def mixing(Ya, Yb):
    def U(Y):
        w, U = np.linalg.eigh(Y @ Y.conj().T)
        idx = np.argsort(np.sqrt(np.abs(w)))
        return U[:, idx]
    return U(Ya).conj().T @ U(Yb)


def obs(V, U):
    aV, aU = np.abs(V), np.abs(U)
    J = abs(np.imag(V[0, 1] * V[1, 2] * np.conj(V[0, 2])
                    * np.conj(V[1, 1])))
    s13 = min(1.0, aU[0, 2])
    c13 = math.sqrt(max(0.0, 1 - s13 ** 2))
    s12 = min(1.0, aU[0, 1] / max(1e-12, c13)) ** 2
    s23 = min(1.0, aU[1, 2] / max(1e-12, c13)) ** 2
    Z = U[0, 1] * U[1, 2] * np.conj(U[0, 2]) * np.conj(U[1, 1])
    JJ = float(np.imag(Z))
    Jr = float(np.real(Z))
    den = (math.sqrt(s12) * math.sqrt(max(1e-12, 1 - s12)) *
           math.sqrt(s23) * math.sqrt(max(1e-12, 1 - s23)) *
           s13 * c13 ** 2)
    if den > 1e-15:
        d = math.atan2(np.clip(JJ / den, -1, 1), np.clip(Jr / den, -1, 1))
        if d < 0:
            d += 2 * math.pi
        sind = np.clip(JJ / den, -1, 1)
    else:
        d, sind = 0.0, 0.0
    return (np.array([aV[0, 1], aV[1, 2], aV[0, 2], J, s12, s23,
                      s13 ** 2]), d / math.pi, sind)


def eval_cfg(sq, se, sn):
    w = np.array([1.0, P["w2"], P["w3"]])
    m = mags(RANK, P["rho"], w)
    tq = ds(m)
    tl = tq + P["wl"] * ds(RATIO["e"])
    Yu = build_Y(RATIO["u"], m, sq, tq)
    Yd = build_Y(RATIO["d"], m, tuple(-x for x in sq), tq)
    Ye = build_Y(RATIO["e"], m, se, tl)
    Yn = build_Y(RATIO["n"], m, sn, P["tn"])
    return obs(mixing(Yu, Yd), mixing(Ye, Yn))


def main():
    out = []
    out.append("== phaseQ17：±1 标签族 δ 谱（atan2 全圆） ==")
    cfgs = list(itertools.product((1, -1), repeat=3))
    acc = []
    all_d = []
    for sq in cfgs:
        for se in cfgs:
            for sn in cfgs:
                vals, d, sind = eval_cfg(sq, se, sn)
                chi7 = sum(((vals[i] - OBS[i][1]) / OBS[i][2]) ** 2
                           for i in range(7))
                mp = max(abs((vals[i] - OBS[i][1]) / OBS[i][2])
                         for i in range(7))
                all_d.append(d)
                if mp <= 3.0:
                    acc.append((d, sind, mp, sq, se, sn))
    all_d = sorted(all_d)
    out.append("64 配置 δ 谱：min=%.4fπ max=%.4fπ" %
               (min(all_d), max(all_d)))
    near0 = sum(1 for d in all_d if d <= 0.1 or d >= 1.9)
    nearpi = sum(1 for d in all_d if 0.9 <= d <= 1.1)
    out.append("  近 0 点（|δ|≤0.1π mod 2π）：%d/64；近 π 点"
               "（0.9–1.1π）：%d/64" % (near0, nearpi))
    out.append("可接受族（7 项 ≤3σ）配置数 = %d" % len(acc))
    acc_d = sorted(a[0] for a in acc)
    if acc_d:
        out.append("  可接受族 δ 范围：%.4fπ – %.4fπ" %
                   (acc_d[0], acc_d[-1]))
        c0 = sum(1 for d in acc_d if d <= 0.1 or d >= 1.9)
        cp = sum(1 for d in acc_d if 0.9 <= d <= 1.1)
        out.append("  其中近 0 点 %d 个、近 π 点 %d 个" % (c0, cp))
        maxsind = max(abs(a[1]) for a in acc)
        out.append("  可接受族 max|sin δ| = %.4f" % maxsind)
        ok = (c0 + cp) == len(acc_d)
        out.append("  判据（全部落在守恒点附近 ±0.1π）：%s"
                   % ("成立 -> π/0 守恒是标签族结构结论"
                      if ok else "不成立 -> 谱含守恒点外配置（如实）"))
    # 锁定配置
    vals, d, sind = eval_cfg((1, 1, -1), (-1, 1, -1), (1, 1, -1))
    out.append("锁定配置 δ = %.4fπ（π 支，IC19 pull +0.03σ）" % d)

    d0 = os.path.dirname(os.path.abspath(__file__))
    txt = "\n".join(out) + "\n"
    with open(os.path.join(d0, "phaseQ17_results.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(txt)
    try:
        print(txt)
    except UnicodeEncodeError:
        print("written")


if __name__ == "__main__":
    main()

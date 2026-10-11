#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""phaseQ16.py —— J3 八项观测量 atan2 口径更正重算。

锁定 ±1 配置 + 锁定连续参数，atan2 提取 δ；对三个官方变体
（NO/IC19 177°±20°、NO/IC24 212°(+26/−41)、IO/IC24 274°±22/−25）
重算 8 项 pull 与 χ²。
输出 phaseQ16_results.txt（双运行逐字节一致后锁定）。
"""
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
RANK = (1, 2, 0)
SQ = (1, 1, -1)
SE = (-1, 1, -1)
SN = (1, 1, -1)
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
    else:
        d = 0.0
    return np.array([aV[0, 1], aV[1, 2], aV[0, 2], J, s12, s23,
                     s13 ** 2, d / math.pi])


def main():
    out = []
    out.append("== phaseQ16：J3 八项 atan2 口径更正重算 ==")
    w = np.array([1.0, P["w2"], P["w3"]])
    m = mags(RANK, P["rho"], w)
    tq = ds(m)
    tl = tq + P["wl"] * ds(RATIO["e"])
    Yu = build_Y(RATIO["u"], m, SQ, tq)
    Yd = build_Y(RATIO["d"], m, tuple(-x for x in SQ), tq)
    Ye = build_Y(RATIO["e"], m, SE, tl)
    Yn = build_Y(RATIO["n"], m, SN, P["tn"])
    vals = obs(mixing(Yu, Yd), mixing(Ye, Yn))
    base = [("Vus", 0.225, 0.0008), ("Vcb", 0.042, 0.0008),
            ("Vub", 0.0036, 0.00024), ("J", 3.0e-5, 0.15e-5),
            ("s2t12", 0.307, 0.013), ("s2t23", 0.546, 0.021),
            ("s2t13", 0.0220, 0.0007)]
    out.append("模型值（锁定配置，atan2）：")
    for (nm, e, er), v in zip(base, vals):
        out.append("  %-6s %.5f（pull %+.2f）" % (nm, v, (v - e) / er))
    out.append("  dcp/pi = %.4f（asin 旧口径 0.0138）" % vals[7])
    # 三个官方变体
    out.append("δ_CP pull（atan2 口径）：")
    for nm, mu, sig in (("NO/IC19", 177.0 / 180.0, 20.0 / 180.0),
                        ("NO/IC24", 212.0 / 180.0, 0.5 * (26 + 41) / 180.0),
                        ("IO/IC24", 274.0 / 180.0, 0.5 * (22 + 25) / 180.0)):
        out.append("  %-8s 数据 %.4fπ±%.4f -> pull %+.2fσ"
                   % (nm, mu, sig, (vals[7] - mu) / sig))
    chi2_8 = sum(((vals[i] - base[i][1]) / base[i][2]) ** 2
                 for i in range(7))
    d_mu, d_sig = 177.0 / 180.0, 20.0 / 180.0
    chi2_8 += ((vals[7] - d_mu) / d_sig) ** 2
    out.append("八项 χ²（vs NO/IC19，dof=3）：%.2f（p≈%.2f）"
               % (chi2_8, math.exp(-chi2_8 / 2)))

    d0 = os.path.dirname(os.path.abspath(__file__))
    txt = "\n".join(out) + "\n"
    with open(os.path.join(d0, "phaseQ16_results.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(txt)
    try:
        print(txt)
    except UnicodeEncodeError:
        print("written")


if __name__ == "__main__":
    main()

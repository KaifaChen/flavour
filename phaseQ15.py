#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""phaseQ15.py —— δ 提取口径更正检验（atan2 vs asin）。

T-E1：锁定 ±1 配置的真 δ（atan2）；
T-E2：连续标签 δ 网格（atan2 口径）的 π 可达性。
输出 phaseQ15_results.txt（双运行逐字节一致后锁定）。
"""
import math
import os

import numpy as np
from scipy.optimize import minimize

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
       ("s2t13", 0.0220, 0.0007), ("dcp_pi", 1.19, 0.22)]
RANK = (1, 2, 0)
SQ0 = np.array([1.0, 1.0, -1.0])
SE0 = np.array([-1.0, 1.0, -1.0])
SN0 = np.array([1.0, 1.0, -1.0])
P0 = np.array([1.199, 0.458, 0.058, 0.886, 0.899])


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
        sin_d = np.clip(JJ / den, -1, 1)
        cos_d = np.clip(Jr / den, -1, 1)
        d_atan2 = math.atan2(sin_d, cos_d)
        if d_atan2 < 0:
            d_atan2 += 2 * math.pi
        d_asin = math.asin(float(np.clip(JJ / den, -1, 1)))
    else:
        d_atan2 = d_asin = 0.0
    return np.array([aV[0, 1], aV[1, 2], aV[0, 2], J, s12, s23,
                     s13 ** 2, d_atan2 / math.pi]), d_asin / math.pi


def unpack(x):
    rho, w2, w3, wl, tn = np.abs(x[:5])
    sq = np.clip(x[5:8], -1, 1)
    se = np.clip(x[8:11], -1, 1)
    sn = np.clip(x[11:14], -1, 1)
    return rho, w2, w3, wl, tn, sq, se, sn


def evaluate(x):
    rho, w2, w3, wl, tn, sq, se, sn = unpack(x)
    w = np.array([1.0, w2, w3])
    m = mags(RANK, rho, w)
    tq = ds(m)
    tl = tq + wl * ds(RATIO["e"])
    Yu = build_Y(RATIO["u"], m, sq, tq)
    Yd = build_Y(RATIO["d"], m, -sq, tq)
    Ye = build_Y(RATIO["e"], m, se, tl)
    Yn = build_Y(RATIO["n"], m, sn, tn)
    v, dasin = obs(mixing(Yu, Yd), mixing(Ye, Yn))
    return v, dasin


def chi2_7(vals):
    idx = [i for i in range(8) if i != 7]
    return float(sum(((vals[i] - OBS[i][1]) / OBS[i][2]) ** 2
                     for i in idx))


def pulls(vals):
    return [(v - e) / er for v, (_, e, er) in zip(vals, OBS)]


def main():
    out = []
    out.append("== phaseQ15：δ 提取口径更正检验 ==")

    # T-E1 锁定配置
    x0 = np.concatenate([P0, SQ0, SE0, SN0])
    v, dasin = evaluate(x0)
    out.append("[T-E1] 锁定 ±1 配置：δ(atan2)=%.4fπ；δ(asin)=%.4fπ"
               % (v[7], dasin))
    out.append("  判据：%s" % ("δ≈π -> 5.35σ 张力为提取伪影（重大更正）"
                              if 0.9 <= v[7] <= 1.1 else
                              "δ≈0 -> 张力真实（sin 口径），继续 T-E2"))

    # T-E2 网格（atan2 口径）
    rng = np.random.default_rng(20261011)
    lam = 1000.0
    out.append("[T-E2] 连续标签 δ 网格（atan2 口径）：")
    rows = []
    for i in range(6, 11):
        target = i / 10.0

        def F(x):
            vv, _ = evaluate(x)
            return chi2_7(vv) + lam * (vv[7] - target) ** 2

        best = None
        for s in [x0] + [x0 + rng.normal(0, 0.5, 14) for _ in range(3)]:
            r = minimize(F, s, method="Nelder-Mead",
                         options={"maxiter": 9000, "xatol": 1e-8,
                                  "fatol": 1e-9})
            if best is None or r.fun < best.fun:
                best = r
        vv, _ = evaluate(best.x)
        pl = pulls(vv)
        rows.append((target, vv[7], chi2_7(vv),
                     max(abs(p) for p in pl[:7])))
        out.append("  δ/π=%.1f -> 落点 %.3f | chi2_7=%.3f | "
                   "max|pull|=%.2f" % (target, vv[7], chi2_7(vv),
                                       max(abs(p) for p in pl[:7])))
    near_pi = [r for r in rows if 0.9 <= r[1] <= 1.1 and r[3] <= 3.0]
    if near_pi:
        out.append("  判定：π 可达（atan2 口径）-> 0.5π 壁垒是 "
                   "asin 伪影；δ 张力可被连续标签消除（更正 v1.0）")
    else:
        out.append("  判定：π 仍不可达（atan2 口径）-> 壁垒真实"
                   "（角度结构排斥 π），v1.0 判定维持")

    d0 = os.path.dirname(os.path.abspath(__file__))
    txt = "\n".join(out) + "\n"
    with open(os.path.join(d0, "phaseQ15_results.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(txt)
    try:
        print(txt)
    except UnicodeEncodeError:
        print("written")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""phaseQ19.py —— ρ 不动点自洽检验（T-F1/T-F2）。

候选：rho = exp(ln3 - H(p(rho, w)))；p_ij = m_ij/Σm。
T-F1 锁定 w 迭代；T-F2 自洽拟合（4 参数）。
δ_CP 用 atan2 全圆提取。
输出 phaseQ19_results.txt（双运行逐字节一致后锁定）。
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
       ("s2t13", 0.0220, 0.0007)]
DCP_MU = 177.0 / 180.0
DCP_SIG = 20.0 / 180.0
RANK = (1, 2, 0)
SQ = (1, 1, -1)
SE = (-1, 1, -1)
SN = (1, 1, -1)
W0 = (1.0, 0.458, 0.058)
P0 = dict(wl=0.886, tn=0.899)


def H_of(p):
    p = np.asarray(p, float)
    p = p / p.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def mags(rho, w):
    return np.array([rho ** float(RANK[k]) * math.sqrt(w[i] * w[j])
                     for k, (i, j) in enumerate(PAIRS)])


def fixed_rho(w, tol=1e-12, itmax=500):
    rho = 1.2
    for _ in range(itmax):
        m = mags(rho, w)
        rho_new = math.exp(math.log(3.0) - H_of(m))
        if abs(rho_new - rho) < tol:
            return rho_new
        rho = rho_new
    return rho


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


def evaluate(x):
    w2, w3, wl, tn = np.abs(x)
    w = np.array([1.0, w2, w3])
    rho = fixed_rho(w)
    m = mags(rho, w)
    tq = math.log(3.0) - H_of(m)
    tl = tq + wl * (math.log(3.0) - H_of(RATIO["e"]))
    Yu = build_Y(RATIO["u"], m, SQ, tq)
    Yd = build_Y(RATIO["d"], m, tuple(-s for s in SQ), tq)
    Ye = build_Y(RATIO["e"], m, SE, tl)
    Yn = build_Y(RATIO["n"], m, SN, tn)
    return obs(mixing(Yu, Yd), mixing(Ye, Yn)), rho


def chi2_8(vals):
    c = sum(((vals[i] - OBS[i][1]) / OBS[i][2]) ** 2 for i in range(7))
    c += ((vals[7] - DCP_MU) / DCP_SIG) ** 2
    return c


def pulls(vals):
    p = [(vals[i] - OBS[i][1]) / OBS[i][2] for i in range(7)]
    p.append((vals[7] - DCP_MU) / DCP_SIG)
    return p


def main():
    out = []
    out.append("== phaseQ19：ρ 不动点自洽检验 ==")
    # T-F1
    rfp = fixed_rho(W0)
    out.append("[T-F1] 锁定 w 下不动点 rho=%.6f（锁定拟合 1.199，"
               "差 %.2f%%）" % (rfp, abs(rfp - 1.199) / 1.199 * 100))
    # T-F2
    x0 = np.array([0.458, 0.058, 0.886, 0.899])
    best = None
    for s in [x0, x0 * 1.05, np.array([0.45, 0.06, 0.90, 0.90])]:
        r = minimize(lambda x: chi2_8(evaluate(x)[0]), s,
                     method="Nelder-Mead",
                     options={"maxiter": 6000, "xatol": 1e-8,
                              "fatol": 1e-9})
        if best is None or r.fun < best.fun:
            best = r
    vals, rho_b = evaluate(best.x)
    c8 = chi2_8(vals)
    pl = pulls(vals)
    out.append("[T-F2] 自洽拟合（4 参数，rho=不动点）：")
    out.append("  w2=%.4f w3=%.4f wL=%.4f thnu=%.4f；rho=%.4f"
               % (np.abs(best.x[0]), np.abs(best.x[1]),
                  np.abs(best.x[2]), np.abs(best.x[3]), rho_b))
    out.append("  chi2_8=%.4f（锁定参考 2.31；判据 <=3.31）" % c8)
    out.append("  八项 pull：%s" % ", ".join("%.2f" % p for p in pl))
    ok = c8 <= 2.31 + 1.0
    out.append("  判据：%s" %
               ("通过 -> ρ 被 ΔS 自洽导出，模型 5 参数降为 4"
                if ok else "失败 -> 不动点口径与锁定拟合不兼容"))

    d0 = os.path.dirname(os.path.abspath(__file__))
    txt = "\n".join(out) + "\n"
    with open(os.path.join(d0, "phaseQ19_results.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(txt)
    try:
        print(txt)
    except UnicodeEncodeError:
        print("written")


if __name__ == "__main__":
    main()

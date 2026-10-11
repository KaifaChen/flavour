#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""phaseQ20.py —— 剩余槽位三测（T-G1 w_L / T-G2 θ_ν / T-G3 w2=8w3）。

ρ 不动点 + atan2 口径；基准 χ²₈=2.2277。
输出 phaseQ20_results.txt（双运行逐字节一致后锁定）。
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
DSM_E = math.log(3.0) + sum(
    p * math.log(p)
    for p in np.asarray(RATIO["e"], float) / RATIO["e"].sum()
    if p > 0)  # = ln3 - H(归一化质量比 e)，修正：必须先归一化
CHI_BASE = 2.2277


def H_of(p):
    p = np.asarray(p, float)
    p = p / p.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def dS_mass(r):
    p = np.asarray(r, float)
    p = p / p.sum()
    p = p[p > 0]
    return math.log(3.0) - float(-(p * np.log(p)).sum())


def mags(rho, w):
    return np.array([rho ** float(RANK[k]) * math.sqrt(w[i] * w[j])
                     for k, (i, j) in enumerate(PAIRS)])


def fixed_rho(w, tol=1e-12):
    rho = 1.2
    for _ in range(500):
        m = mags(rho, w)
        rn = math.exp(math.log(3.0) - H_of(m))
        if abs(rn - rho) < tol:
            return rn
        rho = rn
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


def evaluate(w2, w3, wl, tn):
    w = np.array([1.0, w2, w3])
    rho = fixed_rho(w)
    m = mags(rho, w)
    tq = math.log(3.0) - H_of(m)
    tl = tq + wl * DSM_E
    Yu = build_Y(RATIO["u"], m, SQ, tq)
    Yd = build_Y(RATIO["d"], m, tuple(-s for s in SQ), tq)
    Ye = build_Y(RATIO["e"], m, SE, tl)
    Yn = build_Y(RATIO["n"], m, SN, tn)
    return obs(mixing(Yu, Yd), mixing(Ye, Yn)), rho


def chi2_8(vals):
    c = sum(((vals[i] - OBS[i][1]) / OBS[i][2]) ** 2 for i in range(7))
    c += ((vals[7] - DCP_MU) / DCP_SIG) ** 2
    return c


def fit(fun, x0s):
    best = None
    for s in x0s:
        r = minimize(fun, s, method="Nelder-Mead",
                     options={"maxiter": 6000, "xatol": 1e-8,
                              "fatol": 1e-9})
        if best is None or r.fun < best.fun:
            best = r
    return best


def main():
    out = []
    out.append("== phaseQ20：剩余槽位三测（基准 χ²₈=%.4f）==" % CHI_BASE)
    dsm_n = dS_mass(RATIO["n"])
    w = np.array([1.0, 0.452732, 0.057640])
    rho = fixed_rho(w)
    m = mags(rho, w)
    hpi = H_of(m)
    out.append("候选常数：ΔS_mass(e)=%.4f、H(pi)=%.4f、ΔS_mass(nu)=%.4f"
               % (DSM_E, hpi, dsm_n))

    # T-G1 w_L = DSM_E
    def f1(x):
        w2, w3, tn = np.abs(x)
        return chi2_8(evaluate(w2, w3, DSM_E, tn)[0])
    r1 = fit(f1, [np.array([0.4527, 0.0576, 0.8991]),
                  np.array([0.45, 0.06, 0.90])])
    ok1 = r1.fun <= CHI_BASE + 1.0
    out.append("[T-G1] w_L=ΔS_mass(e) 固定：chi2_8=%.4f -> %s"
               % (r1.fun, "通过（w_L 导出，3 参数）" if ok1 else "失败"))

    # T-G2 θ_ν 候选
    for nm, tv in (("ΔS_mass(e)", DSM_E), ("H(pi)", hpi),
                   ("ΔS_mass(nu)", dsm_n)):
        def f2(x, tv=tv):
            w2, w3, wl = np.abs(x)
            return chi2_8(evaluate(w2, w3, wl, tv)[0])
        r2 = fit(f2, [np.array([0.4527, 0.0576, 0.8852]),
                      np.array([0.45, 0.06, 0.89])])
        ok2 = r2.fun <= CHI_BASE + 1.0
        out.append("[T-G2] θ_ν=%-12s 固定：chi2_8=%.4f -> %s"
                   % (nm, r2.fun, "通过（θ_ν 导出）" if ok2 else "失败"))

    # T-G3 w2 = 8 w3
    def f3(x):
        w3, wl, tn = np.abs(x)
        return chi2_8(evaluate(8.0 * w3, w3, wl, tn)[0])
    r3 = fit(f3, [np.array([0.0576, 0.8852, 0.8991]),
                  np.array([0.06, 0.89, 0.90])])
    ok3 = r3.fun <= CHI_BASE + 1.0
    out.append("[T-G3] w2=8w3 固定：chi2_8=%.4f -> %s"
               % (r3.fun, "通过（宽度比锁定，3 参数）" if ok3 else "失败"))

    d0 = os.path.dirname(os.path.abspath(__file__))
    txt = "\n".join(out) + "\n"
    with open(os.path.join(d0, "phaseQ20_results.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(txt)
    try:
        print(txt)
    except UnicodeEncodeError:
        print("written")


if __name__ == "__main__":
    main()

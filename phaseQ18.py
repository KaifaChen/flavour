#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""phaseQ18.py —— I 轴候选关系二轮（R6-R11）。

格量：H(p)（宽度归一化边际）、H(p|τ)（代←字母后验）、
H(π)（归一化基础耦合）、I(p;τ)（代→字母互信息）。
输出 phaseQ18_results.txt（双运行逐字节一致后锁定）。
"""
import math
import os

W = (1.0, 0.458, 0.058)
TH_REL = 0.184
THL_M_THR = 0.780
WL_LO, WL_HI = 0.8824, 0.9569
DSM = 0.880
THN = 0.899
THN_LO, THN_HI = 0.8977, 0.8994
BAND_LO, BAND_HI = WL_LO * DSM, WL_HI * DSM


def H_of(p):
    p = [x for x in p if x > 0]
    return -sum(x * math.log(x) for x in p)


def lattice(w):
    """返回 (H(p), I, H(p|tau), H(pi))。"""
    wsum = sum(w)
    pi = [x / wsum for x in w]
    # 发射 S1={a}, S2={a,b,c}, S3={b,c} 均匀
    H_cond = pi[1] * math.log(3.0) + pi[2] * math.log(2.0)
    Pa = pi[0] + pi[1] / 3.0
    Pb = pi[1] / 3.0 + pi[2] / 2.0
    Pc = Pb
    H_tau = H_of([Pa, Pb, Pc])
    I = H_tau - H_cond
    Hp = H_of(pi)
    Hpt = Hp - I
    base = [math.sqrt(w[i] * w[j]) for i, j in ((0, 1), (1, 2), (0, 2))]
    Z = sum(base)
    Hpi = H_of([b / Z for b in base])
    return Hp, I, Hpt, Hpi


def main():
    out = []
    out.append("== phaseQ18：I 轴候选关系二轮 ==")
    Hp, I, Hpt, Hpi = lattice(W)
    out.append("格量（锁定 w）：H(p)=%.4f  I=%.4f  H(p|tau)=%.4f  "
               "H(pi)=%.4f" % (Hp, I, Hpt, Hpi))
    out.append("对照带：theta_L-theta_rel=%.4f（1σ [%.4f,%.4f]）；"
               "w_L 1σ [%.4f,%.4f]；theta_nu=%.4f（1σ [%.4f,%.4f]）"
               % (THL_M_THR, BAND_LO, BAND_HI, WL_LO, WL_HI,
                  THN, THN_LO, THN_HI))
    tests = [
        ("R6", "theta_L-theta_rel =? H(p)", Hp, THL_M_THR,
         BAND_LO <= Hp <= BAND_HI),
        ("R7", "theta_L-theta_rel =? I+H(p|tau)(=H(p))", Hp,
         THL_M_THR, BAND_LO <= Hp <= BAND_HI),
        ("R8", "w_L =? H(p)/ΔS_mass", Hp / DSM, 0.886,
         WL_LO <= Hp / DSM <= WL_HI),
        ("R9", "theta_nu =? H(p)", Hp, THN, THN_LO <= Hp <= THN_HI),
        ("R10", "theta_nu =? H(pi)", Hpi, THN, THN_LO <= Hpi <= THN_HI),
        ("R11", "theta_rel =? I", I, TH_REL,
         abs(I - TH_REL) / TH_REL <= 0.01),
    ]
    n_pass = 0
    for rid, name, val, ref, ok in tests:
        out.append("  %-4s %-32s 计算 %.4f vs 对照 %.4f -> %s"
                   % (rid, name, val, ref, "通过" if ok else "失败"))
        n_pass += 1 if ok else 0
    out.append("判据：%d/6 通过 -> %s" %
               (n_pass,
                "I 格量进入味结构（新通道候选）" if n_pass else
                "二轮负结果：I 及其格量与味结构锁定值无自然数值关系；"
                "I 的价值在框架层而非味域数值"))

    d0 = os.path.dirname(os.path.abspath(__file__))
    txt = "\n".join(out) + "\n"
    with open(os.path.join(d0, "phaseQ18_results.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(txt)
    try:
        print(txt)
    except UnicodeEncodeError:
        print("written")


if __name__ == "__main__":
    main()

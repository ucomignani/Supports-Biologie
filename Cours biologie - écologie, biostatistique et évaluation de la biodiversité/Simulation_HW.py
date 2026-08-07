#!/usr/bin/env python3
"""
Hardy-Weinberg : explorateur interactif.

Deux panneaux, pilotes par les sliders :

  - En haut, la STRUCTURE GENOTYPIQUE a la frequence allelique choisie, dans une
    population de taille N et avec un coefficient de consanguinite F. On compare
    les frequences observees aux proportions attendues sous Hardy-Weinberg, et le
    test du khi-deux (a 1 degre de liberte) indique si l'ecart est significatif.

  - En bas, la DYNAMIQUE de la frequence allelique au fil des generations, sous
    l'effet de la selection (coefficient s, dominance h) et de la derive genetique
    (liee a la taille N). Plusieurs trajectoires illustrent le hasard de la derive ;
    la courbe rouge est la trajectoire sans derive (population infinie).

Lancement :
    python3 Simulation_HW.py

Dependances : numpy, matplotlib.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

BLEU = "#4682b4"
ORANGE = "#e07b39"
ROUGE = "#b52b2b"

CHI2_SEUIL_5PCT_DDL1 = 3.8415   # quantile a 95 % de la loi du khi-deux a 1 ddl


# ----------------------------------------------------------------------------
# Coeur de calcul
# ----------------------------------------------------------------------------
def simule_population(p, F, N, rng):
    """Genere N individus a la frequence allelique p, avec consanguinite F.
    Renvoie les effectifs (nAA, nAa, naa)."""
    N = int(N)
    ibd = rng.random(N) < F                       # deux alleles identiques par ascendance
    a1 = rng.random(N) < p                         # True = allele A
    a2 = np.where(ibd, a1, rng.random(N) < p)      # copie du premier, ou tire au hasard
    nA = a1.astype(int) + a2.astype(int)           # 0 = aa, 1 = Aa, 2 = AA
    return int(np.sum(nA == 2)), int(np.sum(nA == 1)), int(np.sum(nA == 0))


def test_hwe(nAA, nAa, naa):
    """Test du khi-deux de conformite a Hardy-Weinberg (frequence allelique estimee).
    Renvoie (phat, statistique, p_value, effectif_attendu_min). ddl = 1.
    p_value calculee sans scipy : P(chi2_1 > x) = erfc(sqrt(x/2))."""
    N = nAA + nAa + naa
    phat = (2 * nAA + nAa) / (2 * N)
    qhat = 1 - phat
    att = np.array([N * phat**2, N * 2 * phat * qhat, N * qhat**2])
    obs = np.array([nAA, nAa, naa], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        termes = np.where(att > 0, (obs - att) ** 2 / att, 0.0)
    stat = float(termes.sum())
    p_value = math.erfc(math.sqrt(stat / 2.0)) if stat > 0 else 1.0
    return phat, stat, p_value, float(att.min())


def trajectoires(p0, N, s, h, n_gen, n_traj, rng):
    """Modele de Wright-Fisher avec selection : fitness (AA, Aa, aa) = (1, 1-hs, 1-s).
    Renvoie un tableau (n_gen+1, n_traj) de frequences alleliques de A."""
    N = int(N); n_gen = int(n_gen)
    w_AA, w_Aa, w_aa = 1.0, 1.0 - h * s, 1.0 - s
    P = np.full(n_traj, float(p0))
    hist = np.empty((n_gen + 1, n_traj))
    hist[0] = P
    for g in range(1, n_gen + 1):
        Q = 1 - P
        num = P * P * w_AA + P * Q * w_Aa                       # freq de A apres selection
        wbar = P * P * w_AA + 2 * P * Q * w_Aa + Q * Q * w_aa
        p_sel = np.where(wbar > 0, num / wbar, P)
        p_sel = np.clip(p_sel, 0.0, 1.0)
        P = rng.binomial(2 * N, p_sel) / (2 * N)               # derive genetique
        hist[g] = P
    return hist


def trajectoire_deterministe(p0, s, h, n_gen):
    """Meme modele de selection, sans derive (population infinie)."""
    w_AA, w_Aa, w_aa = 1.0, 1.0 - h * s, 1.0 - s
    p = float(p0); out = [p]
    for _ in range(int(n_gen)):
        q = 1 - p
        num = p * p * w_AA + p * q * w_Aa
        wbar = p * p * w_AA + 2 * p * q * w_Aa + q * q * w_aa
        p = num / wbar if wbar > 0 else p
        out.append(min(max(p, 0.0), 1.0))
    return np.array(out)


DEFAUTS = dict(p=0.5, F=0.0, N=300, s=0.0, h=0.5, n_gen=150)
N_TRAJ = 15


def lancer_gui():
    seed = {"val": 20260}

    fig = plt.figure(figsize=(9.5, 9.0))
    ax_bar = fig.add_axes([0.11, 0.735, 0.85, 0.20])    # panneau haut : genotypes
    ax_traj = fig.add_axes([0.11, 0.525, 0.85, 0.155])  # panneau bas  : dynamique

    ax_color = "#eef2f7"
    specs = [
        ("p",     "frequence de A  (p)",          0.01, 0.99, 0.01, DEFAUTS["p"]),
        ("F",     "consanguinite  (F)",           0.0,  0.90, 0.05, DEFAUTS["F"]),
        ("N",     "taille de population  (N)",    20,   2000, 10,   DEFAUTS["N"]),
        ("s",     "selection contre aa  (s)",    -0.30, 0.50, 0.02, DEFAUTS["s"]),
        ("h",     "dominance  (h)",               0.0,  1.00, 0.10, DEFAUTS["h"]),
        ("n_gen", "generations",                  20,   400,  10,   DEFAUTS["n_gen"]),
    ]
    sliders = {}
    y = 0.44
    for key, label, lo, hi, step, init in specs:
        axs = fig.add_axes([0.13, y, 0.72, 0.024], facecolor=ax_color)
        sliders[key] = Slider(axs, label, lo, hi, valinit=init, valstep=step)
        y -= 0.048

    ax_re = fig.add_axes([0.30, 0.095, 0.20, 0.042])
    ax_reset = fig.add_axes([0.53, 0.095, 0.18, 0.042])
    bouton_re = Button(ax_re, "Reechantillonner", color="#dbe7f3", hovercolor="#c2d6ec")
    bouton_reset = Button(ax_reset, "Reinitialiser", color="#f3dbdb", hovercolor="#ecc2c2")

    def dessiner(event=None):
        p = float(sliders["p"].val)
        F = float(sliders["F"].val)
        N = int(sliders["N"].val)
        s = float(sliders["s"].val)
        h = float(sliders["h"].val)
        n_gen = int(sliders["n_gen"].val)
        rng = np.random.default_rng(seed["val"])

        # ---- panneau haut : structure genotypique + test ------------------
        nAA, nAa, naa = simule_population(p, F, N, rng)
        obs = np.array([nAA, nAa, naa]) / N
        phat, stat, pval, att_min = test_hwe(nAA, nAa, naa)
        qhat = 1 - phat
        hw = np.array([phat**2, 2 * phat * qhat, qhat**2])

        ax_bar.clear()
        xx = np.arange(3); wbar = 0.38
        ax_bar.bar(xx - wbar / 2, obs, wbar, color=BLEU, label="observe")
        ax_bar.bar(xx + wbar / 2, hw, wbar, color=ORANGE, label="attendu (Hardy-Weinberg)")
        ax_bar.set_xticks(xx); ax_bar.set_xticklabels(["AA", "Aa", "aa"])
        ax_bar.set_ylim(0, max(0.6, obs.max(), hw.max()) * 1.18)
        ax_bar.set_ylabel("frequence")
        ax_bar.set_title("Structure genotypique a la generation courante", fontsize=12)
        ax_bar.legend(loc="upper right", fontsize=9, framealpha=0.9)

        decision = ("ecart significatif (seuil 5 %)" if stat > CHI2_SEUIL_5PCT_DDL1
                    else "compatible avec Hardy-Weinberg")
        note_eff = "   [effectifs faibles : test approximatif]" if att_min < 5 else ""
        info = (
            f"p estime = {phat:.3f}\n"
            f"khi-deux = {stat:.2f}   (ddl = 1)\n"
            f"p-value  = {pval:.3g}{note_eff}\n"
            f"=> {decision}"
        )
        ax_bar.text(0.015, 0.97, info, transform=ax_bar.transAxes, va="top", ha="left",
                    fontsize=9.5, family="monospace",
                    bbox=dict(boxstyle="round", facecolor="#fbfbe8", edgecolor="#cccccc"))

        # ---- panneau bas : dynamique allelique ----------------------------
        hist = trajectoires(p, N, s, h, n_gen, N_TRAJ, rng)
        det = trajectoire_deterministe(p, s, h, n_gen)
        gens = np.arange(n_gen + 1)

        ax_traj.clear()
        ax_traj.plot(gens, hist, color=BLEU, alpha=0.30, lw=0.9)
        ax_traj.plot(gens, det, color=ROUGE, lw=2.2, label="sans derive (N infini)")
        ax_traj.axhline(0, color="#aaaaaa", lw=0.6)
        ax_traj.axhline(1, color="#aaaaaa", lw=0.6)
        ax_traj.set_ylim(-0.03, 1.03)
        ax_traj.set_xlim(0, n_gen)
        ax_traj.set_xlabel("generation")
        ax_traj.set_ylabel("frequence de A")
        ax_traj.set_title(
            f"Dynamique allelique : {N_TRAJ} populations (derive) sous selection s = {s:+.2f}",
            fontsize=11)
        ax_traj.legend(loc="center right", fontsize=9, framealpha=0.9)

        fig.canvas.draw_idle()

    for sl in sliders.values():
        sl.on_changed(dessiner)

    def reechantillonner(event):
        seed["val"] += 1
        dessiner()

    def reinitialiser(event):
        for sl in sliders.values():
            sl.eventson = False
            sl.reset()
            sl.eventson = True
        dessiner()

    bouton_re.on_clicked(reechantillonner)
    bouton_reset.on_clicked(reinitialiser)

    dessiner()
    plt.show()


if __name__ == "__main__":
    lancer_gui()

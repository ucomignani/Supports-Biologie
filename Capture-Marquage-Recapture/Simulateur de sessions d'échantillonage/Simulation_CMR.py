#!/usr/bin/env python3
"""
Capture-marquage-recapture : explorateur interactif.

Ajustez les parametres avec les sliders et observez, en direct, la distribution
d'echantillonnage de l'estimateur de Chapman : ou elle se centre par rapport a
l'effectif reel, dans quel sens elle se decale quand une hypothese est violee,
et la part d'etudes ou aucun individu marque n'est recapture (R = 0).

Lancement :
    python3 cmr_gui.py

Dependances : numpy, matplo tlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# ----------------------------------------------------------------------------
# macros de legende
# ----------------------------------------------------------------------------
legN =  "N (effectif réel)"

# ----------------------------------------------------------------------------
# Coeur de simulation (vectorise : n_rep etudes calculees d'un coup)
# ----------------------------------------------------------------------------
def simule(N, p, het, reponse, perte, survie, n_rep, rng):
    """Simule n_rep etudes de capture-marquage-recapture a deux occasions.

    Renvoie trois tableaux de longueur n_rep :
      M : nombre de marques relaches a l'occasion 1
      C : taille de la seconde capture
      R : nombre de recaptures portant une marque lisible
    """
    N = int(N)

    # Probabilite de capture propre a chaque individu ------------------------
    if het > 0:
        # tirage dans une loi Beta de moyenne p ; concentration decroissante
        conc = (1.0 - het) / het
        p_i = rng.beta(p * conc, (1.0 - p) * conc, size=(n_rep, N))
    else:
        p_i = np.full((n_rep, N), p)

    # Occasion 1 : capture puis marquage -------------------------------------
    marque = rng.random((n_rep, N)) < p_i
    M = marque.sum(axis=1)

    # Entre les deux occasions -----------------------------------------------
    surv_draw = rng.random((n_rep, N)) < survie
    present = np.where(marque, surv_draw, True)        # les non-marques restent presents
    perdu = rng.random((n_rep, N)) < perte
    marque_visible = marque & present & ~perdu

    # Occasion 2 : capture (les marques peuvent changer de comportement) ------
    p2 = np.where(marque, np.minimum(p_i * reponse, 1.0), p_i)
    cap2 = (rng.random((n_rep, N)) < p2) & present

    C = cap2.sum(axis=1)
    R = (cap2 & marque_visible).sum(axis=1)
    return M, C, R


def estimateurs(M, C, R):
    """Estimateurs de Lincoln-Petersen (infini si R = 0) et de Chapman (toujours fini)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        LP = M * C / R
    chapman = (M + 1) * (C + 1) / (R + 1) - 1
    return LP, chapman


# ----------------------------------------------------------------------------
# Parametres par defaut
# ----------------------------------------------------------------------------
DEFAUTS = dict(N=500, p=0.35, het=0.0, reponse=1.0, perte=0.0, survie=1.0, n_rep=1000)


def lancer_gui():
    seed = {"val": 12345}  # graine courante (le bouton "Reechantillonner" la change)

    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    plt.subplots_adjust(left=0.11, right=0.97, top=0.92, bottom=0.50)

    # --- definition des sliders --------------------------------------------
    ax_color = "#eef2f7"
    specs = [
        ("N",       legN,          50,   2000, 10,   DEFAUTS["N"]),
        ("p",       "p  (proba de capture)",        0.02, 0.90, 0.01, DEFAUTS["p"]),
        ("het",     "heterogeneite",                0.0,  0.90, 0.05, DEFAUTS["het"]),
        ("reponse", "reponse au piege (1 = neutre)",0.3,  2.5,  0.1,  DEFAUTS["reponse"]),
        ("perte",   "perte de marque",              0.0,  0.60, 0.02, DEFAUTS["perte"]),
        ("survie",  "survie des marques",           0.30, 1.00, 0.02, DEFAUTS["survie"]),
        ("n_rep",   "nombre d'etudes simulees",     200,  3000, 100,  DEFAUTS["n_rep"]),
    ]

    sliders = {}
    y = 0.44
    for key, label, lo, hi, step, init in specs:
        axs = fig.add_axes([0.13, y, 0.74, 0.025], facecolor=ax_color)
        sliders[key] = Slider(axs, label, lo, hi, valinit=init, valstep=step)
        y -= 0.045

    # --- boutons ------------------------------------------------------------
    ax_re = fig.add_axes([0.30, 0.045, 0.20, 0.045])
    ax_reset = fig.add_axes([0.53, 0.045, 0.18, 0.045])
    bouton_re = Button(ax_re, "Reechantillonner", color="#dbe7f3", hovercolor="#c2d6ec")
    bouton_reset = Button(ax_reset, "Reinitialiser", color="#f3dbdb", hovercolor="#ecc2c2")

    # --- fonction de dessin -------------------------------------------------
    def dessiner(event=None):
        N = int(sliders["N"].val)
        p = float(sliders["p"].val)
        het = float(sliders["het"].val)
        reponse = float(sliders["reponse"].val)
        perte = float(sliders["perte"].val)
        survie = float(sliders["survie"].val)
        n_rep = int(sliders["n_rep"].val)

        rng = np.random.default_rng(seed["val"])
        M, C, R = simule(N, p, het, reponse, perte, survie, n_rep, rng)
        LP, chapman = estimateurs(M, C, R)

        part_R0 = float(np.mean(R == 0))
        moy_chap = float(np.mean(chapman))
        biais_rel = moy_chap / N - 1.0
        lp_moy = np.mean(LP)
        lp_fini = LP[np.isfinite(LP)]

        ax.clear()
        borne_haute = max(1.6 * N, np.percentile(chapman, 98))
        ax.hist(chapman, bins=45, range=(0, borne_haute),
                color="#4682b4", edgecolor="white", alpha=0.85)
        ax.axvline(N, color="firebrick", lw=2.2, label= legN +f" = {N}")
        ax.axvline(moy_chap, color="#0b2545", ls="--", lw=1.8,
                   label=f"moyenne Chapman = {moy_chap:.0f}")
        ax.set_xlim(0, borne_haute)
        ax.set_xlabel("effectif estime (Chapman)")
        ax.set_ylabel("nombre d'etudes")
        ax.set_title("Distribution d'echantillonnage de l'estimateur", fontsize=12)
        ax.legend(loc="upper right", framealpha=0.9)

        if np.isfinite(lp_moy):
            lp_txt = f"moyenne Lincoln-Petersen = {lp_moy:.0f}"
        else:
            med = np.median(lp_fini) if lp_fini.size else float("nan")
            lp_txt = f"moyenne Lincoln-Petersen = infinie  (mediane des cas finis = {med:.0f})"

        sens = "surestimation" if biais_rel > 0 else "sous-estimation"
        info = (
            f"biais relatif de Chapman : {biais_rel * 100:+.1f} %  ({sens})\n"
            f"part d'etudes ou R = 0 : {part_R0 * 100:.1f} %\n"
            f"{lp_txt}"
        )
        ax.text(0.02, 0.97, info, transform=ax.transAxes, va="top", ha="left",
                fontsize=10, family="monospace",
                bbox=dict(boxstyle="round", facecolor="#fbfbe8", edgecolor="#cccccc"))
        fig.canvas.draw_idle()

    # --- connexions ---------------------------------------------------------
    for s in sliders.values():
        s.on_changed(dessiner)

    def reechantillonner(event):
        seed["val"] += 1
        dessiner()

    def reinitialiser(event):
        for s in sliders.values():
            s.eventson = False
            s.reset()
            s.eventson = True
        dessiner()

    bouton_re.on_clicked(reechantillonner)
    bouton_reset.on_clicked(reinitialiser)

    dessiner()
    plt.show()


if __name__ == "__main__":
    lancer_gui()

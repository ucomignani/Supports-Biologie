#!/usr/bin/env python3
"""
Capture-marquage-recapture : explorateur interactif.

Ajustez les parametres avec les sliders et observez, en direct :
  - la distribution d'echantillonnage de l'estimateur de Chapman (panneau haut),
    ou elle se centre par rapport a l'effectif reel, et dans quel sens elle se
    decale quand une hypothese est violee ;
  - un echantillon d'intervalles de confiance individuels (panneau bas), colores
    selon qu'ils contiennent ou non le vrai N, avec le taux de couverture reel.

Lancement :
    python3 Simulation_CMR.py

Dependances : numpy, matplotlib.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# ----------------------------------------------------------------------------
# macros de legende
# ----------------------------------------------------------------------------
legN =  "N (effectif reel)"


# ----------------------------------------------------------------------------
# Quantile de la loi normale (sans dependance a scipy)
# Approximation d'Acklam de l'inverse de la fonction de repartition normale.
# ----------------------------------------------------------------------------
def _norm_ppf(p):
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


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


def var_chapman(M, C, R):
    """Variance estimee de l'estimateur de Chapman (formule de Seber).

    Toujours positive ou nulle, car R <= min(M, C) donc (M-R) et (C-R) >= 0.
    """
    return (M + 1) * (C + 1) * (M - R) * (C - R) / ((R + 1)**2 * (R + 2))


# ----------------------------------------------------------------------------
# Parametres par defaut
# ----------------------------------------------------------------------------
DEFAUTS = dict(N=500, p=0.35, het=0.0, reponse=1.0, perte=0.0,
               survie=1.0, n_rep=1000, niveau=0.95)


def lancer_gui():
    seed = {"val": 12345}  # graine courante (le bouton "Reechantillonner" la change)

    fig = plt.figure(figsize=(9.5, 9.2))
    ax_hist = fig.add_axes([0.11, 0.72, 0.86, 0.21])   # panneau haut : distribution
    ax_ci = fig.add_axes([0.11, 0.52, 0.86, 0.15])     # panneau bas  : intervalles

    # --- definition des sliders --------------------------------------------
    ax_color = "#eef2f7"
    specs = [
        ("N",       legN,                            50,   2000, 10,   DEFAUTS["N"]),
        ("p",       "p  (proba de capture)",         0.02, 0.90, 0.01, DEFAUTS["p"]),
        ("het",     "heterogeneite",                 0.0,  0.90, 0.05, DEFAUTS["het"]),
        ("reponse", "reponse au piege (1 = neutre)", 0.3,  2.5,  0.1,  DEFAUTS["reponse"]),
        ("perte",   "perte de marque",               0.0,  0.60, 0.02, DEFAUTS["perte"]),
        ("survie",  "survie des marques",            0.30, 1.00, 0.02, DEFAUTS["survie"]),
        ("n_rep",   "nombre d'etudes simulees",      200,  3000, 100,  DEFAUTS["n_rep"]),
        ("niveau",  "niveau de confiance",           0.80, 0.99, 0.01, DEFAUTS["niveau"]),
    ]

    sliders = {}
    y = 0.45
    for key, label, lo, hi, step, init in specs:
        axs = fig.add_axes([0.13, y, 0.74, 0.025], facecolor=ax_color)
        sliders[key] = Slider(axs, label, lo, hi, valinit=init, valstep=step)
        y -= 0.045

    # --- boutons ------------------------------------------------------------
    ax_re = fig.add_axes([0.30, 0.05, 0.20, 0.045])
    ax_reset = fig.add_axes([0.53, 0.05, 0.18, 0.045])
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
        niveau = float(sliders["niveau"].val)

        rng = np.random.default_rng(seed["val"])
        M, C, R = simule(N, p, het, reponse, perte, survie, n_rep, rng)
        LP, chapman = estimateurs(M, C, R)

        # Intervalles de confiance de Chapman (approximation normale)
        z = _norm_ppf(0.5 + niveau / 2.0)
        se = np.sqrt(np.maximum(var_chapman(M, C, R), 0.0))
        bas = chapman - z * se
        haut = chapman + z * se
        couvre = (bas <= N) & (N <= haut)
        couverture = float(np.mean(couvre))

        part_R0 = float(np.mean(R == 0))
        moy_chap = float(np.mean(chapman))
        biais_rel = moy_chap / N - 1.0
        lp_moy = np.mean(LP)
        lp_fini = LP[np.isfinite(LP)]

        borne = max(1.6 * N, float(np.percentile(haut, 92)))

        # --- panneau haut : histogramme ------------------------------------
        ax_hist.clear()
        ax_hist.hist(chapman, bins=45, range=(0, borne),
                     color="#4682b4", edgecolor="white", alpha=0.85)
        ax_hist.axvline(N, color="firebrick", lw=2.2, label=legN + f" = {N}")
        ax_hist.axvline(moy_chap, color="#0b2545", ls="--", lw=1.8,
                        label=f"moyenne Chapman = {moy_chap:.0f}")
        ax_hist.set_xlim(0, borne)
        ax_hist.set_ylabel("nombre d'etudes")
        ax_hist.set_title("Distribution d'echantillonnage de l'estimateur", fontsize=12)
        ax_hist.legend(loc="upper right", framealpha=0.9)
        ax_hist.tick_params(labelbottom=False)

        if np.isfinite(lp_moy):
            lp_txt = f"moyenne Lincoln-Petersen = {lp_moy:.0f}"
        else:
            med = np.median(lp_fini) if lp_fini.size else float("nan")
            lp_txt = f"moyenne Lincoln-Petersen = infinie  (mediane des cas finis = {med:.0f})"

        sens = "surestimation" if biais_rel > 0 else "sous-estimation"
        info = (
            f"biais relatif de Chapman : {biais_rel * 100:+.1f} %  ({sens})\n"
            f"couverture empirique : {couverture * 100:.1f} %   (cible {niveau * 100:.0f} %)\n"
            f"part d'etudes ou R = 0 : {part_R0 * 100:.1f} %\n"
            f"{lp_txt}"
        )
        ax_hist.text(0.02, 0.97, info, transform=ax_hist.transAxes, va="top", ha="left",
                     fontsize=10, family="monospace",
                     bbox=dict(boxstyle="round", facecolor="#fbfbe8", edgecolor="#cccccc"))

        # --- panneau bas : intervalles de confiance individuels ------------
        ax_ci.clear()
        K = min(45, n_rep)                       # nombre d'etudes affichees
        order = np.argsort(chapman[:K])          # tri par estimation, pour la lisibilite
        yy = np.arange(K)
        ch_s = chapman[:K][order]
        bas_s = bas[:K][order]
        haut_s = haut[:K][order]
        cov_s = couvre[:K][order]
        couleurs = np.where(cov_s, "#2e7d32", "#c62828")   # vert = contient N, rouge = manque
        ax_ci.hlines(yy, bas_s, haut_s, color=couleurs, lw=1.6, alpha=0.85)
        ax_ci.plot(ch_s, yy, "o", ms=2.6, color="#222222")
        ax_ci.axvline(N, color="firebrick", lw=2)
        ax_ci.set_xlim(0, borne)
        ax_ci.set_ylim(-1, K)
        ax_ci.set_yticks([])
        ax_ci.set_xlabel("effectif estime (Chapman) et intervalles de confiance")
        ax_ci.set_title(
            f"{K} etudes : intervalles a {niveau * 100:.0f} %  "
            f"(vert = contient N, rouge = le manque)", fontsize=11)

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

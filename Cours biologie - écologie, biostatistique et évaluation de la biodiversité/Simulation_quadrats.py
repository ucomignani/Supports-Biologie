#!/usr/bin/env python3
"""
Methode des quadrats : explorateur interactif.

  - A gauche, une CARTE du terrain : les individus sont places selon un patron
    spatial reglable (regulier, aleatoire ou agrege), et des quadrats sont tires
    au hasard par-dessus. La couleur d'un quadrat indique son nombre d'individus.

  - A droite, la DISTRIBUTION des comptages par quadrat, comparee a la loi de
    Poisson (repartition aleatoire) de meme moyenne. Un encadre donne le rapport
    variance/moyenne, l'indice de dispersion et le test associe (khi-deux a n-1
    degres de liberte), ainsi que la densite estimee et son intervalle de confiance.

Lancement :
    python3 Simulation_quadrats.py

Dependances : numpy, matplotlib.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, Button

W = H = 10.0            # dimensions du terrain (unites)
AIRE = W * H
VERT = "#2e7d32"
BLEU = "#4682b4"
ROUGE = "#b52b2b"


# ----------------------------------------------------------------------------
# Quantiles de la loi normale et du khi-deux (sans scipy)
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
        q = p - 0.5; r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def chi2_quantile(p, k):
    """Quantile de la loi du khi-deux a k ddl (approximation de Wilson-Hilferty)."""
    z = _norm_ppf(p)
    return k * (1 - 2/(9*k) + z * math.sqrt(2/(9*k)))**3


def chi2_sf(x, k):
    """P(khi-deux_k > x) (Wilson-Hilferty)."""
    z = ((x/k)**(1/3) - (1 - 2/(9*k))) / math.sqrt(2/(9*k))
    return 0.5 * math.erfc(z / math.sqrt(2))


# ----------------------------------------------------------------------------
# Placement des individus selon le patron spatial
# ----------------------------------------------------------------------------
def placer(N, a, rng):
    """a >= 0 : agregation croissante (processus a clusters) ;
       a <  0 : regularite croissante (grille perturbee).
       a == 0 : repartition aleatoire (dans les deux limites)."""
    if a >= 0:
        n_parents = max(1, int(N / (2 + 8 * a)))
        sigma = (0.5 * min(W, H)) / (1 + 6 * a)
        px = rng.uniform(0, W, n_parents)
        py = rng.uniform(0, H, n_parents)
        idx = rng.integers(0, n_parents, N)
        x = np.clip(px[idx] + rng.normal(0, sigma, N), 0, W)
        y = np.clip(py[idx] + rng.normal(0, sigma, N), 0, H)
    else:
        t = min(-a / 0.8, 1.0)
        ncol = max(1, int(np.ceil(np.sqrt(N * W / H))))
        nrow = max(1, int(np.ceil(N / ncol)))
        gx = (np.arange(ncol) + 0.5) * W / ncol
        gy = (np.arange(nrow) + 0.5) * H / nrow
        GX, GY = np.meshgrid(gx, gy)
        pts = np.column_stack([GX.ravel(), GY.ravel()])[:N]
        jit = (0.9 * (1 - t) + 0.03) * min(W / ncol, H / nrow)
        x = np.clip(pts[:, 0] + rng.normal(0, jit, len(pts)), 0, W)
        y = np.clip(pts[:, 1] + rng.normal(0, jit, len(pts)), 0, H)
    return x, y


def compter(qx, qy, cote, px, py):
    counts = np.empty(len(qx), dtype=int)
    for i in range(len(qx)):
        dedans = (px >= qx[i]) & (px < qx[i] + cote) & (py >= qy[i]) & (py < qy[i] + cote)
        counts[i] = int(dedans.sum())
    return counts


DEFAUTS = dict(densite=2.0, agregation=0.0, n_quad=40, cote=1.0)


def lancer_gui():
    seed = {"val": 40404}

    fig = plt.figure(figsize=(11.5, 8.6))
    ax_map = fig.add_axes([0.055, 0.40, 0.40, 0.55])
    ax_hist = fig.add_axes([0.56, 0.66, 0.40, 0.29])
    ax_txt = fig.add_axes([0.56, 0.40, 0.40, 0.21]); ax_txt.axis("off")

    ax_color = "#eef2f7"
    specs = [
        ("densite",    "densite (ind. / unite carree)",     0.2, 5.0, 0.1, DEFAUTS["densite"]),
        ("agregation", "agregation  (- regulier ... + agrege)", -0.8, 3.0, 0.1, DEFAUTS["agregation"]),
        ("n_quad",     "nombre de quadrats",                5,   100, 1,   DEFAUTS["n_quad"]),
        ("cote",       "cote du quadrat (unites)",          0.4, 2.0, 0.1, DEFAUTS["cote"]),
    ]
    sliders = {}
    y = 0.29
    for key, label, lo, hi, step, init in specs:
        axs = fig.add_axes([0.16, y, 0.70, 0.028], facecolor=ax_color)
        sliders[key] = Slider(axs, label, lo, hi, valinit=init, valstep=step)
        y -= 0.050

    ax_re = fig.add_axes([0.33, 0.05, 0.20, 0.042])
    ax_reset = fig.add_axes([0.56, 0.05, 0.18, 0.042])
    bouton_re = Button(ax_re, "Reechantillonner", color="#dbe7f3", hovercolor="#c2d6ec")
    bouton_reset = Button(ax_reset, "Reinitialiser", color="#f3dbdb", hovercolor="#ecc2c2")

    cmap = plt.cm.Blues

    def dessiner(event=None):
        densite = float(sliders["densite"].val)
        a = float(sliders["agregation"].val)
        n = int(sliders["n_quad"].val)
        cote = float(sliders["cote"].val)
        rng = np.random.default_rng(seed["val"])

        N = max(1, int(round(densite * AIRE)))
        px, py = placer(N, a, rng)

        # quadrats tires au hasard (a l'interieur du terrain)
        qx = rng.uniform(0, W - cote, n)
        qy = rng.uniform(0, H - cote, n)
        comptages = compter(qx, qy, cote, px, py)

        moy = comptages.mean()
        var = comptages.var(ddof=1) if n > 1 else 0.0
        ratio = var / moy if moy > 0 else float("nan")
        I = (n - 1) * ratio if moy > 0 else float("nan")
        k = n - 1
        se = math.sqrt(var / n)
        aire_q = cote * cote
        D_est = moy / aire_q
        z = _norm_ppf(0.975)
        ic = ((moy - z * se) / aire_q, (moy + z * se) / aire_q)

        # test bilateral de dispersion
        bas, haut = chi2_quantile(0.025, k), chi2_quantile(0.975, k)
        if I > haut:
            verdict = "AGREGE  (variance > moyenne)"
        elif I < bas:
            verdict = "REGULIER  (variance < moyenne)"
        else:
            verdict = "compatible avec une repartition ALEATOIRE"

        # ---- carte ----
        ax_map.clear()
        ax_map.scatter(px, py, s=7, color=VERT, alpha=0.55, edgecolors="none")
        vmax = max(1, comptages.max())
        for i in range(n):
            ax_map.add_patch(Rectangle((qx[i], qy[i]), cote, cote,
                                       facecolor=cmap(comptages[i] / vmax), alpha=0.55,
                                       edgecolor="#333333", lw=0.7, zorder=3))
        ax_map.set_xlim(0, W); ax_map.set_ylim(0, H); ax_map.set_aspect("equal")
        ax_map.set_xticks([]); ax_map.set_yticks([])
        ax_map.set_title(f"Terrain : {N} individus, {n} quadrats", fontsize=12)

        # ---- histogramme des comptages + reference Poisson ----
        ax_hist.clear()
        kmax = int(comptages.max())
        bins = np.arange(-0.5, kmax + 1.5, 1)
        ax_hist.hist(comptages, bins=bins, color=BLEU, alpha=0.75, edgecolor="white")
        ks = np.arange(0, kmax + 1)
        pois = np.array([math.exp(-moy) * moy**kk / math.factorial(kk) for kk in ks]) * n
        ax_hist.plot(ks, pois, "o-", color=ROUGE, ms=4, lw=1.3,
                     label="attendu si aleatoire (Poisson)")
        ax_hist.set_xlabel("individus par quadrat")
        ax_hist.set_ylabel("nombre de quadrats")
        ax_hist.set_title("Distribution des comptages", fontsize=12)
        ax_hist.legend(fontsize=8.5, framealpha=0.9)

        # ---- encadre statistique ----
        ax_txt.clear(); ax_txt.axis("off")
        info = (
            f"moyenne = {moy:.2f}     variance = {var:.2f}\n"
            f"rapport variance / moyenne = {ratio:.2f}\n"
            f"indice de dispersion I = {I:.1f}   (ddl = {k})\n"
            f"test bilateral 5 % : [{bas:.1f} ; {haut:.1f}]\n"
            f"=> {verdict}\n"
            f"\n"
            f"densite estimee = {D_est:.2f} ind./unite carree\n"
            f"IC 95 % (approx.) : [{ic[0]:.2f} ; {ic[1]:.2f}]"
        )
        ax_txt.text(0.0, 0.98, info, transform=ax_txt.transAxes, va="top", ha="left",
                    fontsize=10.5, family="monospace",
                    bbox=dict(boxstyle="round", facecolor="#fbfbe8", edgecolor="#cccccc"))

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

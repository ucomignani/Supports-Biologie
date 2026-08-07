#!/usr/bin/env python3
"""
Metabarcoding : explorateur interactif.

  - En haut, la COURBE DE RAREFACTION : le nombre d'OTU detectes en fonction de la
    profondeur de sequencage. Un trait vertical marque la profondeur choisie, un
    trait horizontal la richesse reelle. On voit la richesse observee grimper avec
    l'effort sans jamais tout a fait atteindre la richesse reelle.

  - En bas, la COMPOSITION : pour les taxons les plus abondants, l'abondance
    relative reelle comparee a la proportion de lectures observee. Le biais
    d'amplification deforme la seconde par rapport a la premiere.

Lancement :
    python3 Simulation_metabarcoding.py

Dependances : numpy, matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

VERT = "#e07b39"     # abondance reelle
BLEU = "#4682b4"     # proportion observee (lectures)
ROUGE = "#b52b2b"


def communaute(S, sigma, rng):
    """Abondances relatives reelles de S taxons (loi log-normale, inegalite = sigma)."""
    ab = np.exp(rng.normal(0, sigma, S))
    return ab / ab.sum()


def richesse_attendue(p, d):
    """Nombre d'OTU attendus a la profondeur d : somme sur les taxons de 1-(1-p_i)^d."""
    return np.sum(1 - (1 - p) ** d)


def shannon(p):
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


DEFAUTS = dict(S=200, sigma=1.5, prof_log=4.0, biais=0.0)


def lancer_gui():
    seed = {"val": 71000}

    fig = plt.figure(figsize=(11.0, 8.6))
    ax_raref = fig.add_axes([0.09, 0.63, 0.87, 0.30])
    ax_comp = fig.add_axes([0.09, 0.36, 0.52, 0.20])
    ax_txt = fig.add_axes([0.66, 0.35, 0.30, 0.22]); ax_txt.axis("off")

    ax_color = "#eef2f7"
    specs = [
        ("S",        "richesse reelle (nb de taxons)",   20,  500,  10,   DEFAUTS["S"]),
        ("sigma",    "inegalite des abondances",         0.2, 2.5,  0.1,  DEFAUTS["sigma"]),
        ("prof_log", "profondeur (log10 des lectures)",  2.0, 5.0,  0.05, DEFAUTS["prof_log"]),
        ("biais",    "biais d'amplification",            0.0, 1.5,  0.1,  DEFAUTS["biais"]),
    ]
    sliders = {}
    y = 0.26
    for key, label, lo, hi, step, init in specs:
        axs = fig.add_axes([0.16, y, 0.68, 0.028], facecolor=ax_color)
        sliders[key] = Slider(axs, label, lo, hi, valinit=init, valstep=step)
        y -= 0.050

    ax_re = fig.add_axes([0.33, 0.045, 0.20, 0.042])
    ax_reset = fig.add_axes([0.56, 0.045, 0.18, 0.042])
    bouton_re = Button(ax_re, "Reechantillonner", color="#dbe7f3", hovercolor="#c2d6ec")
    bouton_reset = Button(ax_reset, "Reinitialiser", color="#f3dbdb", hovercolor="#ecc2c2")

    def dessiner(event=None):
        S = int(sliders["S"].val)
        sigma = float(sliders["sigma"].val)
        depth = int(round(10 ** float(sliders["prof_log"].val)))
        biais = float(sliders["biais"].val)
        rng = np.random.default_rng(seed["val"])

        ab = communaute(S, sigma, rng)                       # abondances relatives reelles
        facteurs = np.exp(rng.normal(0, biais, S))           # biais d'amplification par taxon
        amp = ab * facteurs; amp = amp / amp.sum()           # ce que le sequenceur "voit"
        reads = rng.multinomial(depth, amp)                  # lectures obtenues
        obs = reads / reads.sum()                            # proportions observees
        otu_detectes = int(np.sum(reads > 0))

        # ---- courbe de rarefaction ----
        ax_raref.clear()
        ds = np.logspace(2, 5, 70)
        rich = [richesse_attendue(amp, d) for d in ds]
        ax_raref.plot(ds, rich, color=BLEU, linewidth=2)
        ax_raref.axhline(S, color=ROUGE, linestyle="--", linewidth=1.5,
                         label=f"richesse reelle = {S}")
        ax_raref.axvline(depth, color="#333333", linestyle=":", linewidth=1.5)
        ax_raref.plot([depth], [richesse_attendue(amp, depth)], "o", color="#0b2545", ms=7)
        ax_raref.set_xscale("log")
        ax_raref.set_xlim(100, 100000)
        ax_raref.set_ylim(0, S * 1.08)
        ax_raref.set_xlabel("profondeur de sequencage (nombre de lectures)")
        ax_raref.set_ylabel("OTU detectes")
        ax_raref.set_title("Courbe de rarefaction", fontsize=12)
        ax_raref.legend(loc="lower right", fontsize=9, framealpha=0.9)

        # ---- composition : reelle vs observee (taxons les plus abondants) ----
        ax_comp.clear()
        K = min(15, S)
        ordre = np.argsort(ab)[::-1][:K]
        xx = np.arange(K); w = 0.4
        ax_comp.bar(xx - w/2, ab[ordre], w, color=VERT, label="abondance reelle")
        ax_comp.bar(xx + w/2, obs[ordre], w, color=BLEU, label="proportion observee")
        ax_comp.set_xticks([])
        ax_comp.set_xlabel(f"{K} taxons les plus abondants")
        ax_comp.set_ylabel("part relative")
        ax_comp.set_title("Composition : reelle vs observee", fontsize=11)
        ax_comp.legend(fontsize=8.5, framealpha=0.9)

        # ---- encadre ----
        ax_txt.clear(); ax_txt.axis("off")
        pct = 100 * otu_detectes / S
        note_biais = ("aucun biais : la composition observee\nreflete le reel (au bruit pres)"
                      if biais == 0 else
                      "biais d'amplification : la composition\nobservee est faussee")
        info = (
            f"profondeur choisie : {depth:,} lectures\n".replace(",", " ") +
            f"richesse reelle : {S} taxons\n"
            f"OTU detectes : {otu_detectes}  ({pct:.0f} %)\n"
            f"\n"
            f"Shannon reel     : {shannon(ab):.2f}\n"
            f"Shannon observe  : {shannon(obs):.2f}\n"
            f"\n"
            f"{note_biais}"
        )
        ax_txt.text(0.0, 0.98, info, transform=ax_txt.transAxes, va="top", ha="left",
                    fontsize=10, family="monospace",
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

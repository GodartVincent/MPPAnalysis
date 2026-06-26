"""
Correction « 120 minutes » des phases finales (NB18).

PROBLÈME : MPP score les matchs à la fin des éventuelles PROLONGATIONS (120'),
alors que les cotes saisies (1N2 de CDM_2026.csv et scores exacts de
exact_scores.csv, toutes Pinnacle) sont à 90'. En knockout les nuls sont donc
sur-représentés par les cotes 90' (un nul à 90' part toujours en prolongation),
et les crowds de scores exacts Winamax (90') sont biaisés vers les nuls. Les
crowds 1N2 de CDM_2026.csv viennent DIRECTEMENT de MPP -> déjà à 120', fiables ;
seuls les scores exacts (cotes ET crowds) doivent être recalés à 120'.

PIPELINE (par match KO renseigné) :
  1. 1N2@90 = Shin(cote_1/N/2) (déjà calculé par le pipeline = base_true_probas).
  2. Q_A = Shin(cote_qualif_A, cote_qualif_B) (marché « to qualify », 2 issues).
  3. 1N2@120 = get_120m_outcome_probas(P_A, P_B, P_N, Q_A) (cible fiable ; fallback
     prior z=0.45 si Q_A indisponible ou singularité P_A≈P_B).
  4. Fit (λ_A, λ_B) par moment-matching sur la grille de scores exacts 90' ancrée
     sur le 1N2@90 (Poisson indépendant ; Dixon-Coles ρ négligeable en prolongation,
     λ_ET ≈ 0.2 -> buts quasi toujours nuls).
  5. Calibration du multiplicateur de prolongation m (ex-0.15 figé) : recherche de
     racine 1D pour que la RÉGION NULLE de P_120 (Σ_x P_120(x,x)) égale P_N_120.
     Σ_x P_120(x,x) est monotone décroissante en m -> racine unique.
  6. P_120(x,y) (conservation de masse) :
         P_120(x,y) = P_90(x,y)·1[x≠y] + Σ_{k=0}^{min(x,y)} P_90(k,k)·P_30(x-k,y-k)
     avec P_30(a,b) = Poisson(a; m·λ_A)·Poisson(b; m·λ_B). Le terme P_90 est EXCLU
     pour les nuls (un nul à 90' n'est jamais final) -> Σ P_120 = 1 exactement.
  7. Ré-ancrage des scores exacts listés sur le 1N2@120 par outcome (les régions
     décisives collent exactement à P_A_120 / P_B_120 ; le nul est déjà calé par m).
  8. Crowds 120' : Winamax -> MPP (forme), reversement de l'excédent de crowd des
     nuls (vs le crowd nul MPP) sur les scores « +1 but », normalisation par outcome
     en TOUTE FIN (cf. correct_cond_crowd_120).

Le module ne dépend PAS de numba : tout est en O(scores) par match, appelé une
fois par match KO (pas dans la boucle DP).
"""

import re
import warnings

import numpy as np
from scipy.stats import poisson
from scipy.optimize import brentq

from mpp_project.core import (
    ExactScoreMarket,
    calculate_true_outcome_probas_from_odds,
    correct_cond_crowd,
    exact_score_bonus,
    _score_to_outcome,
)


def _parse_score(score):
    """'b1-b2' -> (b1, b2) entiers."""
    b1, b2 = (int(x) for x in str(score).strip().split("-"))
    return b1, b2


# ==========================================================================
# 1. 1N2 à 120 minutes (marché « to qualify »)
# ==========================================================================
def devig_to_qualify(cote_qualif_A, cote_qualif_B):
    """
    Dé-vig du marché « to qualify / to advance » (2 issues, somme 1) par Shin ->
    renvoie Q_A = P(l'équipe A se qualifie). None si une cote est absente/<=0
    (marché indisponible -> l'appelant retombe sur le prior z=0.45).
    """
    if cote_qualif_A is None or cote_qualif_B is None:
        return None
    try:
        a = float(cote_qualif_A)
        b = float(cote_qualif_B)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0.0 or b <= 0.0:
        return None
    p = calculate_true_outcome_probas_from_odds(np.array([a, b]), check_margin=False)
    return float(p[0])


def get_120m_outcome_probas(P_A, P_B, P_N, Q_A=None, z_prior=0.45):
    """
    1N2 à 120' à partir du 1N2 à 90' (P_A, P_N, P_B) et de la proba de
    qualification Q_A (marché « to qualify »).

    On décompose le nul à 90' (masse P_N) en : A gagne en prolongation (part x),
    B gagne en prolongation (part y), toujours nul à 120' -> tirs au but (part z,
    score nul conservé par MPP). La résolution algébrique utilise Q_A :
        Q_A = P_A + x·P_N + 0.5·z·P_N      (TAB ~ 50/50)
      => delta := x - y = 2·(Q_A - P_A - 0.5·P_N) / P_N
    avec l'hypothèse x/y = P_A/P_B (le vainqueur en prolongation suit la force 90').

    FALLBACK (prior z = `z_prior`, défaut 0.45) si Q_A est None ou si P_A ≈ P_B
    (le marché 'to qualify' n'aide alors pas à séparer x et y) : on garde z = z_prior
    et on répartit (1 - z) entre x et y au prorata de P_A et P_B (équiprobable si
    P_A == P_B). Conserve la masse : P_A_120 + P_N_120 + P_B_120 = P_A + P_N + P_B.

    Renvoie (P_A_120, P_N_120, P_B_120).
    """
    P_A = float(P_A)
    P_B = float(P_B)
    P_N = float(P_N)
    if P_N <= 0.0:
        return P_A, P_N, P_B  # pas de nul à redistribuer

    use_fallback = Q_A is None or abs(P_A - P_B) < 0.01
    if use_fallback:
        z = z_prior
        denom = P_A + P_B
        if denom > 0.0:
            x = (1.0 - z) * (P_A / denom)
            y = (1.0 - z) * (P_B / denom)
        else:
            x = y = (1.0 - z) / 2.0
    else:
        delta = 2.0 * (float(Q_A) - P_A - 0.5 * P_N) / P_N
        x = delta * (P_A / (P_A - P_B))
        y = x * (P_B / P_A)
        z = 1.0 - x - y
        # Sécurité : cotes incohérentes -> clip puis renormalisation de (x, y, z).
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        z = max(0.0, min(1.0, z))
        total = x + y + z
        if total > 0.0:
            x, y, z = x / total, y / total, z / total
        else:
            z = z_prior
            x = y = (1.0 - z) / 2.0

    P_A_120 = P_A + x * P_N
    P_B_120 = P_B + y * P_N
    P_N_120 = z * P_N
    return P_A_120, P_N_120, P_B_120


# ==========================================================================
# 2. Fit des taux de buts + grille P_120
# ==========================================================================
def fit_goal_rates(scores, p_score):
    """
    Estime (λ_A, λ_B) par MOMENT-MATCHING (espérances de buts) sur la distribution
    de scores `p_score` (alignée sur `scores`, somme ~1). Poisson indépendant.

    Légère sous-estimation par troncature (scores extrêmes non listés) — acceptée :
    les λ servent uniquement à générer l'incrément de prolongation P_30 (λ_ET = m·λ,
    très petit), dont la forme exacte importe peu.
    """
    p = np.asarray(p_score, dtype=float)
    s = p.sum()
    if s <= 0.0:
        return 0.0, 0.0
    p = p / s
    xs = np.array([_parse_score(sc)[0] for sc in scores], dtype=float)
    ys = np.array([_parse_score(sc)[1] for sc in scores], dtype=float)
    lam_A = float(np.dot(p, xs))
    lam_B = float(np.dot(p, ys))
    return lam_A, lam_B


def build_p120_grid(grid90, lam_A, lam_B, m):
    """
    Grille P_120 (conservation de masse) à partir de la grille 90' `grid90`
    ((G+1, G+1), somme 1) et des taux de buts. Multiplicateur de prolongation `m`
    (λ_ET = m·λ). P_30 = Poisson(·; m·λ_A) ⊗ Poisson(·; m·λ_B).

        P_120(x,y) = grid90(x,y)·1[x≠y] + Σ_{k=0}^{min(x,y)} grid90(k,k)·P_30(x-k,y-k)

    Le terme grid90(x,y) est EXCLU sur la diagonale (nuls 90' -> toujours en
    prolongation). Renvoie une grille (G+1, G+1) sommant à 1 (si grid90 somme à 1).
    """
    G = grid90.shape[0] - 1
    idx = np.arange(G + 1)
    p30_A = poisson.pmf(idx, max(m * lam_A, 1e-12))
    p30_B = poisson.pmf(idx, max(m * lam_B, 1e-12))
    P30 = np.outer(p30_A, p30_B)  # (G+1, G+1)
    # P_30 = distribution PROPRE de l'incrément de prolongation sur la grille (la queue
    # au-delà de G est négligeable pour λ_ET≈0.2 ; on renormalise pour la conservation).
    s30 = P30.sum()
    if s30 > 0.0:
        P30 = P30 / s30

    P120 = np.zeros_like(grid90)
    # Terme régulation (hors diagonale)
    for x in range(G + 1):
        for y in range(G + 1):
            if x != y:
                P120[x, y] += grid90[x, y]
    # Terme prolongation : pour chaque nul 90' k-k, ajoute l'incrément P_30. L'index
    # d'arrivée est REPLIÉ sur la borne G (k+a, k+b clampés) -> toute la masse du nul
    # est placée (Σ_{a,b} P_30 = 1) => P_120 conserve la masse au bit près.
    for k in range(G + 1):
        pkk = grid90[k, k]
        if pkk == 0.0:
            continue
        for a in range(G + 1):
            ix = min(k + a, G)
            for b in range(G + 1):
                iy = min(k + b, G)
                P120[ix, iy] += pkk * P30[a, b]
    return P120


def _draw_mass(grid):
    """Σ_x grid(x,x) (masse de la région nulle d'une grille de scores)."""
    return float(np.trace(grid))


def _p_equal_extra_time(lam_A, lam_B, m, support=60):
    """
    P(ΔA = ΔB) où ΔA ~ Poisson(m·λ_A), ΔB ~ Poisson(m·λ_B) = Σ_d pmf(d;m·λ_A)·pmf(d;m·λ_B),
    calculée sur un GRAND support (indépendant de la grille de scores). Strictement
    décroissante en m, de 1 (m=0) vers 0 (m grand). C'est la fraction des nuls 90' qui
    RESTENT nuls à 120' (-> tirs au but) : la masse de la région nulle de P_120 vaut
    exactement D90 · P(ΔA=ΔB), où D90 = Σ_k P_90(k,k) (cf. build_p120_grid sommée).
    """
    d = np.arange(support + 1)
    pa = poisson.pmf(d, max(m * lam_A, 1e-12))
    pb = poisson.pmf(d, max(m * lam_B, 1e-12))
    return float(np.dot(pa, pb))


def calibrate_extra_time_multiplier(grid90, lam_A, lam_B, target_PN120,
                                    m_lo=0.0, m_hi=5.0, tol=1e-9):
    """
    Trouve le multiplicateur de prolongation m tel que la masse de la région nulle de
    P_120 égale `target_PN120` (= P_N_120 du marché 'to qualify').

    La masse de la région nulle = D90 · P(ΔA=ΔB) (D90 = masse nulle 90', invariante en m),
    où P(ΔA=ΔB) est calculée analytiquement (_p_equal_extra_time) sur un grand support :
    strictement décroissante de D90 (m=0, nuls inchangés) vers 0 (m grand, presque toujours
    un but en prolongation) -> racine unique. On calibre sur cette forme analytique plutôt
    que sur la grille tronquée (dont la masse nulle remonte artificiellement à grand m, par
    repli de la queue Poisson sur le coin de la grille). Renvoie m (>= 0).

    Cas limites : pas de masse nulle 90', ou cible >= masse nulle 90' (pas de déflation
    requise) -> m = 0.
    """
    draw90 = _draw_mass(grid90)
    if draw90 <= tol:
        return 0.0  # aucune masse nulle à déflater

    def f(m):
        return draw90 * _p_equal_extra_time(lam_A, lam_B, m) - target_PN120

    if f(m_lo) <= tol:
        return float(m_lo)  # cible déjà >= masse nulle (pas de déflation requise)

    # Élargit la borne haute jusqu'à changer de signe (cible très petite -> grand m)
    hi = m_hi
    n_expand = 0
    while f(hi) > 0.0 and n_expand < 12:
        hi *= 2.0
        n_expand += 1
    if f(hi) > 0.0:
        return float(hi)  # cible inatteignable même à m grand -> borne haute

    return float(brentq(f, m_lo, hi, xtol=1e-6))


# ==========================================================================
# 3. Crowds des scores exacts à 120 minutes
# ==========================================================================
def correct_cond_crowd_120(crowds_raw, scores, outcomes, mpp_outcome_crowd_120,
                           mpp_outcome_crowd_90=None):
    """
    Convertit les crowds Winamax (90') de scores exacts en crowd CONDITIONNEL MPP
    (120'), renvoyé normalisé PAR OUTCOME (somme 1 par issue), aligné sur `scores`.

    Étapes :
      1. Winamax -> forme MPP GLOBALE (somme 1 sur tous les scores), via
         `correct_cond_crowd(..., return_global=True)` ancrée sur le 1N2
         `mpp_outcome_crowd_90` (90', pour préserver la sur-représentation des nuls
         propre au marché Winamax). None -> simple pénalité 0-0 sur la forme Winamax.
      2. Reversement de l'EXCÉDENT de nul : excédent = (masse de crowd des scores
         nuls) − crowd nul MPP 120' (`mpp_outcome_crowd_120[1]`). Pour chaque nul
         k-k, son excédent est retiré et reversé sur les scores « +1 but » (k+1,k)
         (victoire 1) et (k,k+1) (victoire 2), au prorata de la hype des deux scores
         cibles (repli : hype d'outcome crowd_1 vs crowd_2 ; sinon 50/50 ; si un seul
         cible existe, tout pour lui). Masse globale conservée. Sauté si excédent <= 0.
      3. Normalisation par outcome EN TOUTE FIN -> crowd conditionnel (somme 1 par
         issue). C'est le « bémol » assumé : fixer les totaux par outcome après le
         reversement n'altère que la FORME intra-outcome (les scores « +1 but » sont
         re-pondérés), pas les totaux.
    """
    cr = np.asarray(crowds_raw, dtype=float)
    outc = np.asarray(outcomes)
    q120 = np.asarray(mpp_outcome_crowd_120, dtype=float)
    q120 = q120 / q120.sum() if q120.sum() > 0 else q120

    # 1. Forme MPP globale (somme 1) conservant le poids des nuls 90'.
    g = correct_cond_crowd(cr, scores, outc, mpp_outcome_crowd=mpp_outcome_crowd_90,
                           return_global=True)
    g = np.asarray(g, dtype=float)
    if g.sum() <= 0.0:
        return np.zeros_like(cr)

    # 2. Reversement de l'excédent de nul.
    xy = [_parse_score(s) for s in scores]
    pos = {f"{x}-{y}": i for i, (x, y) in enumerate(xy)}
    draw_idx = [i for i, (x, y) in enumerate(xy) if x == y]
    draw_total = float(sum(g[i] for i in draw_idx))
    qN = float(q120[1]) if q120.size >= 2 else 0.0

    if draw_total > qN + 1e-12 and draw_total > 0.0:
        scale = qN / draw_total
        q1 = float(q120[0]) if q120.size >= 1 else 0.0
        q2 = float(q120[2]) if q120.size >= 3 else 0.0
        for i in draw_idx:
            k = xy[i][0]
            i1 = pos.get(f"{k + 1}-{k}")        # +1 but pour A (victoire 1)
            i2 = pos.get(f"{k}-{k + 1}")        # +1 but pour B (victoire 2)
            if i1 is None and i2 is None:
                continue  # aucun score cible -> on garde la masse sur le nul
            ex = g[i] * (1.0 - scale)
            g[i] -= ex
            h1 = g[i1] if i1 is not None else 0.0
            h2 = g[i2] if i2 is not None else 0.0
            if i1 is None:
                w1, w2 = 0.0, 1.0
            elif i2 is None:
                w1, w2 = 1.0, 0.0
            elif h1 + h2 > 0.0:
                w1, w2 = h1 / (h1 + h2), h2 / (h1 + h2)
            elif q1 + q2 > 0.0:
                w1, w2 = q1 / (q1 + q2), q2 / (q1 + q2)
            else:
                w1, w2 = 0.5, 0.5
            if i1 is not None:
                g[i1] += ex * w1
            if i2 is not None:
                g[i2] += ex * w2

    # 3. Normalisation par outcome -> conditionnel (somme 1 par issue).
    cond = np.zeros_like(cr)
    for o in (0, 1, 2):
        mask = outc == o
        tot = float(g[mask].sum())
        if tot > 0.0:
            cond[mask] = g[mask] / tot
    return cond


# ==========================================================================
# 4. Marché de scores exacts corrigé 120' (orchestration)
# ==========================================================================
def _parse_exact_score_data(exact_score_data):
    """
    Sépare les scores d'un dict { 'b1-b2': (cote, crowd) } en :
      - listés (cote valide > 0) : scores / odds / outcomes (pariables) ;
      - porteurs de crowd (cote OU crowd renseigné) : c_scores / c_outcomes /
        c_vals / c_listed (servent à la distribution de crowd).
    Identique à la logique de core.build_exact_score_market.
    """
    scores, odds, outcomes = [], [], []
    c_scores, c_outcomes, c_vals, c_listed = [], [], [], []
    for score, data in exact_score_data.items():
        cote, crowd = data
        o = _score_to_outcome(score)
        valid = cote is not None and cote > 0
        if valid:
            scores.append(str(score))
            odds.append(float(cote))
            outcomes.append(o)
        if valid or crowd is not None:
            c_scores.append(str(score))
            c_outcomes.append(o)
            c_vals.append(float(crowd) if crowd else 0.0)
            c_listed.append(valid)
    return (scores, np.asarray(odds, dtype=float), np.asarray(outcomes, dtype=np.int8),
            c_scores, np.asarray(c_outcomes, dtype=np.int8),
            np.asarray(c_vals, dtype=float), np.asarray(c_listed, dtype=bool))


def build_exact_score_market_120(exact_score_data, outcome_probas_90,
                                 mpp_outcome_crowd_120, Q_A=None,
                                 z_prior=0.45, goal_grid=None):
    """
    Construit le marché de scores exacts CORRIGÉ 120' d'un match KO et renvoie
    (ExactScoreMarket, info). Pendant 120' de `core.build_exact_score_market`.

    Paramètres :
      exact_score_data     : dict { 'b1-b2': (cote_pinnacle_90, crowd_winamax_90) }.
      outcome_probas_90    : 1N2@90 (3,) = Shin(cote_1/N/2) du CSV (= base_true_probas).
      mpp_outcome_crowd_120: 1N2 MPP du match [crowd_1, crowd_N, crowd_2] (CDM, 120').
      Q_A                  : proba de qualification de A (marché 'to qualify'), ou None
                             (-> prior z = `z_prior`).
      goal_grid            : taille de la grille de buts par équipe (auto si None).

    info = { 'm', 'lam_A', 'lam_B', 'Q_A', 'p1n2_90', 'p1n2_120',
             'region_sums_90', 'region_sums_120', 'draw_mass_90' }.
    """
    (scores, odds, outcomes, c_scores, c_outcomes,
     c_vals, c_listed) = _parse_exact_score_data(exact_score_data)
    if len(scores) == 0:
        raise ValueError("build_exact_score_market_120 : aucun score avec une cote valide.")

    op90 = np.asarray(outcome_probas_90, dtype=float)

    # --- P_90 ancré sur le 1N2@90 (Shin des cotes scores, ré-échelonné par outcome) ---
    p_raw = np.asarray(calculate_true_outcome_probas_from_odds(odds), dtype=float)
    p90 = np.zeros_like(p_raw)
    for o in (0, 1, 2):
        mask = outcomes == o
        s = float(p_raw[mask].sum())
        if s > 0.0:
            p90[mask] = p_raw[mask] / s * op90[o]
    tot = float(p90.sum())
    if tot > 0.0:
        p90 = p90 / tot

    # --- Grille 90' ---
    xy = [_parse_score(s) for s in scores]
    max_goals = max((max(x, y) for x, y in xy), default=0)
    G = int(goal_grid) if goal_grid is not None else max(10, max_goals + 3)
    grid90 = np.zeros((G + 1, G + 1), dtype=float)
    for (x, y), p in zip(xy, p90):
        if x <= G and y <= G:
            grid90[x, y] += p
    # renormalise la grille (les scores hors grille — rares — sont ignorés)
    if grid90.sum() > 0.0:
        grid90 = grid90 / grid90.sum()

    region_sums_90 = (
        float(np.tril(grid90, -1).sum()),   # x>y : victoire 1
        float(np.trace(grid90)),            # x==y : nul
        float(np.triu(grid90, 1).sum()),    # x<y : victoire 2
    )

    # --- 1N2@120 (cible) + fit + calibration de m ---
    P_A90, P_N90, P_B90 = float(op90[0]), float(op90[1]), float(op90[2])
    P_A120, P_N120, P_B120 = get_120m_outcome_probas(P_A90, P_B90, P_N90, Q_A, z_prior)

    lam_A, lam_B = fit_goal_rates(scores, p90)
    m = calibrate_extra_time_multiplier(grid90, lam_A, lam_B, P_N120)
    grid120 = build_p120_grid(grid90, lam_A, lam_B, m)

    region_sums_120 = (
        float(np.tril(grid120, -1).sum()),
        float(np.trace(grid120)),
        float(np.triu(grid120, 1).sum()),
    )

    # --- p_score@120 des scores listés, ré-ancrés par outcome sur le 1N2@120 ---
    p120_listed = np.array([grid120[x, y] for (x, y) in xy], dtype=float)
    op120 = np.array([P_A120, P_N120, P_B120], dtype=float)
    p_anchored = np.zeros_like(p120_listed)
    for o in (0, 1, 2):
        mask = outcomes == o
        s = float(p120_listed[mask].sum())
        if s > 0.0:
            p_anchored[mask] = p120_listed[mask] / s * op120[o]
    tot = float(p_anchored.sum())
    if tot > 0.0:
        p_anchored = p_anchored / tot

    # --- Crowds 120' (forme à 90', reversement de l'excédent de nul, norm finale) ---
    cc_full = correct_cond_crowd_120(
        c_vals, c_scores, c_outcomes,
        mpp_outcome_crowd_120=mpp_outcome_crowd_120,
        mpp_outcome_crowd_90=op90,
    )
    cond_crowd = cc_full[c_listed]
    bonus = np.asarray([exact_score_bonus(cc) for cc in cond_crowd], dtype=np.int64)

    market = ExactScoreMarket(scores, outcomes, p_anchored, cond_crowd, bonus)
    info = {
        "m": m, "lam_A": lam_A, "lam_B": lam_B, "Q_A": Q_A,
        "p1n2_90": (P_A90, P_N90, P_B90),
        "p1n2_120": (P_A120, P_N120, P_B120),
        "region_sums_90": region_sums_90,
        "region_sums_120": region_sums_120,
        "draw_mass_90": _draw_mass(grid90),
    }
    return market, info

"""
Met à jour les cotes (1N2 + scores exacts) + crowds des matchs PAS ENCORE JOUÉS de
CDM_2026.csv / exact_scores.csv à partir des pages sauvegardées dans data/MatchInfo/.

Sources (identifiées par le NOM DE FICHIER, l'équipe étant dans le nom) :
  - « Cotes pour Football _ Paris sur Football.html » : tableau 1N2 de TOUS les
    matchs à venir (Pinnacle, vue « Tous les matchs »).
  - « Cotes des paris de <Eq1> vs <Eq2> _ Cotes de FIFA - World Cup.html » : page
    Pinnacle détaillée d'UN match (1N2 + scores exacts).
  - « <Eq1>_<Eq2>_Winamax.{txt,html} » : crowds (%) des scores exacts (Winamax).

⚠️ Les pages Pinnacle doivent être enregistrées en « Page Web, COMPLÈTE » : une
sauvegarde « HTML seul » ne capture PAS le tableau de cotes (rendu en JavaScript)
-> fichier vide -> rien à parser.

Pilotage : on parcourt les matchs dont la colonne `result` est VIDE dans
CDM_2026.csv (matchs futurs). Pour chacun :
  - 1N2 : page par-match si elle existe ET est plus récente (mtime) que
    « Cotes pour Football », sinon « Cotes pour Football ». Si aucune cote
    Pinnacle n'est disponible, les cotes existantes sont PRÉSERVÉES ;
  - scores exacts (cotes) : page par-match ; crowds : fichier Winamax.

Les matchs présents dans « Cotes pour Football » mais ABSENTS de CDM_2026.csv ne
sont PAS ajoutés automatiquement : le script demande confirmation pour chacun
(--add-mode ask|skip|all).

Gotchas gérés :
  - Fichiers UTF-8 (accents fragiles) -> lecture UTF-8 + normalisation via
    mpp_project.core.normalize_team_name (+ traduction EN->FR + alias + fuzzy).
  - Noms d'équipes ANGLAIS dans les noms de fichiers (Ecuador, Ivory Coast...).
  - Winamax en 2 formats : ancien (bloc « Score exact » … « Moins de sélections »)
    et compact (triplets score/cote/% bruts) -> les deux sont gérés.
  - Home/Away : inversion détectée en comparant l'équipe « domicile » (1re du nom
    de fichier / 1re listée) à team_A du CDM.
  - Scores qui APPARAISSENT / DISPARAISSENT : avec données Pinnacle, on remplace
    toutes les cotes du match (scores disparus sans crowd supprimés) ; sans données
    Pinnacle, les cotes existantes sont conservées.
  - crowd Winamax prioritaire sur l'existant ; pts/bet/result préservés.

Usage :
  python update_odds_from_pinnacle.py                 # applique et écrit les CSV
  python update_odds_from_pinnacle.py --dry-run       # aperçu, sans écrire
  python update_odds_from_pinnacle.py --add-mode skip  # n'ajoute aucun match inconnu
"""

import argparse
import re
import sys
import warnings
from difflib import get_close_matches
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from mpp_project.core import normalize_team_name

# La console Windows est souvent en cp1252 -> bascule stdout en UTF-8 (emojis de log).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

# Traduction des noms d'équipes ANGLAIS (clé = forme normalisée) vers les noms
# normalisés français du CDM. Les noms déjà français passent au travers.
EN_TO_CDM = {
    "mexico": "mexique", "south_africa": "afrique_du_sud", "south_korea": "coree_du_sud",
    "czech_republic": "tchequie", "czechia": "tchequie", "canada": "canada",
    "bosnia": "bosnie", "bosnia_and_herzegovina": "bosnie", "united_states": "etats_unis",
    "usa": "etats_unis", "paraguay": "paraguay", "qatar": "qatar", "switzerland": "suisse",
    "brazil": "bresil", "morocco": "maroc", "haiti": "haiti", "scotland": "ecosse",
    "australia": "australie", "turkey": "turquie", "turkiye": "turquie",
    "germany": "allemagne", "curacao": "curacao", "netherlands": "pays_bas",
    "japan": "japon", "ivory_coast": "cote_d_ivoire", "cote_divoire": "cote_d_ivoire",
    "ecuador": "equateur", "sweden": "suede", "tunisia": "tunisie", "spain": "espagne",
    "cape_verde": "cap_vert", "cabo_verde": "cap_vert", "belgium": "belgique",
    "egypt": "egypte", "saudi_arabia": "arabie_saoudite", "uruguay": "uruguay",
    "iran": "iran", "new_zealand": "nouvelle_zelande", "france": "france",
    "senegal": "senegal", "iraq": "irak", "norway": "norvege", "argentina": "argentine",
    "algeria": "algerie", "austria": "autriche", "jordan": "jordanie",
    "portugal": "portugal", "dr_congo": "congo", "rd_congo": "congo", "congo": "congo",
    "england": "angleterre", "croatia": "croatie", "ghana": "ghana", "panama": "panama",
    "uzbekistan": "ouzbekistan", "colombia": "colombie",
}

# Alias appliqués APRÈS traduction (coquilles Pinnacle FR qui ne normalisent pas).
TEAM_ALIASES = {"japani": "japon"}

_ODD_RE = re.compile(r"^\d+\.\d{2,3}$")            # cote décimale "1.609", "131.540"
_TIME_RE = re.compile(r"^\d{1,2}:\d{2}$")          # heure "22:00"
_SCORE_LINE_RE = re.compile(r"^(.+?)\s+(\d+),\s*(.+?)\s+(\d+)$")  # "Ecuador 0, Germany 1"
_SCORE_DASH_RE = re.compile(r"^(\d+)\s*-\s*(\d+)$")               # "1 - 0" (Winamax)
_PCT_RE = re.compile(r"^(\d+)\s*%$")                              # "3%" (crowd Winamax)
_PINNACLE_FN_RE = re.compile(r"Cotes des paris de (.+?) vs (.+?) _", re.I)
_TIME_H_RE = re.compile(r"^\d{1,2}h\d{2}$")        # heure MPP "22h00", "1h00"
_INT_RE = re.compile(r"^\d+$")                      # gain MPP entier

# Bornes du bloc des scores exacts (page par-match Pinnacle).
_EXACT_START = "score exact"
_EXACT_END = "score exact en 1re mi-temps"
# Bornes du bloc des scores exacts (page Winamax, format ancien).
_WMX_EXACT_START = "score exact"
_WMX_EXACT_END = ("moins de sélections", "score exact multichance",
                  "score exact en 1re mi-temps")

ALL_MATCHES_FN_PREFIX = "cotes pour football"
MPP_FN_PREFIX = "mon petit prono"


# ----------------------------------------------------------------------------
# Résolution des noms d'équipes vers les noms normalisés du CDM
# ----------------------------------------------------------------------------
def _norm(raw):
    n = normalize_team_name(raw) or ""
    n = n.replace("'", "_").replace("’", "_")  # apostrophes droite ET courbe
    while "__" in n:
        n = n.replace("__", "_")
    return n.strip("_")


def _is_team_token(s):
    """True si `s` ressemble à un nom d'équipe (lettres, hors labels connus)."""
    s = s.strip()
    if len(s) < 2 or not re.search(r"[A-Za-zÀ-ÿ]", s):
        return False
    return s.lower() not in ("match nul", "activer mon bonus mpp")


def resolve_team(raw, cdm_teams, cutoff=0.8):
    """Nom CDM normalisé pour `raw` (FR ou EN), ou None si rien de fiable."""
    n = _norm(raw)
    if not n:
        return None
    n = EN_TO_CDM.get(n, n)
    n = TEAM_ALIASES.get(n, n)
    if n in cdm_teams:
        return n
    cand = get_close_matches(n, list(cdm_teams), n=1, cutoff=cutoff)
    return cand[0] if cand else None


def _read_strings(path):
    """Lit une page (UTF-8) et renvoie la liste des chaînes visibles."""
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    return list(BeautifulSoup(txt, "html.parser").stripped_strings)


# ----------------------------------------------------------------------------
# Équipes depuis le NOM DE FICHIER
# ----------------------------------------------------------------------------
def is_all_matches_file(name):
    return name.lower().startswith(ALL_MATCHES_FN_PREFIX)


def teams_from_pinnacle_filename(name):
    """('Ecuador', 'Germany') depuis « Cotes des paris de Ecuador vs Germany _ … »."""
    m = _PINNACLE_FN_RE.search(name)
    return (m.group(1).strip(), m.group(2).strip()) if m else (None, None)


def teams_from_winamax_filename(path, cdm_teams):
    """
    Résout les 2 équipes d'un nom « <Eq1>_<Eq2>_Winamax.ext ». Les noms d'équipes
    pouvant contenir des underscores (ex. Ivory_Coast), on teste tous les points de
    coupe et on retient celui où les DEUX côtés se résolvent. Renvoie (home, away)
    en noms CDM, ou (None, None).
    """
    stem = re.sub(r"_?winamax$", "", Path(path).stem, flags=re.I)
    toks = stem.split("_")
    for k in range(1, len(toks)):
        h = resolve_team(" ".join(toks[:k]), cdm_teams)
        a = resolve_team(" ".join(toks[k:]), cdm_teams)
        if h and a and h != a:
            return h, a
    return None, None


# ----------------------------------------------------------------------------
# Parsing « Cotes pour Football » (1N2 de tous les matchs à venir)
# ----------------------------------------------------------------------------
def parse_all_matches(path):
    """Liste de dicts {home, away, c1, cN, c2} (noms BRUTS, home = 1er listé)."""
    texts = _read_strings(path)
    out, i, n = [], 0, len(texts)
    while i < n - 1:
        a, b = texts[i], texts[i + 1]
        if a.endswith("(Match)") and b.endswith("(Match)"):
            home = a[:-len("(Match)")].strip()
            away = b[:-len("(Match)")].strip()
            j = i + 2
            if j < n and _TIME_RE.match(texts[j]):
                j += 1
            odds = []
            while j < n and len(odds) < 3:
                if _ODD_RE.match(texts[j]):
                    odds.append(float(texts[j]))
                    j += 1
                elif texts[j].endswith("(Match)"):
                    break
                else:
                    j += 1
            if len(odds) == 3:
                out.append(dict(home=home, away=away, c1=odds[0], cN=odds[1], c2=odds[2]))
            i += 2
        else:
            i += 1
    return out


def is_mpp_file(name):
    return name.lower().startswith(MPP_FN_PREFIX)


def parse_mpp_crowds(path):
    """
    Page MPP (« Mon Petit Prono ») : renvoie une liste de dicts
    {home, away, c1, cN, c2} où c* = crowd 1N2 EN POURCENTAGE (int), home = équipe
    listée en 1er. Chaque match s'y présente comme :
        <rang> <teamA> 'J.x' '-' <heure> g1 c1% gN cN% g2 c2% <rang> <teamB> ...
    On ancre sur le triplet (gain, %)×3 précédé d'une heure « 22h00 ».
    """
    texts = _read_strings(path)
    n = len(texts)
    out = []
    for i in range(4, n - 7):
        # 3 paires (gain entier, pourcentage) consécutives.
        if not (_INT_RE.match(texts[i]) and _PCT_RE.match(texts[i + 1])
                and _INT_RE.match(texts[i + 2]) and _PCT_RE.match(texts[i + 3])
                and _INT_RE.match(texts[i + 4]) and _PCT_RE.match(texts[i + 5])):
            continue
        # Ancrage : juste avant le 1er gain se trouve l'heure (texts[i-1]).
        if not _TIME_H_RE.match(texts[i - 1]):
            continue
        home, away = texts[i - 4].strip(), texts[i + 7].strip()
        if not (_is_team_token(home) and _is_team_token(away)):
            continue
        out.append(dict(home=home, away=away,
                        c1=int(_PCT_RE.match(texts[i + 1]).group(1)),
                        cN=int(_PCT_RE.match(texts[i + 3]).group(1)),
                        c2=int(_PCT_RE.match(texts[i + 5]).group(1))))
    return out


# ----------------------------------------------------------------------------
# Parsing d'une page par-match Pinnacle (1N2 + scores exacts)
# ----------------------------------------------------------------------------
def parse_match_page(path):
    """
    Renvoie (c1, cN, c2, exact) où exact = {(buts_home, buts_away): cote}. Les
    équipes viennent du NOM de fichier (home = 1re listée), cohérent avec l'ordre
    des scores exacts (« <home> i, <away> j »). c1/cN/c2 = None si introuvables.
    """
    texts = _read_strings(path)
    n = len(texts)
    c1 = cN = c2 = None
    qa_home = qa_away = None

    # 1N2 : on repère « Nul » du money line (home avant, away après).
    for k in range(2, n - 3):
        if texts[k].strip().lower() != "nul":
            continue
        if (_ODD_RE.match(texts[k - 1]) and _ODD_RE.match(texts[k + 1])
                and _ODD_RE.match(texts[k + 3])
                and not _ODD_RE.match(texts[k - 2]) and not _ODD_RE.match(texts[k + 2])):
            c1, cN, c2 = float(texts[k - 1]), float(texts[k + 1]), float(texts[k + 3])
            break

    # « Pari sur le vainqueur – Qualification » (matchs de phase finale) : 2 cotes
    # « to qualify » dans l'ordre home / away. Les 2 premières cotes après l'entête,
    # bornées par le marché suivant (« Pari sur le vainqueur … »).
    for idx, t in enumerate(texts):
        low = t.strip().lower()
        if "vainqueur" in low and "qualif" in low:
            odds = []
            for j in range(idx + 1, min(idx + 10, n)):
                lj = texts[j].strip().lower()
                if "vainqueur" in lj:  # entête du marché suivant
                    break
                if _ODD_RE.match(texts[j]):
                    odds.append(float(texts[j]))
                    if len(odds) == 2:
                        break
            if len(odds) == 2:
                qa_home, qa_away = odds
            break

    # Scores exacts : bloc « Score exact » … « Score exact en 1re mi-temps ».
    exact = {}
    start = end = None
    for idx, t in enumerate(texts):
        low = t.strip().lower()
        if low == _EXACT_START and start is None:
            start = idx
        elif low == _EXACT_END:
            end = idx
            break
    if start is not None:
        stop = end if end is not None else n
        idx = start + 1
        while idx < stop:
            m = _SCORE_LINE_RE.match(texts[idx])
            if m and idx + 1 < stop and _ODD_RE.match(texts[idx + 1]):
                exact[(int(m.group(2)), int(m.group(4)))] = float(texts[idx + 1])
                idx += 2
            else:
                idx += 1

    return c1, cN, c2, exact, qa_home, qa_away


# ----------------------------------------------------------------------------
# Parsing d'une page Winamax (crowds des scores exacts) — 2 formats
# ----------------------------------------------------------------------------
def parse_winamax_page(path):
    """
    Renvoie crowds = {(buts_home, buts_away): pourcentage_int}. Gère deux formats :
      - ancien : bloc « Score exact » … « Moins de sélections » (+ sections 1re
        mi-temps à exclure via la borne de fin) ;
      - compact : triplets bruts « i - j » / cote / « x% » sur tout le document.
    Les équipes viennent du NOM de fichier (home = 1re listée).
    """
    texts = _read_strings(path)
    n = len(texts)

    # Bornes (format ancien). Si « Score exact » absent -> format compact (tout le doc).
    start = next((i for i, t in enumerate(texts)
                  if t.strip().lower() == _WMX_EXACT_START), None)
    if start is None:
        lo, hi = 0, n
    else:
        lo = start + 1
        hi = next((i for i in range(lo, n)
                   if texts[i].strip().lower() in _WMX_EXACT_END), n)

    crowds = {}
    idx = lo
    while idx < hi:
        m = _SCORE_DASH_RE.match(texts[idx])
        if m:
            hg, ag = int(m.group(1)), int(m.group(2))
            for j in range(idx + 1, min(idx + 4, hi)):
                pm = _PCT_RE.match(texts[j])
                if pm:
                    crowds[(hg, ag)] = int(pm.group(1))
                    break
        idx += 1
    return crowds


# ----------------------------------------------------------------------------
# Recherche de ligne CDM + helpers
# ----------------------------------------------------------------------------
def _find_pair(df_cdm, home, away):
    """(row_label, invert) du match {home, away}, ou (None, None)."""
    direct = (df_cdm["team_A"] == home) & (df_cdm["team_B"] == away)
    inv = (df_cdm["team_A"] == away) & (df_cdm["team_B"] == home)
    if direct.any():
        return df_cdm.index[direct][0], False
    if inv.any():
        return df_cdm.index[inv][0], True
    return None, None


def _fmt_odd(x):
    """Float de cote -> chaîne compacte ('5.0'->'5', '19.610'->'19.61')."""
    return f"{float(x):g}"


def _score_csv(hg, ag, invert):
    """Score « team_A-team_B » à partir des buts home/away et de l'inversion."""
    return f"{ag}-{hg}" if invert else f"{hg}-{ag}"


# ----------------------------------------------------------------------------
# Pipeline principal
# ----------------------------------------------------------------------------
def update_all(matchinfo_dir=DATA_DIR / "MatchInfo",
               cdm_csv=DATA_DIR / "CDM_2026.csv",
               scores_csv=DATA_DIR / "exact_scores.csv",
               dry_run=False, add_mode="ask", verbose=True):
    matchinfo_dir = Path(matchinfo_dir)
    df_cdm = pd.read_csv(cdm_csv, dtype=str, keep_default_na=False)
    df_scores = pd.read_csv(scores_csv, dtype=str, keep_default_na=False)
    scores_cols = list(df_scores.columns)
    cdm_teams = set(df_cdm["team_A"]) | set(df_cdm["team_B"])

    # --- 1N2 global : « Cotes pour Football » ---
    all_records = []          # liste de records résolus (home/away CDM, cotes, mtime)
    all_mtime = None
    all_path = next((p for p in matchinfo_dir.glob("*.html")
                     if is_all_matches_file(p.name)), None)
    if all_path is not None:
        all_mtime = all_path.stat().st_mtime
        for rec in parse_all_matches(all_path):
            h = resolve_team(rec["home"], cdm_teams)
            a = resolve_team(rec["away"], cdm_teams)
            if h and a and h != a:
                all_records.append(dict(home=h, away=a, c1=rec["c1"],
                                        cN=rec["cN"], c2=rec["c2"], mtime=all_mtime))
        if verbose:
            print(f"📊 {all_path.name} : {len(all_records)} matchs 1N2.")
    elif verbose:
        print("⚠️  « Cotes pour Football … » introuvable -> 1N2 global ignoré.")
    n1n2_by_pair = {frozenset((r["home"], r["away"])): r for r in all_records}

    # --- Crowds 1N2 MPP : « Mon Petit Prono … » ---
    mpp_by_pair = {}          # frozenset({home,away}) -> record (home, c1, cN, c2 en %)
    mpp_path = next((p for p in matchinfo_dir.glob("*.html") if is_mpp_file(p.name)), None)
    if mpp_path is not None:
        for rec in parse_mpp_crowds(mpp_path):
            h = resolve_team(rec["home"], cdm_teams)
            a = resolve_team(rec["away"], cdm_teams)
            if h and a and h != a:
                mpp_by_pair[frozenset((h, a))] = dict(
                    home=h, away=a, c1=rec["c1"], cN=rec["cN"], c2=rec["c2"])
        if verbose:
            print(f"📊 {mpp_path.name} : {len(mpp_by_pair)} crowds 1N2 MPP.")

    # --- Pages par-match Pinnacle (1N2 + scores exacts), équipes par nom de fichier ---
    match_pages = {}          # frozenset({home,away}) -> record
    for p in sorted(matchinfo_dir.glob("*.html")):
        if is_all_matches_file(p.name) or p.stem.endswith("_Winamax"):
            continue
        t1, t2 = teams_from_pinnacle_filename(p.name)
        if t1 is None:
            continue
        h, a = resolve_team(t1, cdm_teams), resolve_team(t2, cdm_teams)
        if not (h and a) or h == a:
            if verbose:
                print(f"⚠️  {p.name} : équipes non résolues ({t1!r}/{t2!r}) -> ignoré.")
            continue
        c1, cN, c2, exact, qa_home, qa_away = parse_match_page(p)
        match_pages[frozenset((h, a))] = dict(
            home=h, away=a, c1=c1, cN=cN, c2=c2, exact=exact,
            qa_home=qa_home, qa_away=qa_away, mtime=p.stat().st_mtime, src=p.name)
        if verbose:
            tag = "" if c1 is not None or exact else "  (VIDE : sauvegarde non complète ?)"
            print(f"📄 {p.name} : {h} vs {a}, {len(exact)} scores exacts{tag}.")

    # --- Pages Winamax (crowds), équipes par nom de fichier ---
    wmx_pages = {}            # frozenset({home,away}) -> record
    for p in sorted(list(matchinfo_dir.glob("*_Winamax.txt"))
                    + list(matchinfo_dir.glob("*_Winamax.html"))):
        h, a = teams_from_winamax_filename(p, cdm_teams)
        if not (h and a) or h == a:
            if verbose:
                print(f"⚠️  {p.name} (Winamax) : équipes non résolues -> ignoré.")
            continue
        crowds = parse_winamax_page(p)
        wmx_pages[frozenset((h, a))] = dict(home=h, away=a, crowds=crowds,
                                            src=p.name, mtime=p.stat().st_mtime)
        if verbose:
            tag = "" if crowds else "  (VIDE : sauvegarde non complète ?)"
            print(f"📄 {p.name} : {h} vs {a}, {len(crowds)} crowds Winamax{tag}.")

    # --- Vérifications de cohérence des fichiers (avant tout ajout/écriture) ---
    mpp_used = (mpp_path.name, mpp_path.stat().st_mtime) if mpp_path is not None else None
    all_used = (all_path.name, all_mtime) if all_path is not None else None
    future_idx = df_cdm[df_cdm["result"].str.strip() == ""].index
    _coherence_checks(df_cdm, future_idx, match_pages, wmx_pages, mpp_used, all_used)

    # --- Matchs de « Cotes pour Football » ABSENTS de CDM -> demande d'ajout ---
    if all_records:
        unknown = [r for r in all_records if _find_pair(df_cdm, r["home"], r["away"])[0] is None]
        for r in unknown:
            if _confirm_add(r, add_mode, verbose):
                df_cdm = _add_cdm_row(df_cdm, r, verbose)
                cdm_teams |= {r["home"], r["away"]}

    # --- Parcours des matchs FUTURS (result vide) ---
    future = df_cdm[df_cdm["result"].str.strip() == ""]
    if verbose:
        print(f"\n🎯 {len(future)} matchs futurs (result vide) dans CDM.\n")

    n_1n2, n_exact, n_crowd, n_qualif = 0, 0, 0, 0
    qualif_missing = []
    for lbl in future.index:
        team_a, team_b = df_cdm.at[lbl, "team_A"], df_cdm.at[lbl, "team_B"]
        match_id = df_cdm.at[lbl, "match_id"]
        phase = str(df_cdm.at[lbl, "phase"]).strip()
        is_knockout = not phase.lower().startswith("poule_j")
        key = frozenset((team_a, team_b))
        page, glob, wmx = match_pages.get(key), n1n2_by_pair.get(key), wmx_pages.get(key)
        mpp = mpp_by_pair.get(key)
        if page is None and glob is None and wmx is None and mpp is None:
            # Match KO sans aucune donnée -> cote_qualif attendue mais absente.
            if is_knockout and not str(df_cdm.at[lbl, "cote_qualif_A"]).strip():
                qualif_missing.append((match_id, team_a, team_b))
            continue

        # --- cote_qualif_A / cote_qualif_B (matchs de phase finale) ---
        if page is not None and page.get("qa_home") is not None:
            inv_q = page["home"] == team_b
            qa_a, qa_b = ((page["qa_away"], page["qa_home"]) if inv_q
                         else (page["qa_home"], page["qa_away"]))
            df_cdm.at[lbl, "cote_qualif_A"] = _fmt_odd(qa_a)
            df_cdm.at[lbl, "cote_qualif_B"] = _fmt_odd(qa_b)
            n_qualif += 1
            if verbose:
                print(f"  • Match {match_id} ({team_a} vs {team_b}) cote_qualif "
                      f"-> ({_fmt_odd(qa_a)}/{_fmt_odd(qa_b)})  [{page['src']}]")
        elif is_knockout and not str(df_cdm.at[lbl, "cote_qualif_A"]).strip():
            qualif_missing.append((match_id, team_a, team_b))

        # --- Crowds 1N2 du CDM (source : MPP) ---
        if mpp is not None:
            inv_m = mpp["home"] == team_b
            c1m, c2m = (mpp["c2"], mpp["c1"]) if inv_m else (mpp["c1"], mpp["c2"])
            old_cr = (df_cdm.at[lbl, "crowd_1"], df_cdm.at[lbl, "crowd_N"], df_cdm.at[lbl, "crowd_2"])
            df_cdm.at[lbl, "crowd_1"] = f"{c1m / 100:.2f}"
            df_cdm.at[lbl, "crowd_N"] = f"{mpp['cN'] / 100:.2f}"
            df_cdm.at[lbl, "crowd_2"] = f"{c2m / 100:.2f}"
            n_crowd += 1
            if verbose:
                print(f"  • Match {match_id} ({team_a} vs {team_b}) crowd 1N2 {old_cr} -> "
                      f"({c1m/100:.2f}/{mpp['cN']/100:.2f}/{c2m/100:.2f})  [MPP]")

        # --- 1N2 : source la plus récente (mtime) entre page par-match et global ---
        cands = []
        if page is not None and page["c1"] is not None:
            cands.append((page["mtime"], page, page["src"]))
        if glob is not None and glob["c1"] is not None:
            cands.append((glob["mtime"], glob, all_path.name if all_path else "All_Matches"))
        if cands:
            _, rec, src = max(cands, key=lambda x: x[0])
            invert = rec["home"] == team_b
            c1, c2 = (rec["c2"], rec["c1"]) if invert else (rec["c1"], rec["c2"])
            old = (df_cdm.at[lbl, "cote_1"], df_cdm.at[lbl, "cote_N"], df_cdm.at[lbl, "cote_2"])
            df_cdm.at[lbl, "cote_1"] = _fmt_odd(c1)
            df_cdm.at[lbl, "cote_N"] = _fmt_odd(rec["cN"])
            df_cdm.at[lbl, "cote_2"] = _fmt_odd(c2)
            n_1n2 += 1
            if verbose:
                print(f"  • Match {match_id} ({team_a} vs {team_b}) 1N2 {old} -> "
                      f"({_fmt_odd(c1)}/{_fmt_odd(rec['cN'])}/{_fmt_odd(c2)})  [{src}]")

        # --- Scores exacts : cotes Pinnacle + crowds Winamax ---
        pin_scores, wmx_crowds = {}, {}
        if page is not None and page["exact"]:
            inv_p = page["home"] == team_b
            for (hg, ag), cote in page["exact"].items():
                pin_scores[_score_csv(hg, ag, inv_p)] = cote
        if wmx is not None and wmx["crowds"]:
            inv_w = wmx["home"] == team_b
            for (hg, ag), pct in wmx["crowds"].items():
                wmx_crowds[_score_csv(hg, ag, inv_w)] = pct

        if pin_scores or wmx_crowds:
            df_scores, st = _apply_exact_scores(
                df_scores, scores_cols, match_id, pin_scores, wmx_crowds,
                have_pin=bool(pin_scores))
            n_exact += 1
            if verbose:
                print(f"      scores exacts : {st['added']} ajoutés, {st['updated']} maj, "
                      f"{st['removed']} supprimés, {st['kept_crowd']} conservés ; "
                      f"{st['crowds_set']} crowds Winamax renseignés"
                      f"{'' if pin_scores else ' (cotes préservées : pas de données Pinnacle)'}.")

    # cote_qualif absente pour des matchs de phase finale -> warning.
    if qualif_missing:
        lst = "\n".join(f"  - match {mid} ({ta} vs {tb})" for mid, ta, tb in qualif_missing)
        warnings.warn(
            "cote_qualif_A/B absente pour des matchs de phase finale (marché "
            f"« Pari sur le vainqueur – Qualification » introuvable) :\n{lst}", stacklevel=2)

    # Finalisation de exact_scores : colonnes team_A/team_B (depuis CDM) + tri des
    # scores de chaque match dans l'ordre alphabétique (0-0, 0-1, 0-2, …).
    df_scores = _finalize_scores(df_scores, df_cdm)

    # --- Écriture ---
    if dry_run:
        if verbose:
            print(f"\n🧪 DRY-RUN : aucun fichier écrit ({n_1n2} 1N2, {n_crowd} crowds 1N2, "
                  f"{n_qualif} cote_qualif, {n_exact} matchs exacts).")
    else:
        df_cdm.to_csv(cdm_csv, index=False)
        df_scores.to_csv(scores_csv, index=False)
        if verbose:
            print(f"\n💾 Écrit : {n_1n2} matchs 1N2, {n_crowd} crowds 1N2, "
                  f"{n_qualif} cote_qualif, {n_exact} matchs de scores exacts.")
    return n_1n2, n_crowd, n_qualif, n_exact


def _finalize_scores(df_scores, df_cdm):
    """
    Ajoute/rafraîchit les colonnes team_A/team_B (depuis CDM, par match_id, dupliquées
    sur chaque ligne) juste après match_id, et trie les scores de CHAQUE match dans
    l'ordre alphabétique/numérique (0-0, 0-1, …, 1-0, …). Renvoie le df réordonné.
    """
    team_map = {str(r.match_id): (r.team_A, r.team_B) for r in df_cdm.itertuples()}
    df = df_scores.copy()
    df["team_A"] = df["match_id"].astype(str).map(lambda m: team_map.get(m, ("", ""))[0])
    df["team_B"] = df["match_id"].astype(str).map(lambda m: team_map.get(m, ("", ""))[1])

    # Ordre des colonnes : match_id, team_A, team_B, score, cote, crowd, puis le reste.
    head = ["match_id", "team_A", "team_B", "score", "cote", "crowd"]
    cols = head + [c for c in df.columns if c not in head]
    df = df[cols]

    # Clés de tri : id numérique du match puis (buts_A, buts_B) du score.
    def _score_key(s):
        try:
            h, a = str(s).split("-")
            return int(h), int(a)
        except (ValueError, AttributeError):
            return (999, 999)

    df["_mid"] = pd.to_numeric(df["match_id"], errors="coerce")
    df["_hg"] = df["score"].map(lambda s: _score_key(s)[0])
    df["_ag"] = df["score"].map(lambda s: _score_key(s)[1])
    df = (df.sort_values(["_mid", "_hg", "_ag"], kind="stable")
            .drop(columns=["_mid", "_hg", "_ag"]).reset_index(drop=True))
    return df


def _coherence_checks(df_cdm, future_idx, match_pages, wmx_pages,
                      mpp_used, all_used, warn_minutes=30):
    """
    Trois contrôles de cohérence des fichiers d'entrée. Les checks 1 et 2 ne portent
    que sur les matchs de POULE à venir (phase Poule_J*) : les matchs de phase finale
    n'ont pas forcément de fichier Winamax (Pinnacle seul + marché qualif), ils sont
    donc exemptés de l'appariement et du contrôle de couverture.
      1. Pour un match de poule à venir, page Pinnacle ET fichier Winamax doivent être
         présents tous les deux (sinon ValueError).
      2. Un match de poule à venir sans aucun fichier par-match alors qu'un match de
         poule d'id SUPÉRIEUR en a au moins un -> ValueError (trou de couverture).
      3. mtimes des fichiers UTILISÉS (MPP, All_Matches, et les Pinnacle/Winamax de
         TOUS les matchs à venir) à plus de `warn_minutes` du plus récent -> warning.
    """
    fut, fut_group = [], []
    for lbl in future_idx:
        ta, tb = df_cdm.at[lbl, "team_A"], df_cdm.at[lbl, "team_B"]
        phase = str(df_cdm.at[lbl, "phase"]).strip().lower()
        rec = (int(df_cdm.at[lbl, "match_id"]), frozenset((ta, tb)), ta, tb)
        fut.append(rec)
        if phase.startswith("poule_j"):
            fut_group.append(rec)
    fut.sort()
    fut_group.sort()
    group_pairs = {key for _, key, _, _ in fut_group}

    # --- 1. Pinnacle <-> Winamax appariés (matchs de poule à venir uniquement) ---
    errs = []
    for key in (set(match_pages) | set(wmx_pages)) & group_pairs:
        has_p, has_w = key in match_pages, key in wmx_pages
        if has_p and not has_w:
            teams = f"{match_pages[key]['home']} vs {match_pages[key]['away']}"
            errs.append(f"  - {teams} : Pinnacle « {match_pages[key]['src']} » présent "
                        f"mais fichier Winamax manquant.")
        elif has_w and not has_p:
            teams = f"{wmx_pages[key]['home']} vs {wmx_pages[key]['away']}"
            errs.append(f"  - {teams} : Winamax « {wmx_pages[key]['src']} » présent "
                        f"mais page Pinnacle manquante.")
    if errs:
        raise ValueError(
            "Fichiers Pinnacle/Winamax incohérents (les DEUX sont requis par match) :\n"
            + "\n".join(errs))

    # --- 2. Trou de couverture (id inférieur sans fichier, id supérieur avec) ---
    def _has_file(key):
        return key in match_pages or key in wmx_pages

    covered_ids = [mid for mid, key, _, _ in fut_group if _has_file(key)]
    if covered_ids:
        max_cov = max(covered_ids)
        gaps = [(mid, ta, tb) for mid, key, ta, tb in fut_group
                if mid < max_cov and not _has_file(key)]
        if gaps:
            lst = "\n".join(f"  - match {mid} ({ta} vs {tb})" for mid, ta, tb in gaps)
            raise ValueError(
                f"Couverture incohérente : des matchs à venir d'id < {max_cov} n'ont "
                f"AUCUN fichier alors qu'un match d'id supérieur en a au moins un :\n{lst}")

    # --- 3. Étalement des mtimes des fichiers utilisés (> warn_minutes) ---
    used = {}
    for tup in (mpp_used, all_used):
        if tup is not None:
            used[tup[0]] = tup[1]
    for _, key, _, _ in fut:
        if key in match_pages:
            used[match_pages[key]["src"]] = match_pages[key]["mtime"]
        if key in wmx_pages:
            used[wmx_pages[key]["src"]] = wmx_pages[key]["mtime"]
    if used:
        newest = max(used.values())
        stale = [(name, (newest - m) / 60.0) for name, m in used.items()
                 if (newest - m) > warn_minutes * 60]
        if stale:
            lst = "\n".join(f"  - {name} ({d:.0f} min plus ancien)"
                            for name, d in sorted(stale, key=lambda x: -x[1]))
            warnings.warn(
                f"Fichiers à plus de {warn_minutes} min du plus récent du lot "
                f"(cotes/crowds potentiellement désynchronisés) :\n{lst}", stacklevel=2)


def _confirm_add(rec, add_mode, verbose):
    """Demande (ou décide) l'ajout d'un match inconnu au CDM."""
    label = f"{rec['home']} vs {rec['away']} (1N2 {_fmt_odd(rec['c1'])}/" \
            f"{_fmt_odd(rec['cN'])}/{_fmt_odd(rec['c2'])})"
    if add_mode == "all":
        if verbose:
            print(f"➕ Ajout (mode all) : {label}")
        return True
    if add_mode == "skip":
        if verbose:
            print(f"⏭️  Ignoré (mode skip) : {label}")
        return False
    try:
        ans = input(f"➕ Match absent du CDM : {label}\n   L'ajouter ? [o/N] ").strip().lower()
    except EOFError:
        ans = ""
    return ans in ("o", "oui", "y", "yes")


def _add_cdm_row(df_cdm, rec, verbose):
    """Ajoute une ligne CDM minimale (id, équipes, cotes ; reste à compléter)."""
    cols = list(df_cdm.columns)
    ids = pd.to_numeric(df_cdm["match_id"], errors="coerce")
    new_id = int(ids.max()) + 1 if ids.notna().any() else 1
    row = dict.fromkeys(cols, "")
    row.update({"match_id": str(new_id), "team_A": rec["home"], "team_B": rec["away"],
                "cote_1": _fmt_odd(rec["c1"]), "cote_N": _fmt_odd(rec["cN"]),
                "cote_2": _fmt_odd(rec["c2"])})
    if verbose:
        print(f"   -> ajouté match_id={new_id} ({rec['home']} vs {rec['away']}). "
              f"⚠️ Compléter date/gain_mpp/crowd/phase à la main.")
    return pd.concat([df_cdm, pd.DataFrame([row], columns=cols)], ignore_index=True)


def _apply_exact_scores(df_scores, cols, match_id, pin_scores, wmx_crowds=None,
                        have_pin=True):
    """
    Met à jour les scores exacts du `match_id` : cotes <- `pin_scores`, crowds <-
    `wmx_crowds`. Ensemble final = union(Pinnacle ∪ Winamax ∪ existants à crowd).
    Le crowd Winamax PRIME ; pts/bet/result préservés. Bloc réinséré en place.

    `have_pin` : True si des cotes Pinnacle sont fournies (source autoritaire des
    cotes) -> les scores disparus sans crowd sont SUPPRIMÉS et les cotes des scores
    absents de Pinnacle vidées. Si False (pas de données Pinnacle), les cotes
    existantes sont CONSERVÉES (on ne met à jour que les crowds).
    Renvoie (df_scores_modifié, stats).
    """
    mid = str(match_id)
    wmx_crowds = wmx_crowds or {}
    rows = df_scores.to_dict("records")

    ex_by_score = {}
    for r in rows:
        if str(r["match_id"]) != mid:
            continue
        s = r["score"]
        if s not in ex_by_score or str(r.get("crowd", "")).strip():
            ex_by_score[s] = r

    def _has_crowd(r):
        return bool(str(r.get("crowd", "")).strip())

    # Ordre : Pinnacle (HTML), puis Winamax-only, puis existants restants.
    final_scores, seen = list(pin_scores.keys()), set(pin_scores)
    for s in list(wmx_crowds.keys()):
        if s not in seen:
            final_scores.append(s); seen.add(s)
    for s, r in ex_by_score.items():
        keep = _has_crowd(r) or not have_pin   # sans données Pinnacle, on garde tout
        if s not in seen and keep:
            final_scores.append(s); seen.add(s)

    stats = dict(added=0, updated=0, removed=0, kept_crowd=0, crowds_set=0)
    if have_pin:
        for s, r in ex_by_score.items():
            if s not in pin_scores and s not in wmx_crowds and not _has_crowd(r):
                stats["removed"] += 1

    block = []
    for s in final_scores:
        base = dict.fromkeys(cols, "")
        if s in ex_by_score:
            base.update({k: ex_by_score[s].get(k, "") for k in cols})
        base["match_id"], base["score"] = mid, s
        if s in pin_scores:
            base["cote"] = _fmt_odd(pin_scores[s])
            stats["updated" if s in ex_by_score else "added"] += 1
        elif have_pin:
            base["cote"] = ""        # Pinnacle autoritaire : score disparu / Winamax-only
            if s not in wmx_crowds:
                stats["kept_crowd"] += 1
        # else : cote existante conservée (base déjà copiée)
        if s in wmx_crowds:
            base["crowd"] = str(wmx_crowds[s])
            stats["crowds_set"] += 1
        block.append(base)

    out, inserted = [], False
    for r in rows:
        if str(r["match_id"]) == mid:
            if not inserted:
                out.extend(block); inserted = True
        else:
            out.append(r)
    if not inserted:
        out.extend(block)
    return pd.DataFrame(out, columns=cols), stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matchinfo", default=str(DATA_DIR / "MatchInfo"))
    parser.add_argument("--cdm", default=str(DATA_DIR / "CDM_2026.csv"))
    parser.add_argument("--scores", default=str(DATA_DIR / "exact_scores.csv"))
    parser.add_argument("--dry-run", action="store_true", help="Aperçu sans écrire.")
    parser.add_argument("--add-mode", choices=("ask", "skip", "all"), default="ask",
                        help="Matchs inconnus : demander (ask), ignorer (skip), tout ajouter (all).")
    args = parser.parse_args()
    update_all(matchinfo_dir=args.matchinfo, cdm_csv=args.cdm, scores_csv=args.scores,
               dry_run=args.dry_run, add_mode=args.add_mode)

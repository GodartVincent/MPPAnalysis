"""
Watcher « feu vert » : lance bracket_simulator.py dès que data/GO.txt est rafraîchi.

Usage : à lancer chez soi (depuis la racine du dépôt) AVANT de partir. Le matin,
depuis le boulot, mettre à jour les CSV sur le Drive PUIS modifier/recréer GO.txt en
TOUT DERNIER. Le watcher détecte le changement de mtime, attend que la synchro Drive
locale soit stable, puis lance le calcul de l'horizon des phases finales.

Durcissements vs version naïve :
  - sys.executable -> utilise le Python du venv (numba/llvmlite/shin présents) ;
  - cwd=PROJECT_DIR -> les chemins relatifs de bracket_simulator (data/...) résolvent ;
  - poll de STABILITÉ (taille+mtime stables N fois) au lieu d'un sleep(60) magique :
    couvre AUSSI les sorties NB23/NB24 qui doivent redescendre du Drive ;
  - garde-fou « fichiers vraiment rafraîchis » (mtime > baseline) ;
  - sortie redirigée vers un .log horodaté (run non surveillé) + exit code rapporté.

⚠️ RAPPELS (cf. CLAUDE.md) :
  - beta : le __main__ de bracket_simulator.py NE passe PAS beta -> défaut 0.89.
    Reporter À LA MAIN le beta calibré (NB23) avant que ce watcher ne se déclenche.
  - alpha : NB24 doit avoir écrit la colonne `alpha` dans le CSV buteurs ET ce CSV
    doit être synchronisé en local, sinon compute_robust_endgame_horizon lève ValueError.
  - Ne créer GO.txt qu'une fois NB23 ET NB24 terminés et synchronisés.
"""

import time
import os
import sys
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA = PROJECT_DIR / "data"
TRIGGER = DATA / "GO.txt"
SCRIPT = PROJECT_DIR / "mpp_project" / "bracket_simulator"
LOG = DATA / f"bracket_run_{time.strftime('%Y%m%d_%H%M%S')}.log"

# Fichiers qui DOIVENT être à jour (synchronisés depuis le Drive) avant le lancement.
REQUIRED = [
    DATA / "CDM_2026.csv",
    DATA / "CDM_2026_group_stage_odds.csv",
    DATA / "CDM_2026_goal_scorer_and_favorite.csv",  # porte `alpha` (NB24) + current_goals
]

# Cadence
POLL_GAP = 30          # s entre deux vérifications du feu vert
STABLE_CHECKS = 3      # nb de mesures identiques consécutives requises
STABLE_GAP = 10        # s entre deux mesures de stabilité


def _stamp() -> str:
    return time.strftime("%H:%M:%S")


def wait_until_stable(paths, checks=STABLE_CHECKS, gap=STABLE_GAP):
    """Bloque jusqu'à ce que (taille, mtime) de tous les fichiers soient identiques
    `checks` fois de suite -> la synchro Drive locale est terminée."""
    last = None
    ok = 0
    while ok < checks:
        try:
            sig = tuple((p.stat().st_size, p.stat().st_mtime) for p in paths)
        except FileNotFoundError:
            ok = 0
            last = None
            time.sleep(gap)
            continue
        ok = ok + 1 if sig == last else 0
        last = sig
        time.sleep(gap)


def main(test_mode=False):
    """Watcher principal. `test_mode` (flag --test) : court-circuite le garde-fou de
    fraîcheur des CSV et la vérif de stabilité, et REMPLACE le lancement de
    bracket_simulator par un simple message -> sert à valider le déclenchement
    (touch GO.txt) sans modifier les CSV ni lancer le calcul lourd (~1h+)."""
    # Crée GO.txt s'il n'existe pas, et mémorise son mtime de référence.
    if not TRIGGER.exists():
        TRIGGER.write_text("init", encoding="utf-8")
    baseline = TRIGGER.stat().st_mtime

    mode = " [MODE TEST]" if test_mode else ""
    print(f"[{_stamp()}] En attente du feu vert via {TRIGGER} ...{mode}")
    print(f"           (baseline mtime = {baseline:.0f})")
    while True:
        try:
            current = TRIGGER.stat().st_mtime
            if current > baseline:
                print(f"[{_stamp()}] FEU VERT DÉTECTÉ.")

                if test_mode:
                    print(f"[{_stamp()}] ✅ Déclenchement OK (mode test).")
                    # MISE A JOUR DE LA BASELINE POUR NE PAS BOUCLER A L'INFINI
                    baseline = current 
                    break

                print(f"[{_stamp()}] Vérification de la synchro Drive...")
                wait_until_stable(REQUIRED)

                stale = [p.name for p in REQUIRED if p.stat().st_mtime <= baseline]
                if stale:
                    print(f"[{_stamp()}] ⚠️ Fichiers non rafraîchis : {stale} — j'attends encore.")
                    time.sleep(POLL_GAP)
                    continue

                print(f"[{_stamp()}] Synchro stable. Lancement de bracket_simulator.")
                print(f"           Interpréteur : {sys.executable}")
                print(f"           Sortie -> {LOG}")
                
                with open(LOG, "w", encoding="utf-8") as f:
                    result = subprocess.run(
                        [sys.executable, "-m", "mpp_project.bracket_simulator"], # UTILISER sys.executable ICI
                        cwd=str(PROJECT_DIR),
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        # Ajout optionnel mais recommandé: timeout en secondes (ex: 3 heures)
                        # timeout=10800 
                    )
                status = "OK" if result.returncode == 0 else f"ÉCHEC (exit={result.returncode})"
                print(f"[{_stamp()}] Pipeline lourd terminé : {status}. Détails dans {LOG.name}.")
                
                # MISE A JOUR DE LA BASELINE INDISPENSABLE
                baseline = current 
                break

        except FileNotFoundError:
            pass

        time.sleep(POLL_GAP)


if __name__ == "__main__":
    main(test_mode="--test" in sys.argv)

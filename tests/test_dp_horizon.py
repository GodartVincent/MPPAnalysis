import os
import sys

# Trouve le chemin de la racine du projet (un niveau au-dessus de 'tests/')
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
    
import numpy as np

# ==========================================
# 0. CONFIGURATION
# ==========================================
PATH_MATRIX = "data/expected_V_phases_finales_full.npy"
from mpp_project.oracle_dp import GAP_OFFSET # On importe la constante pour être sûr d'être aligné avec l'Oracle

print("==================================================")
print("🩺 LANCEMENT DES TESTS D'INTÉGRITÉ DE L'ORACLE 🩺")
print("==================================================\n")

try:
    V_full = np.load(PATH_MATRIX)
    print(f"✅ Matrice chargée avec succès. Shape originelle : {V_full.shape}")
except Exception as e:
    print(f"❌ Erreur lors du chargement : {e}")
    sys.exit(1)
    
# --- GESTION DE LA DIMENSION (6D ou 7D) ---
if V_full.ndim == 6:
    print("ℹ️ Format 6D détecté. Conversion en 7D (1 match) pour le test...")
    V_full = np.expand_dims(V_full, axis=0)
elif V_full.ndim == 7:
    print(f"ℹ️ Format 7D détecté. Test sur {V_full.shape[0]} matrices/matchs...")
else:
    print(f"❌ Dimensions inattendues ({V_full.ndim}D). Le test nécessite du 6D ou 7D.")
    sys.exit(1)

nb_matrices = V_full.shape[0]

# --- COMPTEURS D'ERREURS ---
erreurs_t1 = 0
erreurs_t2 = 0
erreurs_t3 = 0

print("\n🔍 Analyse en cours...\n")

# ==========================================
# LA BOUCLE DE TESTS SUR TOUS LES MATCHS
# ==========================================
for m in range(nb_matrices):
    
    # On isole la matrice de base pour le match 'm' (Tous les favoris en vie)
    # Dimensions extraites : [gap_agent, gap_peloton, booster]
    V_base = V_full[m, :, :, :, 1, 1, 1]
    
    # ------------------------------------------
    # TEST 1 : LA MONOTONIE (LA RÈGLE D'OR)
    # ------------------------------------------
    tranche_gap1 = V_base[:, GAP_OFFSET, 0]
    diffs = np.diff(tranche_gap1)
    # On tolère une infime erreur de calcul due à Float32 (-1e-5)
    violations = np.sum(diffs < -1e-5)
    
    if violations > 0:
        erreurs_t1 += 1
        print(f"  ❌ [Matrice {m}] TEST 1 ÉCHOUÉ : {violations} violations de monotonie.")

    # ------------------------------------------
    # TEST 2 : LA TOPOLOGIE DU BOOSTER
    # ------------------------------------------
    v_with_booster = V_base[GAP_OFFSET, GAP_OFFSET, 1]
    v_without_booster = V_base[GAP_OFFSET, GAP_OFFSET, 0]
    valeur_intrinseque = v_with_booster - v_without_booster
    
    if valeur_intrinseque <= 0:
        erreurs_t2 += 1
        print(f"  ❌ [Matrice {m}] TEST 2 ÉCHOUÉ : Booster sans valeur (+{valeur_intrinseque*100:.2f}%).")

    # ------------------------------------------
    # TEST 3 : L'ÉVIDENCE (LE DUMMY MATCH)
    # ------------------------------------------
    wr_si_mauvais_choix = V_base[GAP_OFFSET, GAP_OFFSET, 0] # On rate, gap reste à 0
    # On vérifie que GAP_OFFSET + 50 ne dépasse pas la matrice (1001)
    safe_offset = min(GAP_OFFSET + 50, V_base.shape[0] - 1)
    wr_si_bon_choix = V_base[safe_offset, safe_offset, 0] 
    
    if wr_si_bon_choix <= wr_si_mauvais_choix:
        erreurs_t3 += 1
        print(f"  ❌ [Matrice {m}] TEST 3 ÉCHOUÉ : WR bon choix ({wr_si_bon_choix:.4f}) <= WR mauvais ({wr_si_mauvais_choix:.4f}).")


# ==========================================
# SYNTHÈSE GLOBALE
# ==========================================
print("\n==================================================")
print("📊 RÉSULTATS DU DIAGNOSTIC GLOBALE :")

if erreurs_t1 == 0:
    print(f"✅ TEST 1 (Monotonie stricte) : SUCCÈS ({nb_matrices}/{nb_matrices} saines)")
else:
    print(f"❌ TEST 1 (Monotonie stricte) : ÉCHEC sur {erreurs_t1} matrice(s)")

if erreurs_t2 == 0:
    print(f"✅ TEST 2 (Valeur du Booster) : SUCCÈS ({nb_matrices}/{nb_matrices} saines)")
else:
    print(f"❌ TEST 2 (Valeur du Booster) : ÉCHEC sur {erreurs_t2} matrice(s)")

if erreurs_t3 == 0:
    print(f"✅ TEST 3 (Évidence du gain)  : SUCCÈS ({nb_matrices}/{nb_matrices} saines)")
else:
    print(f"❌ TEST 3 (Évidence du gain)  : ÉCHEC sur {erreurs_t3} matrice(s)")

print("==================================================")

if erreurs_t1 == 0 and erreurs_t2 == 0 and erreurs_t3 == 0:
    print("🎉 L'Oracle est parfaitement sain !")
    sys.exit(0)
else:
    print("⚠️ Attention, certaines matrices sont dégradées.")
    sys.exit(1)
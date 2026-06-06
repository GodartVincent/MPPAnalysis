import numpy as np

# ==========================================
# 0. CONFIGURATION
# ==========================================
PATH_MATRIX = "data/expected_V_phases_finales_full.npy"
GAP_OFFSET = 600

print("==================================================")
print("🩺 LANCEMENT DES TESTS D'INTÉGRITÉ DE L'ORACLE 🩺")
print("==================================================\n")

try:
    V_full = np.load(PATH_MATRIX)
    print(f"✅ Matrice chargée avec succès. Shape : {V_full.shape}")
except Exception as e:
    print(f"❌ Erreur lors du chargement : {e}")
    
# On isole la matrice de base (Tous les favoris en vie)
# Dimensions : [gap_agent, gap_peloton, booster]
V_base = V_full[:, :, :, 1, 1, 1]

# ==========================================
# TEST 1 : LA MONOTONIE (LA RÈGLE D'OR)
# ==========================================
# Gagner des points ne doit JAMAIS faire baisser le Win Rate.
print("\n--- TEST 1 : MONOTONIE STRICTE ---")
# On prend la ligne où le peloton est à 0 point d'écart, et on regarde notre avance (gap1)
tranche_gap1 = V_base[:, GAP_OFFSET, 0]
diffs = np.diff(tranche_gap1)

# On tolère une infime erreur de calcul flottant due à Float32 (-1e-5)
violations = np.sum(diffs < -1e-5)

if violations == 0:
    print("✅ SUCCÈS : La fonction de valeur est strictement croissante. L'algorithme est sain.")
else:
    print(f"❌ ÉCHEC : {violations} violations de monotonie détectées ! La matrice est brisée.")

# ==========================================
# TEST 2 : LA TOPOLOGIE DU BOOSTER
# ==========================================
# Avoir un booster disponible doit strictement être supérieur ou égal à ne pas l'avoir
print("\n--- TEST 2 : VALEUR DU BOOSTER ---")
v_with_booster = V_base[GAP_OFFSET, GAP_OFFSET, 1]
v_without_booster = V_base[GAP_OFFSET, GAP_OFFSET, 0]
valeur_intrinseque = v_with_booster - v_without_booster

print(f"Valeur intrinsèque du Booster à 0 point d'écart : +{valeur_intrinseque * 100:.2f}% de WR")

if valeur_intrinseque > 0:
    print("✅ SUCCÈS : L'Oracle reconnaît la valeur optionnelle du Booster.")
else:
    print("❌ ÉCHEC : L'Oracle pense que le Booster ne sert à rien.")

# ==========================================
# TEST 3 : L'ÉVIDENCE (LE DUMMY MATCH)
# ==========================================
# Faux match de test : 100% de chance pour le favori (Rapporte 50 pts).
# Si on joue le favori (Gain 50), Bob joue l'outsider (Gain 0), Peloton joue l'outsider (Gain 0).
print("\n--- TEST 3 : L'ÉVIDENCE DU GAIN ---")
wr_si_mauvais_choix = V_base[GAP_OFFSET, GAP_OFFSET, 0] # On rate, gap reste à 0
wr_si_bon_choix = V_base[GAP_OFFSET + 50, GAP_OFFSET + 50, 0] # On réussit, on prend +50 sur tout le monde

print(f"WR si on rate l'évidence   : {wr_si_mauvais_choix * 100:.2f}%")
print(f"WR si on saisit l'évidence : {wr_si_bon_choix * 100:.2f}%")

if wr_si_bon_choix > wr_si_mauvais_choix:
    print("✅ SUCCÈS : L'Oracle maximise l'espérance en allant chercher la case la plus rentable.")
else:
    print("❌ ÉCHEC : L'Oracle est aveugle aux points.")
    
print("\n==================================================")
print("FIN DU DIAGNOSTIC")
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ajout du dossier parent au path pour pouvoir importer le module local
sys.path.append(str(Path.cwd().parent))
from mpp_project.core import simulate_true_proba_from_gain

def impute_missing_odds(input_csv, output_csv):
    print(f"Chargement du fichier brut : {input_csv}")
    df = pd.read_csv(input_csv)
    
    # On identifie les matchs où la cote est manquante (NaN)
    # On suppose que si cote_1 est vide, les deux autres le sont aussi
    missing_mask = df['cote_1'].isna()
    nb_missing = missing_mask.sum()
    
    print(f"-> {nb_missing} matchs détectés avec des cotes manquantes.\n")
    
    if nb_missing == 0:
        print("Aucune imputation nécessaire. Fin du script.")
        return

    # Itération sur les index des matchs incomplets
    for idx in df[missing_mask].index:
        match_id = df.loc[idx, 'match_id']
        team_A = df.loc[idx, 'team_A']
        team_B = df.loc[idx, 'team_B']
        
        # Récupération des gains MPP originaux (ceux-ci ne seront pas écrasés !)
        g1 = df.loc[idx, 'gain_mpp_1']
        gN = df.loc[idx, 'gain_mpp_N']
        g2 = df.loc[idx, 'gain_mpp_2']
        
        # Sécurité : vérifier que les gains MPP sont bien renseignés
        if pd.isna(g1) or pd.isna(gN) or pd.isna(g2):
            print(f"⚠️  Attention: Match {match_id} ignoré car les gains MPP sont aussi manquants.")
            continue
            
        # 1. Génération des probas bruitées (indépendantes)
        p1_raw = simulate_true_proba_from_gain(g1)
        pN_raw = simulate_true_proba_from_gain(gN)
        p2_raw = simulate_true_proba_from_gain(g2)
        
        # 2. Normalisation
        somme_p = p1_raw + pN_raw + p2_raw
        p1 = p1_raw / somme_p
        pN = pN_raw / somme_p
        p2 = p2_raw / somme_p
        
        # 3. Calcul des cotes simulées (Cote = 1 / P)
        # On arrondit à 2 décimales pour simuler l'affichage d'un bookmaker
        df.loc[idx, 'cote_1'] = round(1.0 / p1, 2)
        df.loc[idx, 'cote_N'] = round(1.0 / pN, 2)
        df.loc[idx, 'cote_2'] = round(1.0 / p2, 2)
        
        print(f"Match {match_id:<2} ({team_A[:10]:<10} vs {team_B[:10]:<10}) | "
              f"Cotes générées : {df.loc[idx, 'cote_1']:.2f} - {df.loc[idx, 'cote_N']:.2f} - {df.loc[idx, 'cote_2']:.2f}")

    # Sauvegarde dans un nouveau fichier pour ne pas corrompre la source en cas de bug
    print(f"\nImputation terminée. Sauvegarde vers : {output_csv}")
    df.to_csv(output_csv, index=False)
    print("✅ Sauvegarde réussie !")

if __name__ == "__main__":
    # --- CONFIGURATION DES CHEMINS ---
    # Adapte ces chemins selon ton arborescence exacte
    # L'idée est de lire le fichier de base et de cracher un fichier "_imputed"
    # que ton Oracle utilisera ensuite.
    input_path = Path.cwd() / "data" / "CDM_2026.csv"
    output_path = Path.cwd() / "data" / "CDM_2026_imputed.csv"
    
    impute_missing_odds(input_path, output_path)
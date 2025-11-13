import itertools
import numpy as np
import matplotlib.pyplot as plt
from math import comb

# Calcul de la distribution d'une loi binomiale avec une liste de probas de succes
fact = 0.085
# Liste des probabilites de succes pour chaque match
probaSuccess = np.array([1*fact, 1*fact, 1*fact, 1*fact])

nbMatch = len(probaSuccess)
matchIndices = np.array(range(nbMatch))
probas = np.zeros(nbMatch+1)
for successNb in range(nbMatch+1):
    successCombinations = itertools.combinations(matchIndices, successNb)
    print(f"{successNb=} : {len(successCombinations)} combinaisons")
    # Calcul, pour chaque combinaison, de sa probabilite
    for curSuccessCombination in successCombinations:
        curProba = 1
        for matchIndex in matchIndices:
            # Si le match fait partie des succes
            if matchIndex in curSuccessCombination:
                curProba *= probaSuccess[matchIndex]
            # Sinon
            else:
                curProba *= 1-probaSuccess[matchIndex]
        probas[successNb] += curProba

# Affichage de l'histogramme des probabilites
plt.bar(np.array(range(nbMatch+1)), probas, width=1)
plt.title("Distribution binomiale avec probas de succes variables")
plt.xlabel("Nombre de succes")
plt.ylabel("Probabilite")
plt.show()
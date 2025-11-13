
import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt

# Comparaison de la proba d'avoir un gain > a une valeur en fonction du nombre de matchs pour une proba et un ev
matchNbMin = 1
matchNbMax = 51
verbose = False
# L'evenement 0 designe celui sur lequel on parie, l'evenement 1 l'alternative privilegiee
p0 = 0.45652173913043476
g0 = 92
p1 = 0.2753623188405797
g1 = 138
# Le nombre de points que l'on veut obtenir en moyenne par match pour atteindre l'objectif
objPtsPerMatch = 46

print("EV 0 =", p0*g0)
print("EV 1 =", p1*g1)

# Calcul de la probabilite d'atteindre l'objectif en fonction du nombre de succes minimum requis
def getProbaWin(ai_successNbMin, ai_matchNb, ai_proba):
    ao_proba = 0
    # On cumule les proba d'avoir ai_successNbMin succes jusqu'a ai_matchNb succes
    for k in range(ai_successNbMin, ai_matchNb+1):
        ao_proba += binom.pmf(k, ai_matchNb, ai_proba)
    return ao_proba

# Pour tous les nombres de matchs possibles
sampleNb = matchNbMax-matchNbMin+1
matchNbArray = np.linspace(matchNbMin, matchNbMax, sampleNb, dtype=int)
probaWin0 = np.zeros(sampleNb)
probaWin1 = np.zeros(sampleNb)
for idx, matchNb in enumerate(matchNbArray):
    # L'objectif global de points
    totalScoreObj = objPtsPerMatch * matchNb

    # On atteint l'objectif si notre nombre de succes est d'au moins successNbMin0
    successNbMin0 = int(np.ceil(totalScoreObj / g0))
    successNbMin1 = int(np.ceil(totalScoreObj / g1))

    # Calcul de la probabilite d'atteinte de l'objectif selon bet 0 et bet 1
    probaWin0[idx] = getProbaWin(successNbMin0, matchNb, p0)
    probaWin1[idx] = getProbaWin(successNbMin1, matchNb, p1)

    if verbose:
        print("0 : p success one match =", p0, "| gain =", g0, "| success nb =", successNbMin0, "| proba win =", probaWin0[idx])
        print("1 : p success one match =", p1, "| gain =", g1, "| success nb =", successNbMin1, "| proba win =", probaWin1[idx])

plt.plot(matchNbArray, probaWin0, label="Proba victoire bet 0")
plt.plot(matchNbArray, probaWin1, label="Proba victoire bet 1")
plt.title("Comparaison des probabilites de depasser l'objectif en fonction du nombre de matchs")
plt.xlabel("Nombre de matchs")
plt.ylabel("Probabilite de depasser l'objectif")
plt.legend()
plt.show()



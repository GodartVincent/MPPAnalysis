
import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt

# Comparaison de la proba d'avoir un gain > a une valeur en fonction de 2 probas de succes a ev differentes
sampleNb = 500
matchNb = 20
verbose = False
# L'evenement 1 designe celui sur lequel on parie, l'evenement 2 son complementaire
# L'esperance de l'evenement complementaire (ev2) est ev2Ev1Ratio fois l'esperance de l'evenement principal (ev1)
ev2Ev1Ratio = 0.917881
probaMin = 0.1
probaMax = 0.9
# Le nombre de points moyen sur un match que l'on veut obtenir est evObjEv1Ratio fois l'esperance de l'evenement principal
# Autrement dit, a combien de fois notre esperance faut il etre pour atteindre l'objectif
evObjEv0Ratio = 1.08947

# L'objectif global de points
totalScoreObj = evObjEv0Ratio*matchNb

# Calcul de la probabilite d'atteindre l'objectif en fonction du nombre de succes minimum requis
def getProbaWin(ai_successNbMin, ai_matchNb, ai_proba):
    ao_proba = 0
    # On cumule les proba d'avoir ai_successNbMin succes jusqu'a ai_matchNb succes
    for k in range(ai_successNbMin, ai_matchNb):
        ao_proba += binom.pmf(k, ai_matchNb, ai_proba)
    return ao_proba

probaSuccesOneMatchArray = np.linspace(probaMin, probaMax, sampleNb)
probaWin0 = np.zeros(sampleNb)
probaWin1 = np.zeros(sampleNb)
# Pour toutes les probas de succes possibles
for idx, p in enumerate(probaSuccesOneMatchArray):
    # Calcul du gain 0 pour avoir une esperance de 1
    g0 = 1/p
    # Le gain 1 est construit pour avoir une esperance de ev2Ev1Ratio
    g1 = ev2Ev1Ratio/p

    # On atteint l'objectif (evObjEv1Ratio*matchNb points) si notre nombre de succes est d'au moins successNbMin0
    successNbMin0 = int(np.ceil(totalScoreObj / g0))
    successNbMin1 = int(np.ceil(totalScoreObj / g1))

    # Calcul de la probabilite d'atteinte de l'objectif selon bet 0 et bet 1
    probaWin0[idx] = getProbaWin(successNbMin0, matchNb, p)
    probaWin1[idx] = getProbaWin(successNbMin1, matchNb, p)
    if verbose:
        print("p success one match =", p, "| gain =", g0, "| success nb =", successNbMin0, "| proba win =", probaWin0[idx])

plt.plot(probaSuccesOneMatchArray, probaWin0, label="Proba victoire bet 0")
plt.plot(1-probaSuccesOneMatchArray, probaWin1, label="Proba complementaire")
plt.title("Comparaison des probabilites de depasser l'objectif en fonction de la proba de succes")
plt.xlabel("Probabilite de succes sur un match")
plt.ylabel("Probabilite de depasser l'objectif")
plt.legend()
plt.show()



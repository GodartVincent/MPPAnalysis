import numpy as np
import matplotlib.pyplot as plt

# Simulation de la distribution des scores finaux
nsample = 100000
nplayer = 12
scoreMean = 0
nvalue = 201

# Parametres pour la simulation des matchs
evAvg = 31
evStd = 4
probaAvg = 0.33
probaStd = 0.12
nmatchs = 51

# Simulation de l'ecart-type du score global
def getStd():
    avgStd = 0
    for _ in range(nsample):
        # On simule les cotes et probabilites des matchs
        rand = np.random.normal(0, 1, nmatchs)
        # Plus l'esperance est elevee, plus la probabilite de gagner est elevee (constation empirique)
        ev = rand*evStd + evAvg
        proba = rand*probaStd + probaAvg
        # On borne les probabilites entre 5% et 75%
        proba = np.array([min(max(p, 0.05), 0.75) for p in proba])

        # On simule les paris et resultats des matchs
        avgScore = 0
        score = 0
        for i in range(nmatchs):
            # On fait l'hypothese que 50% de nos paris sont identiques a notre adversaire direct
            sameBet = np.random.uniform(0, 1)
            if sameBet > 0.5:
                avgScore += ev[i]
                # On simule le resultat du match
                realMatch = np.random.uniform(0, 1)
                if realMatch < proba[i]:
                    score += ev[i]/proba[i]
        # Accumulation de l'ecart quadratique a la moyenne
        avgStd += pow(avgScore-score, 2)
    # Calcul de l'ecart-type
    avgStd = np.sqrt(avgStd/nsample)
    print("Std score ", avgStd)
    return avgStd

# scoreStd = getStd()
scoreStd = 100
print(scoreStd)

# Grace a l'ecart-type, on peut simuler la distribution des scores finaux
# On simule nsample scores gagnants parmi nplayer joueurs
ymax = np.zeros(nsample)
for i in range(nsample):
    # Hypothese de loi normale pour les scores finaux
    rand = np.random.normal(scoreMean, scoreStd, nplayer)
    ymax[i] = max(rand)

# initialisation de l'histogramme des scores max
ymax = sorted(ymax)
xvalue = np.linspace(scoreMean-4*scoreStd, scoreMean+4*scoreStd, nvalue)
yvalue = np.zeros(nvalue)

# Calcul de l'histogramme des scores max
curyidx = 0
for i in range(nvalue):
    while curyidx < nsample and ymax[curyidx] < xvalue[i]:
        yvalue[i] += 1
        curyidx += 1

# Recentrage des bins des 4 histogrammes finaux
xValueWidth = (xvalue[1]-xvalue[0])/4
xvalue -= (xvalue[1]-xvalue[0])/2

gaussValue = np.exp(-(((xvalue-scoreMean)/scoreStd)**2)/2)
probaWin = np.ones(nvalue)
# Normalization
gaussValue /= sum(gaussValue)
yvalue /= sum(yvalue)

# Convolution pour obtenir la distribution des differences de scores
deltaValue = np.zeros(nvalue)
for i in range(nvalue):
    for j in range(nvalue):
        curXvalue = xvalue[i] - xvalue[j]
        curXIdx = int(round((curXvalue - xvalue[0]) * nvalue / (xvalue[-1] - xvalue[0])))
        if 0 <= curXIdx < nvalue:
            deltaValue[curXIdx] += gaussValue[i] * yvalue[j]

# Decenter the bins
xvalue += (xvalue[1]-xvalue[0])/2
# index in xvlaue for which x = 0
i0 = int(np.floor(nvalue/2))
print("zero =", xvalue[i0])
for i in range(nvalue):
    probaWin[i] = sum(deltaValue[i:i0])*(-10/min(-10, xvalue[i])) + sum(deltaValue[max(i0, i):])
# Recenter the bins
xvalue -= (xvalue[1]-xvalue[0])/2

xmaxidx = int(np.amax(probaWin))
probamax = max(probaWin)
print("xmax", xvalue[xmaxidx])
print("probamax", probamax)


plt.figure
plt.bar(xvalue-3*xValueWidth/2, deltaValue, width=xValueWidth, color='b', label='Delta')
plt.bar(xvalue-xValueWidth/2, yvalue, width=xValueWidth, color='r', label='Opponents')
plt.bar(xvalue+xValueWidth/2, gaussValue, width=xValueWidth, color='g', label='Me')
plt.bar(xvalue+3*xValueWidth/2, probaWin, width=xValueWidth, color='y', label='Proba win')
plt.legend()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

# Simulation de la competition et calcul de proba de victoire pour differentes strategies
nsample = 545
nplayer = 12
nmatchs = 51
# Calibre les ev : si on a une proba de 0.35, quel ev veut-on ?
evAvg = 35

# Parametres de generation des probas de match nul
drawFactProbaMin = 0.2
drawFactProbaMax = 0.75
# Parametres de generation des probas de victoire de l'outsider
outsiderFactProbaMin = 1/7.5
outsiderFactProbaMax = 1

def generateOutcomeProbas(nMatchs, drawFactProbaMin, drawFactProbaMax, outsiderFactProbaMin, outsiderFactProbaMax):
    # Tableau de n matchs, contenant les probas des 3 outcomes
    ao_oProbas = np.ones((nMatchs, 3), dtype=float)
    for iMatch in range(nMatchs):
        # La premiere proba est pour le favori, puis nul puis outsider
        # On calcul les facteurs multiplicatifs par rapport a la proba du favori puis on normalise pour sommer a 1
        # fact de 0 alors match tres desequilibre, fact de 1 match tres equilibre
        fact = random.uniform(0, 1)
        ao_oProbas[iMatch, 1] = fact * (drawFactProbaMax - drawFactProbaMin) + drawFactProbaMin
        ao_oProbas[iMatch, 2] = fact * (outsiderFactProbaMax - outsiderFactProbaMin) + outsiderFactProbaMin
        # On ordonne par proba decroissante pour l'analyse des resultats
        ao_oProbas[iMatch] = sorted(ao_oProbas[iMatch], reverse=True)/sum(ao_oProbas[iMatch])
    return ao_oProbas

# Valeurs constatees de paires proba_succes => gain :
#     si p de 0.8  =>  48pts (esperance de 38.5pts = 1.1ev) ;
#     si p de 0.35 => 100pts (esperance de 35  pts = 1  ev) ;
#     si p de 0.1  => 175pts (esperance de 17.5pts = 0.5ev).
# On veut donc une fonction qui satifait :
#     f(0.8 ) = 1.1 ;
#     f(0.35) = 1   ;
#     f(0.1 ) = 0.25.
# Modelisation par une fonction polynomiale d'ordre 2 :
# a*x^2 + b*x + c => -2.54x^2 + 3.14x + 0.21

# Genere le gain associe a une proba d'outcome
def generateGainFromOutcomeProba(oProba, ev):
    # si oProba = 1 alors ao_ev = ev
    return (-2.54*np.power(oProba,2) + 3.14*oProba + 0.21)*ev/oProba

# Genere les gains associes aux probas des outcomes de tous les matchs
def generateGainFromOutcomeProbas(oProbas, nMatchs, ev):
    # Tableau de n matchs, contenant les gains des 3 outcomes
    ao_oGains = np.ones((nMatchs, 3))
    for iMatch in range(nMatchs):
        for iOutcome in range(3):
            ao_oGains[iMatch, iOutcome] = generateGainFromOutcomeProba(oProbas[iMatch, iOutcome], ev)
    return ao_oGains

# Genere la repartition des bets des joueurs pour un match (en fonction des probas de succes)
def generateRepartitionFromMatchOutcomeProbas(oProbas):
    outcomeNb = len(oProbas)
    ao_repartition = np.ones(outcomeNb)
    # Indice du favori
    iFavo = np.argmax(oProbas)
    for iOutcome in range(outcomeNb):
        # Les non favoris ont une proportion de bet egale a la moitie de leurs probas
        if iFavo != iOutcome:
            ao_repartition[iOutcome] = oProbas[iOutcome] / 2
        # Le favori a le reste des bets
        else:
            ao_repartition[iOutcome] = (oProbas[iOutcome] + 1) / 2
    return ao_repartition

# Genere la repartition des bets des joueurs pour tous les matchs (en fonction des probas de succes)
def generateRepartitionFromOutcomeProbas(oProbas, nMatchs):
    # Tableau de n matchs, contenant les gains des 3 outcomes
    ao_oGains = np.ones((nMatchs, 3))
    for iMatch in range(nMatchs):
        ao_oGains[iMatch] = generateRepartitionFromMatchOutcomeProbas(oProbas[iMatch])
    return ao_oGains


##############################################################################
# Strategies de choix d'outcome pour un match
##############################################################################
# Les strategies suivantes prennent pour hypothese que les autres joueurs ne s'adaptent pas a l'etat des scores

# Strategie applicable seulement si on est premier (scoreGap >= 0)
def stayAhead(oProbas, oGains, oRepartitions, scoreGap, nMatchs):
    scoreGapPerMatch = scoreGap / nMatchs

    # La proba de se faire gratter des points sur un outcome est proba que l'adversaire bet dessus * proba de l'outcome
    gappingProbas = oRepartitions[0] * oProbas[0]
    # On bet sur l'outcome qui confere un gain superieur a notre avance et qui en maximise la proba de rattrapage
    return np.argmax(gappingProbas * (oGains[0] >= scoreGapPerMatch))

# Strategie applicable seulement si on est en retard par rapport au premier (scoreGap <= 0)
def gapClose(oProbas, oGains, oRepartitions, scoreGap, nMatchs):
    scoreGapPerMatch = scoreGap / nMatchs
    # Si le retard n'est pas rattrapable sur ce match, on bet le gain max
    if scoreGapPerMatch >= max(oGains[0]):
        return np.argmax(oGains[0])
    # Si le retard est rattrapable
    # La proba de gratter des points sur un outcome est proba que l'adversaire ne bet pas dessus * proba de l'outcome
    gappingProbas = (1 - oRepartitions[0]) * oProbas[0]
    # On bet sur l'outcome qui confere un gain superieur a l'objectif en maximisant la proba de rattrapage
    return np.argmax(gappingProbas * (oGains[0] >= scoreGapPerMatch))

# Lorsqu'on s'approche de la fin de la competition, tentative de rattrapper le premier
def stratOneMatchGetBack(stratName, oProbas, oGains, oRepartitions, stratGlobalScore, globalScores, nMatchs, nEndMatchs):
    # Si on est en fin de competition
    if nMatchs <= nEndMatchs:
        scoreGap = max(globalScores) - stratGlobalScore
        # Si on est en tete
        if scoreGap < 0:
            # On bet la combinaison bet+succes la plus probable (l'adversaire n'a pas de strategie de get back)
            return np.argmax(oRepartitions[0]*oProbas[0])
        # On bet sur l'outcome qui confere un gain superieur a l'objectif en maximisant la proba de rattrapage
        return gapClose(oProbas, oGains, oRepartitions, scoreGap, nMatchs)

    else:
        return stratOneMatch(stratName, oProbas, oGains, oRepartitions, stratGlobalScore, globalScores, nMatchs)

# Estime le nombre de points final du premier et essaie de le battre
def stratOneMatchBeatEvPlus(stratName, oProbas, oGains, oRepartitions, stratGlobalScore, globalScores, nMatchs, factEv):
    scoreToBeat = sum(oRepartitions*oProbas)*factEv + globalScores

# Simule plein de fois la fin de la competition avec des guesses aleatoires et renvoie le guess qui gagne le plus souvent
def stratOneMatchSimulateEndCompet(oProbas, oGains, oRepartitions, stratGlobalScore, globalScores, nPlayers, nMatchs, nSamples):
    nFirst = np.zeros(3, dtype=int)
    nIter = nSamples*nMatchs
    # On ne calcule que si c'est possible de rattraper le premier
    if sum(oGains.max(1)) + stratGlobalScore >= max(globalScores):
        # Generation aleatoire de nIter fin de tournois
        for iSample in range(nIter):
            opponentGuesses = simulate(oRepartitions, nPlayers, nMatchs)
            curOutcomes = simulate(oProbas, 1, nMatchs).ravel()
            newScores = evaluateScores(curOutcomes, opponentGuesses, oGains, nPlayers, nMatchs)
            newScores[0] += globalScores
            cumulativeNewScores = np.cumsum(newScores, axis=0)
            cumulativeMaxNewScores = cumulativeNewScores.max(1)

            # Pour une fin de tournois possible on simule 3 max ev, une qui commence par 0, l'autre par 1 et la derniere par 2
            for guess in range(3):
                globalStratScore = stratGlobalScore + evaluateOneMatchScore(curOutcomes[0], guess, oGains[0])

                # Calcul des autres bets selon la strategie max ev
                for iMatch in range(1, nMatchs):
                    stratGuess = stratOneMatch("get back 50 max ev",
                                               oProbas[iMatch:],
                                               oGains[iMatch:],
                                               oRepartitions[iMatch:],
                                               globalStratScore,
                                               cumulativeNewScores[iMatch-1],
                                               nMatchs-iMatch,
                                               nPlayers=nPlayers
                                               )
                    globalStratScore += evaluateOneMatchScore(curOutcomes[iMatch], stratGuess, oGains[iMatch])
                if globalStratScore >= cumulativeMaxNewScores[-1]:
                    nFirst[guess] += 1
            # Si le max n'est plus rattrapable au vu du nombre d'iterations restantes on s'arrete
            if iSample > 4*nIter/5:
                sortedNFirst = sorted(nFirst, reverse=True)
                if sortedNFirst[0] - sortedNFirst[1] > nIter - 1 - iSample:
                    break
        # Si il y a egalite sur les 2 meilleurs guess on ne guess pas
        #sortedNFirst = sorted(nFirst, reverse=True) sortedNFirst est deja a la bonne valeur
        # if sortedNFirst[0] == sortedNFirst[1]:
        #     if sortedNFirst[0] == 0:
        #         return 4
        #     return 3
        # Meme si il y a egalite on guess pour ne pas perdre de points
        return np.argmax(nFirst)
    # Si on ne peut pas rattraper le premier, on guess le max ev pour remonter au classement
    return stratOneMatchMaxEv(oProbas[0], oGains[0])

# Lorsqu'on est derriere simule plein de fois la fin de la competition avec des guesses aleatoires et renvoie le guess qui gagne le plus souvent
# Lorsqu'on est devant fait la strat normale
def stratOneMatchBehinSimulateEndCompet(oProbas, oGains, oRepartitions, stratGlobalScore, globalScores, nPlayers, nMatchs, nSamples):
    scoreGap = max(globalScores) - stratGlobalScore
    if scoreGap <= 0:
        return stayAhead(oProbas, oGains, oRepartitions, -scoreGap, nMatchs)
    else:
        return stratOneMatchSimulateEndCompet(oProbas, oGains, oRepartitions, stratGlobalScore, globalScores, nPlayers, nMatchs, nSamples)

def stratOneMatch(stratName, oProbas, oGains, oRepartitions, stratGlobalScore, globalScores, nMatchs, nPlayers=12, nSamples=1000):
    if stratName == "max ev":
        return stratOneMatchMaxEv(oProbas[0], oGains[0])
    elif stratName == "max gain":
        return stratOneMatchMaxGain(oGains[0])
    elif stratName == "opponent":
        return stratOneMatchOpponent(oRepartitions[0])
    elif stratName == "simu end compet":
        return stratOneMatchSimulateEndCompet(oProbas, oGains, oRepartitions, stratGlobalScore, globalScores, nPlayers, nMatchs, nSamples)
    elif stratName == "behin simu end compet":
        return stratOneMatchBehinSimulateEndCompet(oProbas, oGains, oRepartitions, stratGlobalScore, globalScores, nPlayers, nMatchs, nSamples)
    elif len(stratName) >= 9 and stratName[:9] == "get back ":
        split = stratName.split(" ")
        nEndMatchs = int(split[2])
        newStratName = " ".join(split[3:])
        return stratOneMatchGetBack(newStratName, oProbas, oGains, oRepartitions, stratGlobalScore, globalScores, nMatchs, nEndMatchs)

# Bet sur l'outcome avec la meilleur esperance de gain
def stratOneMatchMaxEv(oProbas, oGains):
    return np.argmax(oProbas*oGains)
    
def stratMaxEv(oProbas, oGains, nMatchs):
    return np.array([stratOneMatchMaxEv(probas, gains) for probas, gains in zip(oProbas, oGains)])

# Bet sur l'outcome avec la plus gros gain
def stratOneMatchMaxGain(oGains):
    return np.argmax(oGains)

def stratMaxGains(oProbas, oGains, nMatchs):
    return np.array([stratOneMatchMaxGain(gains) for gains in oGains])

def stratOneMatchOpponent(oRepartitions):
    return simulate([oRepartitions], 1, 1)[0][0]

# Retourne l'indice de xValue dans l'histogramme dont les bornes sont dans xLimits
def findHistIdx(xLimits, xValue):
    nLimits = len(xLimits)
    idx = 0
    while idx+1 < nLimits and xLimits[idx+1] < xValue:
        idx += 1
    return idx


# Simule les outcomes en fonction des probas en entree
def simulate(probas, nTimes, nMatchs):
    ao_guesses = np.zeros((nMatchs, nTimes), dtype=int)
    for iMatch in range(nMatchs):
        for iPlayer in range(nTimes):
            rnd = random.uniform(0, 1)
            if rnd < probas[iMatch][0]:
                ao_guesses[iMatch][iPlayer] = 0
            elif rnd < probas[iMatch][0] + probas[iMatch][1]:
                ao_guesses[iMatch][iPlayer] = 1
            else:
                ao_guesses[iMatch][iPlayer] = 2
    return ao_guesses

# Calcule les scores de tout le monde sur l'ensemble des matchs
def evaluateOneMatchScore(outcome, guess, gains):
    if outcome == guess:
        return gains[outcome]
    return 0

def evaluateScores(outcomes, guesses, gains, nTimes, nMatches):
    scores = np.zeros((nMatches, nTimes))
    for iMatch in range(nMatches):
        for iTime in range(nTimes):
            scores[iMatch][iTime] += evaluateOneMatchScore(outcomes[iMatch], guesses[iMatch][iTime], gains[iMatch])
    return scores

def printResults(stratNames, nTimesFirst, simuGuess, nSimu, simuGuessWhenBehin, simuGuessWhenAhead, w_rankingOc):
    [print(stratName + " : " + str(nTimeFirst / nsample)) for stratName, nTimeFirst in zip(stratNames, nTimesFirst)]
    print("Winning guess proportion :")
    print(simuGuess / nSimu)
    print("When behind :")
    print(sum(simuGuessWhenBehin))
    print("When behind : gap less than favorite gain")
    print(simuGuessWhenBehin[0])
    print("When behind : gap between favorite and second favorite gains")
    print(simuGuessWhenBehin[1])
    print("When behind : gap between second favorite and outsider gains")
    print(simuGuessWhenBehin[2])
    print("When behind : gap greater than outsider gains")
    print(simuGuessWhenBehin[3])

    print()
    print("When ahead :")
    print(sum(simuGuessWhenAhead))
    print("When ahead : gap less than favorite gain")
    print(simuGuessWhenAhead[0])
    print("When ahead : gap between favorite and second favorite gains")
    print(simuGuessWhenAhead[1])
    print("When ahead : gap between second favorite and outsider gains")
    print(simuGuessWhenAhead[2])
    print("When ahead : gap greater than outsider gains")
    print(simuGuessWhenAhead[3])

    print()
    print("Ranking")
    print(w_rankingOc)

# Simulations de plein de tournois
# stratNames = ["max ev", "max gain", "opponent",
#               "get back 3 max ev", "get back 3 max gain", "get back 3 opponent",
#               "get back 5 max ev", "get back 5 max gain", "get back 5 opponent",
#               "get back 10 max ev", "get back 10 max gain", "get back 10 opponent",
#               "get back 50 max ev", "get back 50 max gain", "get back 50 opponent"]
stratNames = ["behind simu end compet"]
simuGuess = np.zeros((nmatchs, 3), dtype=int)
simuGuessWhenAhead = np.zeros((4, nmatchs, 3), dtype=int)
simuGuessWhenBehin = np.zeros((4, nmatchs, 3), dtype=int)
nSimu = 0
nStrats = len(stratNames)
nTimesFirst = np.zeros(nStrats, dtype=int)

histBinNb = 18
histScores = np.linspace(1200, 3000, histBinNb+1)
histOpponentOc = np.zeros(histBinNb, dtype=int)
histOc = np.zeros(histBinNb, dtype=int)

w_ranking = np.array(range(nplayer+1), dtype=int)
w_rankingOc = np.zeros(nplayer+1, dtype=int)
for iSample in tqdm(range(nsample)):
    # Generation des matchs, gains et repartitions des bet des joueurs
    outcomeProbas = generateOutcomeProbas(nmatchs, drawFactProbaMin, drawFactProbaMax, outsiderFactProbaMin, outsiderFactProbaMax)
    gains = generateGainFromOutcomeProbas(outcomeProbas, nmatchs, evAvg)
    repartitions = generateRepartitionFromOutcomeProbas(outcomeProbas, nmatchs)

    # Simulation des bets des joueurs
    guesses = simulate(repartitions, nplayer, nmatchs)
    # Simulation des resultats des matchs
    outcomes = simulate(outcomeProbas, 1, nmatchs).ravel()
    # Calcul des scores des joueurs
    scores = evaluateScores(outcomes, guesses, gains, nplayer, nmatchs)
    curScores = np.zeros(nplayer, dtype=float)
    opponentMaxScore = 0

    # Calcul des bets de chaque strategie, au fil des matchs
    stratGuesses = np.zeros((nmatchs, nStrats), dtype=int)
    stratScores = np.zeros((nmatchs, nStrats), dtype=float)
    globalStratScores = np.zeros(nStrats, dtype=float)
    for iMatch in range(nmatchs):
        for iStrat in range(nStrats):
            stratGuesses[iMatch, iStrat] = stratOneMatch(stratNames[iStrat],
                                                         outcomeProbas[iMatch:],
                                                         gains[iMatch:],
                                                         repartitions[iMatch:],
                                                         globalStratScores[iStrat],
                                                         curScores,
                                                         nmatchs-iMatch,
                                                         nPlayers=nplayer,
                                                         nSamples=540
                                                         )

            # Calcul si le gap avec le premier est en range
            absGap = np.abs(globalStratScores[iStrat] - opponentMaxScore) / (nmatchs - iMatch)
            sortedGains = sorted(gains[iMatch])
            if absGap <= sortedGains[0]:
                iGap = 0
            elif absGap <= sortedGains[1]:
                iGap = 1
            elif absGap <= sortedGains[2]:
                iGap = 2
            else:
                iGap = 3
            if globalStratScores[iStrat] >= opponentMaxScore:
                simuGuessWhenAhead[iGap, iMatch, stratGuesses[iMatch, iStrat]] += 1
            else:
                simuGuessWhenBehin[iGap, iMatch, stratGuesses[iMatch, iStrat]] += 1

            # Ajout des points
            stratScores[iMatch, iStrat] = evaluateOneMatchScore(outcomes[iMatch], stratGuesses[iMatch, iStrat], gains[iMatch])
            globalStratScores[iStrat] += stratScores[iMatch, iStrat]
        # Compte des points des adversaires
        curScores += scores[iMatch]
        opponentMaxScore = max(curScores)

    # Compte du nombre de fois que chaque strat a fini premier
    for iStrat in range(nStrats):
        w_sortedScore = sorted(np.append(curScores, globalStratScores[iStrat]), reverse=True)
        w_rankingOc[w_sortedScore == globalStratScores[iStrat]] += 1
        histOpponentOc[findHistIdx(histScores, opponentMaxScore)] += 1
        histOc[findHistIdx(histScores, globalStratScores[iStrat])] += 1
        if globalStratScores[iStrat] >= opponentMaxScore:
            nTimesFirst[iStrat] += 1
            for iMatch in range(nmatchs):
                simuGuess[iMatch, stratGuesses[iMatch, iStrat]] += 1
            nSimu += 1
    if iSample%50 == 9:
        print()
        print("Iteration " + str(iSample+1))
        printResults(stratNames, nTimesFirst, simuGuess, iSample+1, simuGuessWhenBehin, simuGuessWhenAhead, w_rankingOc)
        print("Opponent =", histOpponentOc)
        print("Strat    =", histOc)

print()
print("################ Resultats finaux : ################")
printResults(stratNames, nTimesFirst, simuGuess, nSimu, simuGuessWhenBehin, simuGuessWhenAhead, w_rankingOc)

print("Opponent =", histOpponentOc)
print("Strat    =", histOc)

# Plot des rangs
plt.bar(w_ranking, w_rankingOc, width=1)
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
plt.title("Occurence des differents rangs", fontsize=24)
plt.ylabel("Occurence", fontsize=20)
plt.xlabel("Rang", fontsize=20)
plt.show()


# Plot des densite de scores
binWidth = histScores[1] - histScores[0]
histScores = histScores[1:] - binWidth/2
plt.bar(histScores - binWidth/4, histOpponentOc, width=binWidth/2, label='Best opponent')
plt.bar(histScores + binWidth/4, histOc, width=binWidth/2, label=stratNames[0])

plt.legend()
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
plt.title("Histogramme des scores finaux", fontsize=24)
plt.ylabel("Occurence", fontsize=20)
plt.xlabel("Score", fontsize=20)
plt.show()
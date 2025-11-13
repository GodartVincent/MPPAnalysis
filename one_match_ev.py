from scipy.stats import binom

# Calcul du gain/esperance en fonction des bets des adversaires
typeOfReward = 'MPP'

n = 12
g1 = 32
g2 = 143
g3 = 179

p1 = 0.26
p2 = 0.3
p3 = 1-p1-p2

c1 = 3.3
c2 = 2.85
c3 = 2.6

invc1 = 1/c1
invc2 = 1/c2
invc3 = 1/c3

gtp1 = invc1 / (invc1+invc2+invc3)
gtp2 = invc2 / (invc1+invc2+invc3)
gtp3 = invc3 / (invc1+invc2+invc3)
print("Probas", gtp1, gtp2, gtp3)

gtb1p1 = 0.24
gtb1p2 = 0.14
gtb1p3 = 0.06

gtb2p1 = 0.135
gtb2p2 = 0.06
gtb2p3 = 0.03

def getExpectedGain(n, k, gtp1, gtp2, gtp3, p2, p3):
    expectedGain = gtp1 * (n - k) * (n + 1) / (k + 1)
    # If someone bet on other outcomes
    probaNoOneBet2 = binom.pmf(0, n - k, p2 / (p2 + p3))
    probaNoOneBet3 = binom.pmf(0, n - k, p3 / (p2 + p3))
    expectedGain -= gtp2 * (1-probaNoOneBet2) * (n+1) + gtp3 * (1-probaNoOneBet3) * (n+1)
    return expectedGain

def getExpectedGainMPP(n, k, gtp1, gtp2, gtp3, p2, p3, g1, g2, g3):
    # Gain if won
    expectedGain = gtp1 * (n - k) * g1
    # Loss if lost
    # Since we consider k people bet on 1, n-k people has bet on something else
    for ko in range(n-k+1):
        # p2/(p2+p3) => proba of a bet on outcome 2, knowing he didn't bet on outcome 1
        proba2 = binom.pmf(ko, n - k, p2/(p2+p3))
        proba3 = binom.pmf(ko, n - k, p3/(p2+p3))

        expectedGain -= proba2 * gtp2 * ko * g2
        expectedGain -= proba3 * gtp3 * ko * g3
    return expectedGain

avgGain1 = 0
avgGain2 = 0
avgGain3 = 0
avgEV1 = 0
avgEV2 = 0
avgEV3 = 0
for k in range(n+1):
    # If I bet on this, what is my expected gain
    if typeOfReward == 'MPP':
        expectedGain1 = getExpectedGainMPP(n, k, gtp1, gtp2, gtp3, p2, p3, g1, g2, g3)
        expectedGain2 = getExpectedGainMPP(n, k, gtp2, gtp1, gtp3, p1, p3, g1, g2, g3)
        expectedGain3 = getExpectedGainMPP(n, k, gtp3, gtp1, gtp2, p1, p2, g1, g2, g3)
    else:
        expectedGain1 = getExpectedGain(n, k, gtp1, gtp2, gtp3, p2, p3)
        expectedGain2 = getExpectedGain(n, k, gtp2, gtp1, gtp3, p1, p3)
        expectedGain3 = getExpectedGain(n, k, gtp3, gtp1, gtp2, p1, p2)

    # Proba
    proba1 = binom.pmf(k, n, p1)
    avgGain1 += proba1*expectedGain1
    proba2 = binom.pmf(k, n, p2)
    avgGain2 += proba2*expectedGain2
    proba3 = binom.pmf(k, n, p3)
    avgGain3 += proba3*expectedGain3

    # EV
    avgEV1 += proba1 * gtp1 * (n + 1) / (k + 1)
    avgEV2 += proba2 * gtp2 * (n + 1) / (k + 1)
    avgEV3 += proba3 * gtp3 * (n + 1) / (k + 1)
    # Just to have global EV, for all players, of 1, when no one has the good result we reimburse the players
    # (does not change anything cause no gap of points is created or erased)
    if k == 0:
        # probaN is the proba of no one has bet on outcome N
        avgEV1 += proba2 * gtp2 * 1 + proba3 * gtp3 * 1
        avgEV2 += proba1 * gtp1 * 1 + proba3 * gtp3 * 1
        avgEV3 += proba1 * gtp1 * 1 + proba2 * gtp2 * 1


print("Gain relatif moyen :")
print(avgGain1)
print(avgGain2)
print(avgGain3)
print("E =", p1*avgGain1+p2*avgGain2+p3*avgGain3)
print()
print("Esperance moyenne :")
print(avgEV1)
print(avgEV2)
print(avgEV3)
print("E =", p1*avgEV1+p2*avgEV2+p3*avgEV3)
print()
extremCaseBonusEsp1 = ((gtb1p1-gtb2p1)+gtb2p1*2)*n
extremCaseBonusEsp2 = ((gtb1p2-gtb2p2)+gtb2p2*2)*n
extremCaseBonusEsp3 = ((gtb1p3-gtb2p3)+gtb2p3*2)*n
print(extremCaseBonusEsp1)
print(extremCaseBonusEsp2)
print(extremCaseBonusEsp3)
print("Worst Bonus Case :", 2*extremCaseBonusEsp1-extremCaseBonusEsp2)

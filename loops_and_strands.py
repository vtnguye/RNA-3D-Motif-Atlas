# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 3:00:00 2021

@author:adamc
"""

from discrepancy import matrix_discrepancy_cutoff
from myTimer import myTimer
from copy import copy
from collections import defaultdict
import numpy as np
from time import time
from query_processing import synonym
from fr3d_configuration import SERVER
from pair_processing import get_pairlist

def getPairTypes(interactions):

    pairTypes = []
    for interaction in interactions:
        interactionType = interaction.split("_")[0]  # in case of _exp or similar
        if interactionType in synonym['stack']:
            pairTypes.append('pairsStacks')
        elif interactionType in synonym['pair']:
            pairTypes.append('pairsStacks')
        elif interactionType in synonym['nstack']:
            pairTypes.append('pairsStacks')
        elif interactionType in synonym['npair']:
            pairTypes.append('pairsStacks')
        elif interactionType in synonym['BPh']:
            pairTypes.append('BPh')
        elif interactionType in synonym['nBPh']:
            pairTypes.append('BPh')
        elif interactionType in synonym['BR']:
            pairTypes.append('BR')
        elif interactionType in synonym['nBR']:
            pairTypes.append('BR')
        elif interactionType in synonym['sO3'] or interactionType in synonym['sO5']:
            pairTypes.append('sO')
        elif interactionType in synonym['nsO3'] or interactionType in synonym['nsO5']:
            pairTypes.append('sO')
        else:
            pairTypes.append('misc')

    return interactions, pairTypes


def lookUpInteractions(indices, pairToInteractions, pairToCrossingNumber, units):
    '''
    store the interactions in each candidate according to type,
    so the different types can be output in order
    '''

    interactions = defaultdict(list)

    for a in range(0, len(indices)):             # first position
        for b in range(0, len(indices)):         # second position
            pair = (indices[a], indices[b])      # pair of indices in the file

            if pair in pairToInteractions:      # there is an interaction between these units
                # it would be nice to calculate crossing number for all pairs, not just interacting pairs
                # input("Inpect. press enter")
                interactions[(a, b, "crossingNumber")].append(str(pairToCrossingNumber[pair]))
                inter, pairTypes = getPairTypes(pairToInteractions[pair])
                for i in range(0,len(inter)):
                    interactions[(a, b, pairTypes[i])].append(inter[i])
                # print("Missing interaction positions %d,%d crossing %d" % (a,b,pairToCrossingNumber[pair]))
    for a in range(0, len(indices)):             # first position
        if "glycosidicBondOrientation" in units[indices[a]]:
            gly = units[indices[a]]["glycosidicBondOrientation"]
            chi = units[indices[a]]["chiDegree"]
            if len(gly) > 0:
                interactions[(a, a, "glycosidicBondOrientation")].append(gly)
                interactions[(a, a, "chiDegree")].append(str(chi))

    return interactions



def myIntersect(a, b):
    '''intersect two lists, allowing that one or both could be "full",
    which means that there is no known restriction on the list,
    aside from the elements coming from the current universe of
    possible values'''

    if("full" in a):
        return b
    elif("full" in b):
        return a
    elif isinstance(a, set):
        return a.intersection(b)
    else:
        return list(set(a).intersection(set(b)))


def mySetIntersect(a, b):
    '''intersect two sets, allowing that one or both could be "full",
    which means that there is no known restriction on the list,
    aside from the elements coming from the current universe of
    possible values'''

    if a == "full":
        return b
    elif b == "full":
        return a
    else:
        return a & b


def getDistanceError(Q, units, perm, i, j, a, b):
    '''calculate how far off the distance between units a and b is,
    compared to distance between i and j in the query'''

    pair_distance = np.linalg.norm(units[a]["centers"] - units[b]["centers"])
    return (Q["permutedDistance"][i][j] - pair_distance)**2


def makeFullList(universe1, universe2):

    # newList = list(product(universe1,universe2))  
    # the above line includes pairs like (a,a), which should not occur in FR3D
    newList = []
    for a in universe1:
        for b in universe2:
            if not a == b:
                newList.append((a, b))

    return newList


def printListLengths(Q, numpositions, universe, listOfPairs, text=""):

    if not 'server' in Q and not 'motif_atlas' in Q:
        print("Universe sizes followed by list lengths, upper triangle. %s" % text)
        for i in range(0, numpositions):
            line = "%6d" % len(universe[i])
            for j in range(0, numpositions):
                if j < i+1:
                    line  = "        " + line
                elif listOfPairs[i][j] == "full":
                    line += "    full"
                else:
                    line += "%8d" % len(listOfPairs[i][j])
            print(line)


def sameAlternateId(Q, ifedata, possibilities):
    '''screen to make sure that no possibility has units with different alternate id (typically A or B)'''

    toRemove = []
    for i in reversed(range(0, len(possibilities))):
        possibility = possibilities[i]
        alternateIdsFound = set()
        for index in possibility:
            if index in ifedata['index_to_id']:
                fields = ifedata['index_to_id'][index].split("|")
                if len(fields) >= 7:
                    if len(fields[6]) > 0:
                        alternateIdsFound.add(fields[6])

        if len(alternateIdsFound) > 1:
            ids = []
            for index in possibility:
                ids.append(ifedata['index_to_id'][index])
            print("Found multiple alternate ids", ids)
            del possibilities[i]

    return possibilities


def extendFragment(Q, ifedata, perm, currentFragment, secondElementList, possibilityArray,
    numpositions, previousDistanceError=0):
    """
    Starting with a list of m-unit matches which meet the first m pairwise
    constraints, return a list of (m+1)-unit matches that meet the first
    m+1 pairwise constraints.
    """

    numpositions = Q["numpositions"]

    # if the current fragment of a possibility is the full length, return it,
    # nothing more to be added
    if(len(currentFragment) == numpositions):
        return Q, [currentFragment]

    possibilities = []
    n = len(currentFragment) - 1

    # using the last position of currentFragment, reduce the possible lists
    # for the later positions in the motif
    newPossibilityArray = []
    for i in range(n + 1, numpositions):
        if currentFragment[-1] in secondElementList[n][i]:
            choices = mySetIntersect(possibilityArray[i - n],
            # choices for i vertex given current possibilities
            secondElementList[n][i][currentFragment[-1]])
        else:
            choices = set([])

        # if the current list is empty, there is no way to complete the current fragment
        if len(choices) == 0:
            return Q, []

        newPossibilityArray.append(choices)

    # if the next set to be considered is full, there will be way too many matches
    if newPossibilityArray[0] == "full":
        Q["errorMessage"].append("Query is underspecified, halting search. Add more constraints.")
        Q["halt"] = True
        print("Problem: next position to be added is a full list," +
            " which suggests that the query is underspecified")
        return Q, []

    # loop through the choices for the next position, that can be added to the current fragment
    if (Q["type"] == "symbolic"):
        for extension in newPossibilityArray[0]:
            if not extension in currentFragment:       # enforce that no nucleotide can be repeated
                Q, new_poss = extendFragment(Q, ifedata, perm, currentFragment + (extension,),
                    secondElementList, newPossibilityArray, numpositions, 0)
                possibilities += new_poss
    else:
        for extension in newPossibilityArray[0]:
            totalDistanceError = previousDistanceError
            # add to the sum of distance errors, and compare to maximum values for that
            for j in range(0, n):
                totalDistanceError  +=  getDistanceError(Q, ifedata["units"], perm, j, n + 1,
                    currentFragment[j], extension)
            if (totalDistanceError <= Q["cutoff"][n+1]):
                Q, new_poss = extendFragment(Q, ifedata, perm, currentFragment + (extension,),
                    secondElementList, newPossibilityArray, numpositions, totalDistanceError)
                possibilities += new_poss

    return Q, possibilities


def getPossibilities(Q, ifedata, perm, secondElementList, numpositions, listOfPairs):
    '''
    intersect lists in listOfPairs to get possibilities which satisfy all pairwise constraints.
    possibilities is a list of
    '''

    # An underspecified query with two positions needs special treatment
    if Q['numpositions'] == 2 and listOfPairs[0][1] == 'full':
        Q["errorMessage"].append("Query is underspecified, halting search. Add more constraints.")
        Q["halt"] = True
        print("Problem: next position to be added is a full list," +
            " which suggests that the query is underspecified")
        return Q, []

    possibilities = []
    possibilityArray = []

    # start with all pairs corresponding to positions 0 and 1
    # for each of these, build complete possibilities
    # possibilityArray is a list of lists of still possible positions later in the fragment being built

    for firstPair in listOfPairs[0][1]:
        possibilityArray = []
        emptyArray = False
        for i in range(1,numpositions):
            if firstPair[0] in secondElementList[0][i]:
                possibilityArray.append(secondElementList[0][i][firstPair[0]])
            else:
                emptyArray = True
        if Q["type"] == "geometric" or Q["type"] == "mixed":
            distanceError = getDistanceError(Q, ifedata['units'], perm, 0, 1, firstPair[0], firstPair[1])
        else:
            distanceError = 0

        if not emptyArray:
            Q, new_poss = extendFragment(Q, ifedata, perm, firstPair, secondElementList,
                possibilityArray, numpositions,  distanceError)
            possibilities += new_poss

    return Q, possibilities


def buildSecondElementList(Q, listOfPairs, universe, ifedata):
    '''turn lists of pairs into lists of second elements of pairs'''
    # does this need Q/ifedata?

    numpositions = len(listOfPairs) + 1
    secondElementList = [0] * (numpositions)

    for i in range(0, numpositions - 1):
        secondElementList[i] = [0] * (numpositions)
        for j in range(i + 1, numpositions):
            if listOfPairs[i][j] == "full":
                secondElementList[i][j] = defaultdict()
                for c in universe[i]:
                    secondElementList[i][j][c] = "full"
            else:
                secondElementList[i][j] = defaultdict(set)
                for c in listOfPairs[i][j]:
                    secondElementList[i][j][c[0]].add(c[1])

    return secondElementList


def permuteListOfPairs(listOfPairs, perm):
    '''return new listOfPairs given a permutation'''

    newListOfPairs = {}
    indexOf = {}
    for i in range(0, len(perm)):
        indexOf[perm[i]] = i

    for i in listOfPairs:
        for j in listOfPairs[i]:
            if indexOf[i] > indexOf[j]:
                if listOfPairs[i][j] == "full":
                    pairs = "full"
                else:
                    pairs = []
                    for pair in listOfPairs[i][j]:
                        pairs.append((pair[1],pair[0]))
                if indexOf[j] in newListOfPairs:
                    newListOfPairs[indexOf[j]][indexOf[i]] = pairs
                else:
                    newListOfPairs[indexOf[j]] = {indexOf[i]:pairs}
            else:
                if indexOf[i] in newListOfPairs:
                    newListOfPairs[indexOf[i]][indexOf[j]] = copy(listOfPairs[i][j])
                else:
                    newListOfPairs[indexOf[i]] = {indexOf[j]:copy(listOfPairs[i][j])}

    return newListOfPairs


def reorderPositions(listOfPairs, universe): #reorder positions
    """
    Reorder positions so that intersecting lists is more efficient
    """

    numpositions = len(listOfPairs) + 1
    listOfPairs_sizes = []
    lengths = np.zeros((numpositions,numpositions))
    for i in listOfPairs:
        for j in listOfPairs[i]:
            if listOfPairs[i][j] == "full":
                L = 1000000000                        # needs to be large but not infinite
                lengths[i][j] = L
                lengths[j][i] = L
            else:
                L = len(listOfPairs[i][j])
                lengths[i][j] = L
                lengths[j][i] = L
            listOfPairs_sizes.append(([i,j], L))
    listOfPairs_sizes = sorted(listOfPairs_sizes, key=lambda x: x[1])
    perm = listOfPairs_sizes[0][0]
    del listOfPairs_sizes
    objects = set(range(0, numpositions))
    objects.remove(perm[0])
    objects.remove(perm[1])
    while len(perm) < numpositions:
        bestLen = float("inf")
        bestCon = -1
        for obj in objects:
            myLen = 1
            for p in perm:
                myLen += lengths[p][obj]
            if myLen < bestLen:
                bestLen = myLen
                bestCon = obj

        perm.append(bestCon)
        objects.remove(bestCon)

    newListOfPairs = permuteListOfPairs(listOfPairs, perm)
    newUniverse = [universe[i] for i in perm]

    return newListOfPairs, newUniverse, perm


def pruneListOfPairs(listOfPairs, universe):
    '''Prunes out impossible pairs from listOfPairs'''

    good_elements = []
    numpositions = len(listOfPairs)+1

    # remove from universes any element that is in no pair
    for i in range(0, numpositions - 1):
        for j in range(i + 1, numpositions):
            if not listOfPairs[i][j] == 'full':
                universe = reduceUniverses(i, j, universe, listOfPairs)

    # convert universes from lists to sets ... they should be defined to be sets in the first place
    universe_set = []
    for i in range(0, numpositions):
        universe_set.append(set(universe[i]))

    # remove pairs with at least one element not in the universe
    for i in range(0, numpositions - 1):
        for j in range(i + 1,numpositions):
            if listOfPairs[i][j] != "full":
                newListOfPairs = []
                for (a, b) in listOfPairs[i][j]:
                    if a in universe_set[i] and b in universe_set[j]:
                        newListOfPairs.append((a, b))
                listOfPairs[i][j] = newListOfPairs

    emptyList = False
    for i in range(0, numpositions):
        if len(universe[i]) == 0:
            emptyList = True

    return listOfPairs, universe, emptyList


def reduceUniverses(i, j, universe, listOfPairs):
    '''loop through listOfPairs[i][j], identify elements corresponding to i,j,
    keep only those in universe[i] and universe[j]'''

    universe_i = []
    universe_j = []
    for (a,b) in listOfPairs[i][j]:
        universe_i.append(a)
        universe_j.append(b)
    universe[i] = myIntersect(universe[i], universe_i)
    universe[j] = myIntersect(universe[j], universe_j)

    return universe


def FR3D_search(Q, ifedata, ifename, timerData):

    # impose pairwise distance constraints for geometric and mixed searches
    timerData = myTimer("Calculating pairwise distances")
    listOfPairs = get_pairlist(Q, ifedata["models"], ifedata["centers"])

    # reduce lists of pairs according to constraints
    timerData = myTimer("Reducing lists")

    numpositions = Q['numpositions']
    IFEStartTime = time()

    interactionToPairs = ifedata['interactionToPairs']
    pairToInteractions = ifedata['pairToInteractions']
    pairToCrossingNumber = ifedata['pairToCrossingNumber']
    index_to_id = ifedata['index_to_id']
    units = ifedata["units"]


    # define the universe for each position in the query
    universe = {}
    for i in range(0, numpositions):
        # universe[i] = [n for n in range(len(units))] # range(0,len(units)) # original line
        universe[i] = ifedata['index_to_id'].keys() # ava change

    printListLengths(Q, numpositions, universe, listOfPairs, "After setting up universes and imposing distance constraints.")

    timerData = myTimer("Unary constraints")

    # use unary constraints to reduce each universe
    if "requiredUnitType" in Q: # might include mixes
        for i in range(0, numpositions):
            if(len(Q["requiredUnitType"][i]) > 0): # nonempty unit type constraint
                temp_universe = []
                for index in universe[i]:
                    if(units[index]['unitType'] in Q["requiredUnitType"][i]):
                        temp_universe.append(index)

                universe[i] = myIntersect(universe[i], temp_universe)

    if "requiredMoleculeType" in Q:
        for i in range(0, numpositions):
            if len(Q["requiredMoleculeType"][i]) > 0: #nonempty molecule type constraint
                temp_universe = []
                for index in universe[i]:
                    if(units[index]["moleculeType"] in Q["requiredMoleculeType"][i]):
                        temp_universe.append(index)
                universe[i] = myIntersect(universe[i], temp_universe)

    if "requiredInteractions" in Q:
        for i in range(0, numpositions):
            if len(Q["requiredInteractions"][i][i]) > 0: # required interaction constraint
                temp_universe = []
                for interaction in Q["requiredInteractions"][i][i]:
                    if interaction == "and":
                        pass
                    elif interaction in interactionToPairs:
                        temp_universe = temp_universe + [a for (a, b) in interactionToPairs[interaction][0]]
                universe[i] = myIntersect(universe[i], temp_universe)

    if "prohibitedInteractions" in Q:
        for i in range(0, numpositions):
            if len(Q["prohibitedInteractions"][i][i]) > 0: # nonempty constraint
                for interaction in Q["prohibitedInteractions"][i][i]:
                    if interaction in interactionToPairs:
                        indices = [a for (a, b) in interactionToPairs[interaction][0]]
                        universe[i] = list(set(universe[i]) - set(indices))

    if "glycosidicBondOrientation" in Q:
        for i in range(0, numpositions):
            if(len(Q["glycosidicBondOrientation"][i]) > 0): # nonempty orientation constraint
                temp_universe = []
                for index in universe[i]:
                    if units[index]["glycosidicBondOrientation"] in Q["glycosidicBondOrientation"][i]:
                        temp_universe.append(index)
                universe[i] = myIntersect(universe[i], temp_universe)

    if "chiAngle" in Q:
        for i in range(0, numpositions):
            if(len(Q["chiAngle"][i]) > 0): # nonempty chi angle constraint
                temp_universe = []
                a = Q["chiAngle"][i][1]
                b = Q["chiAngle"][i][2]
                if Q["chiAngle"][i][0] == 'between':
                    for index in universe[i]:
                        if a <= units[index]["chiDegree"] and units[index]["chiDegree"] <= b:
                            temp_universe.append(index)
                else:
                    for index in universe[i]:
                        if a <= units[index]["chiDegree"] or units[index]["chiDegree"] <= b:
                            temp_universe.append(index)

                universe[i] = myIntersect(universe[i], temp_universe)


    """
    # debugging an RNA-protein query
    for i in range(len(universe)):
        for u in universe[i]:
            print(i,units[u]["moleculeType"],units[u]["unitType"])
    """

    printListLengths(Q, numpositions, universe, listOfPairs, "After unary constraints.")


    emptyList = False

    # reduce list of pairs according to pairwise constraints
    if "requiredInteractions" in Q and numpositions > 1:
        timerData = myTimer("Required constraints")
        for i in range(0, numpositions):
            for j in range(i + 1, numpositions):
                # interactions above the diagonal
                if len(Q["requiredInteractions"][i][j]) > 0:
                    newListOfPairs = "full"
                    tempList = []

                    for interaction in Q["requiredInteractions"][i][j]:
                        # apply previous tempList restrictions
                        if interaction == "and":
                            newListOfPairs = myIntersect(newListOfPairs, tempList)
                            tempList = []
                        elif interaction in interactionToPairs and len(interactionToPairs[interaction]) > 0:
                            if not interaction == "bSS" and "crossingNumber" in Q and Q["crossingNumber"][i][j]:
                                for a in range(0,len(interactionToPairs[interaction][1])):
                                    cn = interactionToPairs[interaction][1][a]  # current crossing number
                                    if not cn == None and Q["crossingNumber"][i][j][0] <= cn and (
                                        cn <= Q["crossingNumber"][i][j][1]):
                                        tempList.append(interactionToPairs[interaction][0][a])
                            else:
                                tempList.extend(interactionToPairs[interaction][0])

                    listOfPairs[i][j] = myIntersect(listOfPairs[i][j], newListOfPairs)

                # below the diagonal, for asymmetric pairs like BPh, BR, sO
                if j in Q["requiredInteractions"] and (
                i in Q["requiredInteractions"][j]) and len(Q["requiredInteractions"][j][i]) > 0:
                    newListOfPairs = []
                    for interaction in Q["requiredInteractions"][j][i]:
                        if interaction in interactionToPairs and len(interactionToPairs[interaction]) > 0:
                            if not interaction == "bSS" and "crossingNumber" in Q and Q["crossingNumber"][i][j]:
                                for a in range(0,len(interactionToPairs[interaction][1])):
                                    cn = interactionToPairs[interaction][1][a]  # current crossing number
                                    if Q["crossingNumber"][i][j][0] <= cn and cn <= Q["crossingNumber"][i][j][1]:
                                        u = interactionToPairs[interaction][0][a][0]
                                        v = interactionToPairs[interaction][0][a][1]
                                        newListOfPairs.append((v, u))
                            else:
                                for a in range(0, len(interactionToPairs[interaction][1])):
                                    u = interactionToPairs[interaction][0][a][0]
                                    v = interactionToPairs[interaction][0][a][1]
                                    newListOfPairs.append((v, u))

                    listOfPairs[i][j] = myIntersect(listOfPairs[i][j], newListOfPairs)

        printListLengths(Q, numpositions, universe, listOfPairs, "After required constraints.")

    timerData = myTimer("Pruning after req constraints")
    listOfPairs, universe, emptyList = pruneListOfPairs(listOfPairs, universe)
    printListLengths(Q, numpositions, universe, listOfPairs, "After pruning after required constraints.")

    listOfPairs, universe, emptyList = pruneListOfPairs(listOfPairs, universe)
    printListLengths(Q, numpositions, universe, listOfPairs, "After pruning again after required constraints.")

    # reduce list of pairs according to continuity constraints
    if not emptyList and "continuityConstraint" in Q and numpositions > 1:
        timerData = myTimer("Continuity constraints")

        # look up chain and symmetry once for each unit, to be used when checking continuity constraints
        modelChainSymmetry = []        # continuity constraints implicitly require same model, chain, symmetry

        for a in range(0, len(ifedata["index_to_id"])):
            fields = ifedata['index_to_id'][a].split("|")
            if len(fields) == 9:
                MCS = fields[1] + "_" + fields[2] + "_" + fields[8]
            else:
                MCS = fields[1] + "_" + fields[2]
            modelChainSymmetry.append(MCS)

        for i in range(0, numpositions):
            for j in range(i + 1, numpositions):
                if len(universe[j]) > 0:
                    if i in Q["continuityConstraint"] and (
                    j in Q["continuityConstraint"][i]) and Q["continuityConstraint"][i][j]:
                        constraint = Q["continuityConstraint"][i][j]
                        #print(i,j,constraint)
                        if constraint[0] == "between":
                            newList = []
                            if listOfPairs[i][j] == "full":
                                universe[i] = sorted(universe[i])
                                universe[j] = sorted(universe[j])
                                starting_n = 0
                                for m in range(0, len(universe[i])):
                                    foundOne = False
                                    a = universe[i][m]
                                    p = ifedata['units'][a]["chainindex"]  # sequence position
                                    n = starting_n
                                    b = universe[j][n]
                                    q = ifedata['units'][b]["chainindex"]  # sequence position
                                    # probe for the first match
                                    while n < len(universe[j]) - 1 and (
                                    modelChainSymmetry[a] != modelChainSymmetry[b] or q - p <= constraint[1]):
                                        n += 1
                                        b = universe[j][n]
                                        q = ifedata['units'][b]["chainindex"]  # sequence position
                                    # accumulate matches
                                    while n < len(universe[j]) and (
                                    modelChainSymmetry[a] == modelChainSymmetry[b]) and (
                                    q - p < constraint[2]) and q - p > constraint[1]:
                                        if a != b:
                                            newList.append((a, b))
                                        if not foundOne:
                                            starting_n = n
                                            foundOne = True
                                        n += 1
                                        if n < len(universe[j]):
                                            b = universe[j][n]
                                            q = ifedata['units'][b]["chainindex"]  # sequence position
                            else:
                                for (a, b) in listOfPairs[i][j]:
                                    p = ifedata['units'][a]["chainindex"]  # sequence position
                                    q = ifedata['units'][b]["chainindex"]  # sequence position

                                    if modelChainSymmetry[a] == modelChainSymmetry[b] and (
                                    q - p > constraint[1]) and q - p < constraint[2]:
                                        newList.append((a, b))
                            listOfPairs[i][j] = newList

                        elif constraint[0] == "equal":
                            newList = []
                            if listOfPairs[i][j] == "full":
                                universe[i] = sorted(universe[i])
                                universe[j] = sorted(universe[j])
                                starting_n = 0
                                for m in range(0, len(universe[i])):
                                    foundOne = False
                                    a = universe[i][m]
                                    p = ifedata['units'][a]["chainindex"]  # sequence position
                                    n = starting_n
                                    b = universe[j][n]
                                    q = ifedata['units'][b]["chainindex"]  # sequence position
                                    # probe for the first match
                                    while n < len(universe[j]) - 1 and (
                                    modelChainSymmetry[a] != modelChainSymmetry[b] or q - p <= constraint[1]):
                                        n += 1
                                        b = universe[j][n]
                                        q = ifedata['units'][b]["chainindex"]  # sequence position
                                    # accumulate matches
                                    while n < len(universe[j]) and (
                                    modelChainSymmetry[a] == modelChainSymmetry[b] and q - p < constraint[2]):
                                        if a != b and q-p in constraint[3]:
                                            newList.append((a, b))
                                        if not foundOne:
                                            starting_n = n
                                            foundOne = True
                                        n += 1
                                        if n < len(universe[j]):
                                            b = universe[j][n]
                                            q = ifedata['units'][b]["chainindex"]  # sequence position
                            else:
                                for (a, b) in listOfPairs[i][j]:
                                    p = ifedata['units'][a]["chainindex"]  # sequence position
                                    q = ifedata['units'][b]["chainindex"]  # sequence position

                                    if modelChainSymmetry[a] == modelChainSymmetry[b] and q - p in constraint[3]:
                                        newList.append((a, b))
                            listOfPairs[i][j] = newList

                        elif constraint[0] == "outside":
                            newList = []
                            if listOfPairs[i][j] == "full":
                                universe[i] = sorted(universe[i])
                                universe[j] = sorted(universe[j])
                                starting_n = 0
                                for m in range(0, len(universe[i])):
                                    foundOne = False
                                    a = universe[i][m]
                                    p = ifedata['units'][a]["chainindex"]  # sequence position
                                    n = starting_n
                                    b = universe[j][n]
                                    q = ifedata['units'][b]["chainindex"]  # sequence position
                                    # probe for the first match
                                    while n < len(universe[j])-1 and modelChainSymmetry[a] != modelChainSymmetry[b]:
                                        n += 1
                                        b = universe[j][n]
                                        q = ifedata['units'][b]["chainindex"]  # sequence position
                                    # accumulate matches
                                    while n < len(universe[j]) and modelChainSymmetry[a] == modelChainSymmetry[b]:
                                        if a != b and (q-p < constraint[1] or q-p > constraint[2]):
                                            newList.append((a,b))
                                        if not foundOne:
                                            starting_n = n
                                            foundOne = True
                                        n += 1
                                        if n < len(universe[j]):
                                            b = universe[j][n]
                                            q = ifedata['units'][b]["chainindex"]  # sequence position
                            else:
                                for (a, b) in listOfPairs[i][j]:
                                    p = ifedata['units'][a]["chainindex"]  # sequence position
                                    q = ifedata['units'][b]["chainindex"]  # sequence position

                                    if modelChainSymmetry[a] == modelChainSymmetry[b] and (
                                    q-p < constraint[1] or q-p > constraint[2]):
                                        newList.append((a,b))
                            listOfPairs[i][j] = newList

                        # print(i,j,constraint,newList) # adam peek


                        universe = reduceUniverses(i, j, universe, listOfPairs)

                        

        printListLengths(Q, numpositions, universe, listOfPairs, "After continuity constraints.")

        timerData = myTimer("Pruning after continuity")

        listOfPairs, universe, emptyList = pruneListOfPairs(listOfPairs,universe)
        printListLengths(Q, numpositions, universe, listOfPairs,
            "After pruning after continuity constraints.")

        listOfPairs, universe, emptyList = pruneListOfPairs(listOfPairs,universe)
        printListLengths(Q, numpositions, universe, listOfPairs,
            "After pruning twice after continuity constraints.")

    # reduce list of pairs according to prohibited pairwise constraints
    if not emptyList and "prohibitedInteractions" in Q and numpositions > 1:
        timerData = myTimer("Prohibited constraints")
        for i in range(0, numpositions):
            for j in range(i + 1, numpositions):
                # above the diagonal
                if len(Q["prohibitedInteractions"][i][j]) > 0:
                    if listOfPairs[i][j] == "full":
                        listOfPairs[i][j] = makeFullList(universe[i], universe[j])

                    for interaction in Q["prohibitedInteractions"][i][j]:
                        if interaction in interactionToPairs and len(interactionToPairs[interaction]) > 0:
                            listOfPairs[i][j] = list(set(listOfPairs[i][j]) - 
                                set(interactionToPairs[interaction][0]))

                    universe = reduceUniverses(i,j,universe,listOfPairs)

                # below the diagonal, for asymmetric pairs like BPh, BR, sO
                if (j in Q["prohibitedInteractions"] and (
                    i in Q["prohibitedInteractions"][j]) and (
                    len(Q["prohibitedInteractions"][j][i]))) > 0:
                    if listOfPairs[i][j] == "full":
                        listOfPairs[i][j] = makeFullList(universe[i],universe[j])

                    for interaction in Q["prohibitedInteractions"][j][i]:
                        if interaction in interactionToPairs and len(interactionToPairs[interaction]) > 0:
                            prohibitPairs = [(b,a) for (a,b) in interactionToPairs[interaction][0]]
                            listOfPairs[i][j] = list(set(listOfPairs[i][j]) - set(prohibitPairs))

                    universe = reduceUniverses(i,j,universe,listOfPairs)


        printListLengths(Q, numpositions, universe, listOfPairs, "After prohibited constraints.")

    if not emptyList and "combinationConstraint" in Q and numpositions > 1:
        timerData = myTimer("Combination constraints")
        for i in range(0, numpositions):
            for j in range(i + 1, numpositions):
                if len(Q["combinationConstraint"][i][j]) > 0:
                    #print(i , j)
                    temp_pair_list = []

                    if listOfPairs[i][j] == "full":
                        listOfPairs[i][j] = makeFullList(universe[i], universe[j])

                    for pair in listOfPairs[i][j]:
                        if((ifedata['units'][pair[0]]['unitType'],
                            ifedata['units'][pair[1]]['unitType']) in Q["combinationConstraint"][i][j]):
                            temp_pair_list.append(pair)

                    listOfPairs[i][j] = temp_pair_list

                    universe = reduceUniverses(i, j, universe, listOfPairs)

        printListLengths(Q, numpositions, universe, listOfPairs, "After combination constraints.")

        # prune list of pairs according to restricted universe and to be able to satisfy all pairwise constraints
        timerData = myTimer("Pruning")

        listOfPairs, universe, emptyList = pruneListOfPairs(listOfPairs, universe)

        printListLengths(Q, numpositions, universe, listOfPairs, "After pruning.")

    # No candidates
    if emptyList:
        return Q, []

    if numpositions == 1:
        # just one position, just one universe, those are the possibilities
        possibilities = [[p] for p in universe[0]]
        inverseperm = [0]

    else:
        # intersect pair lists to find complete candidates

        # reorder positions for more efficient intersections
        listOfPairs, universe, perm = reorderPositions(listOfPairs, universe)
        inverseperm = [perm.index(i) for i in range(len(perm))]

        # compute permuted Distance Matrix
        if (Q["type"] == "geometric" or Q["type"] == "mixed"):
            Q["permutedDistance"] = np.zeros((numpositions, numpositions))
            for i in range(numpositions):
                for j in range(numpositions):
                    Q["permutedDistance"][i][j] = Q["distance"][perm[i]][perm[j]]

            printListLengths(Q, numpositions, universe, listOfPairs, "After reordering positions")

        timerData = myTimer("Intersecting pair lists")

        # turn listOfPairs into lists of possible second indices
        secondElementList = buildSecondElementList(Q, listOfPairs, universe, ifedata)

        # intersect lists of pairs
        Q, possibilities = getPossibilities(Q, ifedata, perm, secondElementList, numpositions, listOfPairs)

    # screen to make sure that no possibility has units with different alternate id (typically A or B)
    timerData = myTimer("Same alternate id")
    possibilities = sameAlternateId(Q, ifedata, possibilities)

    if not SERVER or len(possibilities) > 0:
        print("Found %5d possibilities from %s in %10.4f seconds" % (len(possibilities),ifename,(time() - IFEStartTime)))

    timerData = myTimer("Checking candidates")

    candidates = []

    # for geometric and mixed searches, compute discrepancy of possibilities to query motif
    # for purely symbolic searches, every possibility is a candidate
    if((Q["type"] == "geometric" or Q["type"] == "mixed")):
        querycenters = [Q["centers"][i] for i in perm]
        queryrotations = [Q["rotations"][i] for i in perm]
        timerData = myTimer("Discrepancy from query")

        possnum = 0
        for possibility in possibilities:
            # testing ...
            #print([possibility[inverseperm[i]] for i in range(numpositions)])

            possnum += 1
            #if(possnum % 10000 == 0):
            #    print("Checking discrepancy for possibility #%d" % possnum)
            possibilitycenters = []
            for i in range(0, numpositions):
                possibilitycenters.append(units[possibility[i]]["centers"])
            possibilityrotations = []
            for i in range(0, numpositions):
                possibilityrotations.append(units[possibility[i]]["rotations"])

            d = matrix_discrepancy_cutoff(querycenters, queryrotations, possibilitycenters,
                possibilityrotations, Q["discrepancy"])

            if d is not None and d < Q["discrepancy"]:
                newcandidate = {}
                newcandidate['indices'] = [possibility[inverseperm[i]] for i in range(numpositions)]
                newcandidate['unitids'] = [index_to_id[index] for index in newcandidate['indices']]
                newcandidate['chainindices'] = [units[index]["chainindex"] for index in newcandidate['indices']]
                newcandidate['centers'] = [units[index]["centers"] for index in newcandidate['indices']]
                newcandidate['rotations'] = [units[index]["rotations"] for index in newcandidate['indices']]
                newcandidate['discrepancy'] = d
                newcandidate['interactions'] = lookUpInteractions(newcandidate['indices'],
                    pairToInteractions, pairToCrossingNumber, units)
                candidates.append(newcandidate)

    else:
        for possibility in possibilities:
            newcandidate = {}
            newcandidate['indices'] = [possibility[inverseperm[i]] for i in range(numpositions)]
            newcandidate['unitids'] = [index_to_id[index] for index in newcandidate['indices']]
            newcandidate['chainindices'] = [units[index]["chainindex"] for index in newcandidate['indices']]
            newcandidate['centers'] = [units[index]["centers"] for index in newcandidate['indices']]
            newcandidate['rotations'] = [units[index]["rotations"] for index in newcandidate['indices']]
            newcandidate['interactions'] = lookUpInteractions(newcandidate['indices'],
                pairToInteractions, pairToCrossingNumber, units)
            candidates.append(newcandidate)
    
    return Q, candidates

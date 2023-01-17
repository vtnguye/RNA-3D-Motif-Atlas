#from jmespath import search
from cmath import inf
from fr3d_interactions import get_fr3d_pair_to_interaction_list
from test_motif_atlas_code import add_bulged_nucleotides # this line clutters the output

from collections import defaultdict
from sys import path
from time import time
import networkx as nx
import os.path
import sys
import pickle
import numpy as np
import scipy.io as sio
import pandas as pd
import json
import matplotlib.pyplot as plt


# import the version of urlretrieve appropriate to the Python version
if sys.version_info[0] < 3:
    from urllib import urlretrieve as urlretrieve
else:
    from urllib.request import urlretrieve as urlretrieve
    #import seaborn as sns



# this section gives us access to import things from the pythoncode folder
#wd = path[0] # current working directory "~/Dropbox/MotifAtlas"
#place = wd.rfind("\\")
#otherFolder = wd[:place] + "\\2018 FR3D Intersecting Pairs\\pythoncode"
#path.insert(1, otherFolder)

# add the search folder to the Python path
sys.path.append('search')

# this imports everything from "search.py", most notably the `function FR3D_search()`
# from search import *
from search import FR3D_search,matrix_discrepancy_cutoff
from fr3d_configuration import DATAPATHUNITS,DATAPATHLOOPS
from file_reading import readNAPairsFile
from file_reading import readProteinPositionsFile
from myTimer import myTimer
from orderBySimilarity import treePenalizedPathLength
# using these to get other args to feed FR3D_search()
# from pair_processing import get_pairlist # this got moved into search.py
from query_processing import calculateQueryConstraints
from query_processing import retrieveQueryInformation
from loops_and_strands_utilities import *
from proof_of_concept_Rfam_interactions import which_rfam_and_equiv, chain_to_rfam
# from query_processing import emptyInteractionMatrix
# a program on the server can generate the next line
# That lists all the loops in a motif atlas and the unit ids in each strand.  I think.
# Keys are loop ids, then positions and maybe an indication of what strand they are in
# (1, '6DVK|1|H|G|28') means that G28 is on the border of a single-stranded region
# Then A29,
# http://rna.bgsu.edu/rna3dhub/loops/view/IL_6DVK_003

from HL_3_57_loops_and_strands import loops_and_strands
from IL_3_57_loops_and_strands import loops_and_strands

'''
All stacks and basepairs:
'''
bptypes = {'cWW', 'tWW', 'cWH', 'tWH', 'cWS', 'tWS', 'cHH', 'tHH', 'cHS', 'tHS',\
                   'cSS', 'tSS','cHW','tHW','cSW','tSW','cSH','tSH'}
near_bptypes = {'ntHS', 'ntSH', 'ntWH', 'ncHS', 'ncSH', 'ncWW', 'ncHH', 'ntWW', 'ntSS',\
                    'ncSS', 'ncWS', 'ntWS', 'ntHH', 'ntHW', 'ncSW', 'ncHW', 'ncWH', 'ntSW'}
stacks = {'s33','s35','s55','s53'}
near_stacks = {'ns35','ns55','ns33','ns53'}

all_stacks = stacks | near_stacks
all_bptypes = bptypes | near_bptypes
CONFLICTING_BASEPAIRS_AND_STACKS = 1
NO_CANDIDATES = 0
SEARCH_SPACE_CONFLICT = -1
FLANKING_BP_CONFLICT = 9
HAIRPIN_STACK_PENALTY1 = 3
HAIRPIN_STACK_PENALTY2 = 3
BP_PENALTY             = 4
NEAR_BP_PENALTY        = 5
STACK_PENALTY          = 6
MISMATCHED_BULGE       = 10
SERVER = False    # to suppress printing list lengths
'''
Download current IL json file
Save as a pickle file.
'''

def strandify(loops_and_strands,all_structures=None):
    # loops=[]
    # for motif_group in loops_and_strands:
    #     loops_alignments = motif_group["alignment"]
    #     chainbreak = int(motif_group["chainbreak"])
    #     for loop_id,alignment in loops_alignments.items():
    #         pdb_id = loop_id.split("_")[1]
    #         if pdb_id in all_structures:
    #             new_loop = {}
    #             new_loop["loop_id"]= loop_id
    #             new_loop["strand"] = []
    #             new_loop["strand"].append(alignment[0:chainbreak])
    #             new_loop["strand"].append(alignment[chainbreak:len(alignment)])
    #             loops.append(new_loop)
    '''
    The code above are for json file...
    '''
    loops = []
    for loop_id in loops_and_strands:
        fields = loop_id.split("_")
        if fields[1] in all_structures:
            new_loop = {}
            new_loop["loop_id"] = loop_id
            new_loop["strand"] = [] # will be a list of 1 strand for HL, 2 for IL, 3 for J3, etc.

            current_strand = []
            bordercount = 0
            positions = sorted(loops_and_strands[loop_id].keys())
            # Loop over positions in the loop and identify where strands stop and start
            # "border" variable is 1 if the nucleotide starts or ends a single-stranded region, o/w 0
            for position in positions:
                border = loops_and_strands[loop_id][position][0]
                unit_id = loops_and_strands[loop_id][position][1]
                # print("Loop %s position %s has border %s and unit id %s" % (loop_id, position, border, unit_id))
                current_strand.append(unit_id)
                bordercount += border
                # when you get to the second bordering nucleotide on the strand, store the strand, start a new one
                if bordercount == 2:
                    bordercount = 0
                    new_loop["strand"].append(current_strand)
                    current_strand = []
                # append to the current strand until
            loops.append(new_loop)
    return loops

def startup_list_of_dictionaries(loops_and_strands,loops_of_interest=None):
    """
    Loop over the loop ids and extract the PDB IDs and store them in a set
    maybe also pass in "molecule type" for use in Q['requiredMoleculeType']
    """
    all_structures =[]
 
    if loops_of_interest:
        for loop_id in loops_of_interest:
            fields = loop_id.split("_")           
            all_structures.append(fields[1])      
    else:
        for loop_id in loops_and_strands.keys():
            fields = loop_id.split("_")
            all_structures.append(fields[1])     
   
    all_structures = set(all_structures)

    pair_to_interaction_list = defaultdict(list)
    for pdb_id in all_structures:
        pair_to_interaction_list.update(get_fr3d_pair_to_interaction_list(pdb_id))

    # Loop over each loop id to identify and store the unit ids in each strand
    # loops will be a list of dictionaries, one for each loop
    # This adds a key called "strand"

    loops = strandify(loops_and_strands, all_structures)
    loops = add_bulged_nucleotides(loops, pair_to_interaction_list)
    
    return(loops, pair_to_interaction_list)

def make_query_structure(loop):
    '''
    this function should take in a loop-like-object of n strands,
    and return a query dictionary `Q` (as seen in query_definitions.py)
    Here, loop is usually the strands of the **unbulged** units of the loop
    '''

    positions = get_nt_positions(loop)

    Q = defaultdict(dict) # prevents later errors on assignments



    # philosophy = if the key,value takes more than one line to assign, define a `make` function for it
    Q['type'] = "mixed"     # geometric AND symbolic
    Q['name'] = "AVA"       # this is to DODGE Q possibly getting re-defined # could be named by loop ID
    Q['motif_atlas'] = True # make it possible to tailor output for motif atlas AVA
    Q['discrepancy'] = 1.0  # This is what the Motif Atlas uses
    Q = make_num_positions(Q, positions)

    # seeding section
    Q["interactionMatrix"] = emptyInteractionMatrix(Q['numpositions'])
    Q['unitID'] = []        # these are the unit ids that the query will actually use for the search
    Q['requiredUnitType'] = [None] * Q['numpositions'] # these list operations are copied from query_processing.py
    Q['requiredMoleculeType'] = ["RNA"] * Q['numpositions'] # future, find a way to discern in DNA/RNA, or be a user input variable

    Q['errorMessage'] = []
    Q['MAXTIME'] = 60*60    # maximum search time 1 hour, may not be imposed
    Q["CPUTimeUsed"] = 0    # needs to be tracked, but actually ignored here

    # below are things that prefer to be fed strand by strand
    for index, strand in enumerate(loop['strand']):
        Q = make_unit_ids(Q, strand)
        if len(positions[index]) > 1: 
        # if strand is more than 1 nt, find direction + constraint, else `pass`
            Q = make_directional_constraints(Q, positions[index], "increasing")

    # below are things requiring above functions to be done, but dislike strand by strand
    Q = make_search_files(Q)
    Q = make_req_interactions(Q, positions)

    return(Q)


def make_flanking_bp_query_structure(loop):
    '''
    A small query for IL and J3 and J4 flanking basepairs; not for hairpins.
    this function should take in a loop-like-object of n strands,
    and return a query dictionary `Q` (as seen in query_definitions.py)'''

    positions = get_nt_positions(loop)
    flanking_positions = [p for p in positions]

    for i in range(len(positions)):
        flanking_positions[i] = [ positions[i][0],positions[i][-1]]

    # we probably need a new function here to return only the flanking nt positions
    flanking_strands = [s for s in loop["strand"]]
    for i in range(len(loop['strand'])):
        flanking_strands[i] = [ loop['strand'][i][0],loop['strand'][i][-1]]


    Q = defaultdict(dict) # prevents later errors on assignments

    # philosophy = if the key,value takes more than one line to assign, define a `make` function for it
    Q['type'] = "mixed"     # geometric AND symbolic
    Q['name'] = "AVA"       # this is to DODGE Q possibly getting re-defined # could be named by loop ID
    Q['motif_atlas'] = True # make it possible to tailor output for motif atlas AVA
    Q['discrepancy'] = 1.0  # be strict but not super strict about the flanking pairs
    Q = make_num_positions(Q, flanking_positions)

    Q["interactionMatrix"] = emptyInteractionMatrix(Q['numpositions'])

    #Only for IL
    if Q['numpositions'] ==4:
        Q["interactionMatrix"][0][3] = "cWW and GC CG AU UA GU UG"
        Q["interactionMatrix"][1][2] = "cWW and GC CG AU UA GU UG" 
        Q["interactionMatrix"][1][0] = ">" 
        Q["interactionMatrix"][3][2] = ">"

    Q['unitID'] = []
    Q['requiredUnitType'] = [None] * Q['numpositions'] # these list operations are copied from query_processing.py
    Q['requiredMoleculeType'] = ["RNA"] * Q['numpositions'] # future, find a way to discern in DNA/RNA, or be a user input variable

    Q['errorMessage'] = []

    # below are things that prefer to be fed strand by strand
    for index, strand in enumerate(flanking_strands):
        Q = make_unit_ids(Q, strand)
        # if len(flanking_positions[index]) > 1:
        # # if strand is more than 1 nt, find direction + constraint, else `pass`
        #     Q = make_directional_constraints(Q, flanking_positions[index], "increasing")

    # below are things requiring above functions to be done, but dislike strand by strand
    Q = make_search_files(Q)

    # this will have fewer interactions when just searching for flanking pairs
    #Q = make_req_interactions(Q, positions)

    return(Q)


def get_nt_positions(loop):
    '''takes in a list of strings representing nucleotides
    returns a list of lists of nucleotide position numbers (by strand)
    tiny ex: positions = [[nt1, nt2, nt3], [nt4, nt5, nt6]]
    ^ nts 1-3 are on one strand and nt4-6 are on another
    also, nts here look like: "5J7L|1|AA|C|569" '''

    positions = []
    count = 0
    for index, strand in enumerate(loop['strand']):
        tempList = []
        for element in strand:
            tempList.append(count)
            count += 1
        positions.append(tempList)

    return(positions)


def make_directional_constraints(query, positions, order):
    '''Adam, make a good description of this'''

    if order.lower() == "increasing": # these LOOK backwards, but are not
        symbol = ">"
    elif order.lower() == "decreasing":
        symbol = "<"
    else:
        raise NameError('Tried to make directional without "increasing" or "decreasing" order argument')

    previousPositions = [] # needed to FILL lower diagonal
    firstIter = True
    for position in positions: # remember that this function is called on each strand
        if firstIter == False: # if first iteration, previousPositions is empty
            for prev in previousPositions:
                query['interactionMatrix'][position][prev] = symbol
            previousPositions.append(position)
        else:
            firstIter = False
            previousPositions.append(position)

    return(query)


def make_num_positions(query, positions):
    '''counts the nts in a loop'''
    # adam, consider removing this
    counter = 0
    for strand in positions:
        counter += len(strand)
    query['numpositions'] = counter

    return(query)

def emptyInteractionMatrix(n):
    '''set up an empty list of required interactions,
    then the user only needs to set the actual constraints needed'''

    emptyList = defaultdict(dict)
    for i in range(0, n):
        for j in range(0, n):
            emptyList[i][j] = ""

    return emptyList

def make_req_interactions(query, positionsList):
    '''fills out a matrix of required cWWs:
    last nt of first strand needs to cWW to first nt of second strand (and so on)
    should be putting "cWW" ABOVE diagonal'''

    heads = [] #remember that lists are ordered, so these are parallel
    tails = []
    for strand in positionsList: # filling points of interest
        heads.append(strand[0])
        tails.append(strand[-1])

    prevTail = -1
    firstHead = -1
    for head, tail in zip(heads, tails): # adding constraint
        # first tail -> second head, second tail -> third head,
        # ... last tail -> first head
        if firstHead == -1: # if first pass
            firstHead = head
        else: # if not first pass, we have connections to make
            query['interactionMatrix'][prevTail][head] = "cWW GC CG AU UA GU UG"
        if tail == tails[-1]: # if last pass
            query['interactionMatrix'][firstHead][tail] = "cWW GC CG AU UA GU UG"
        prevTail = tail

    return(query)


def make_search_files(query):
    '''this function assumes `make_unit_ids()` was already ran
    Q['searchFiles'] = [FIRST part of unit_id]'''

    chains = set()
    for nt in query['unitID']:
        fields = nt.split("|")
        link = "|".join(fields[0:3])
        chains.add(link)
    query['searchFiles'] = ["+".join(chains)] # future, don't overwrite

    return(query)


def make_unit_ids(query, strand):
    '''example of what this should make
    Q["unitID"] = ["4V9F|1|0|U|1026","4V9F|1|0|A|1032","4V9F|1|0|G|1034", ... ]'''

    for nt in strand:
        query['unitID'].append(nt)

    return(query)

def combine_dicts(x, y):
    z = x.copy()
    z.update(y)
    return(z)

def readPAI(query, ifename, alternate = ""):
    '''WAS readPositionsAndInteractions() from ifedata.py.
    It got copied + changed here for AvA needs'''

    RNA_positions_file_name = ifename.replace("|","-").replace("+","_")
    starting_index = 0
    ifedata = {}
    ifedata['index_to_id'] = {}
    ifedata['id_to_index'] = {}
    ifedata['centers'] = np.empty((0, 3))
    ifedata['rotations'] = np.empty((0, 3, 3))
    ifedata['ids'] = []
    ifedata['units'] = []
    allCenters = []
    allModels = []

    # lists should start empty, append with RNA if necessary, with protein if necessary, with DNA if necessary, etc.
    if(any("RNA" in query["requiredMoleculeType"][index] for index in range(len(query["requiredMoleculeType"])))):
        # check to see if RNA is a required unit type, and if so, read RNA data

        for chainString in RNA_positions_file_name.split("_"):

            query, centers, rotations, ids, id_to_index, index_to_id, chainIndices = read_RNA_pos_file(query, chainString, starting_index)

            ifedata['index_to_id'] = combine_dicts(ifedata['index_to_id'], index_to_id)
            ifedata['id_to_index'] = combine_dicts(ifedata['id_to_index'], id_to_index)
            ifedata['centers'] = np.append(ifedata['centers'], centers, axis = 0)
            ifedata['rotations'] = np.append(ifedata['rotations'], rotations, axis = 0)
            ifedata['ids'].extend(ids)

            for unitID in ids:
                unit_information = {}
                unit_information["centers"] = centers[id_to_index[unitID]-starting_index]
                unit_information["rotations"] = rotations[id_to_index[unitID]-starting_index]
                data = unitID.split("|")
                allModels.append(data[1]) # extract model number
                unit_information["unitType"] = data[3] # supposed to get the nucleotide letter
                unit_information["moleculeType"] = "RNA"
                unit_information["chainindex"] = chainIndices[id_to_index[unitID]-starting_index]
                ifedata["units"].append(unit_information)

            starting_index += len(centers)

    PDBID = ifename.split("|")[0]
    query, interactionToPairs, pairToInteractions, pairToCrossingNumber = readNAPairsFile(query, PDBID, ifedata["id_to_index"], alternate)
    ifedata['interactionToPairs'] = interactionToPairs
    ifedata['pairToInteractions'] = pairToInteractions
    ifedata['pairToCrossingNumber'] = pairToCrossingNumber
    ifedata['models'] = allModels

    return query, ifedata

# Read .pickle file of RNA base center and rotation matrix; download if necessary
def read_RNA_pos_file(Q, chainString, starting_index):

    ids = []
    chainIndices = []
    centers = []
    rotations = []
    line_num = starting_index
    id_to_index = defaultdict()
    index_to_id = defaultdict()
    our_indices = []

    filename = chainString + "_RNA.pickle"
    pathAndFileName = os.path.join(DATAPATHUNITS, filename)

    if not os.path.exists(DATAPATHUNITS):
        os.mkdir(DATAPATHUNITS)

    if not os.path.exists(pathAndFileName) and not SERVER:
        print("Downloading "+filename)
        urlretrieve("http://rna.bgsu.edu/units/" + filename, pathAndFileName)


    if os.path.exists(pathAndFileName):

        print("Reading "+pathAndFileName)

        if sys.version_info[0] < 3:
            try:
                ids, chainIndices, centers, rotations = pickle.load(open(pathAndFileName,"rb"))
            except:
                print("Could not read "+filename+" A==================================")
                Q["userMessage"].append("Could not retrieve RNA unit file "+filename)
                centers = np.zeros((0, 3))  #of lines/nucleotides in file
                rotations = np.zeros((0, 3, 3))
        else:
            try:
                ids, chainIndices, centers, rotations = pickle.load(open(pathAndFileName,"rb"), encoding = 'latin1')
            except:
                print("Could not read "+filename+" A*********************************")
                Q["userMessage"].append("Could not retrieve RNA unit file "+filename)
                centers = np.zeros((0, 3))  # of lines/nucleotides in file
                rotations = np.zeros((0, 3, 3))


        for i in range(0, len(ids)):
            # if ids[i] in Q['unitID']: # big Q reference, leaves questions about removing bulges from Q. Try to update to Q['fullUnitID']
            if ids[i] in Q['fullUnits']:
                our_indices.append(i) # above line could also add "or in Q['bulges']"
                id_to_index[ids[i]] = line_num
                index_to_id[line_num] = ids[i]
                line_num += 1

    else:
        print("Could not find "+filename)
        Q["userMessage"].append("Could not retrieve RNA unit file "+filename)
        centers = np.zeros((0, 3))  # of lines/nucleotides in file
        rotations = np.zeros((0, 3, 3))

    ids = [ids[i] for i in our_indices] # equivalent to: ids = ids[our_indices]
    centers = np.asarray([centers[i] for i in our_indices])
    rotations = np.asarray([rotations[i] for i in our_indices])
    chainIndices = [chainIndices[i] for i in our_indices]

    return Q, centers, rotations, ids, id_to_index, index_to_id, chainIndices


def create_query(loop):
    unbulged = {}
    # double list comprehension to keep the structure of ['strand'] being a list of lists, while removing bulges
    # when a 5-nt IL with one bulged nt, keep all 5 nucleotides in unbulged variable
    query_length = sum([len(strand) for strand in loop["strand"]])
    query_bulged_length = len(loop["bulged"])

    unbulged['strand'] = [[unitID for unitID in strand if unitID not in loop['bulged']] for strand in loop['strand']]
    
    # makes the query (Q) without bulges, while bulges should be left IN targets (ifedata)
    if query_length ==5 and query_bulged_length ==1:
        query = make_query_structure(loop)
        query['discrepancy']= 99
    else:
        query = make_query_structure(unbulged)
    query['fullUnits'] = [unitID for strand in loop['strand'] for unitID in strand]
    query = retrieveQueryInformation(query)
    if 'errorStatus' in query and query['errorStatus'] == "write and exit":
        return None
    else:
        query = calculateQueryConstraints(query)
        query['numStrands'] = len(loop['strand'])
    return(query)

def create_search_space(loop):    
    query = defaultdict(list)
    # makes the query (Q) without bulges, while bulges should be left IN targets (ifedata)
    for strand in loop['strand']:
        for nt in strand:
            query['unitID'].append(nt)
    positions = get_nt_positions(loop)
    query = make_num_positions(query, positions)

    query['fullUnits'] = [unitID for strand in loop['strand'] for unitID in strand]
    query['requiredMoleculeType'] = ["RNA"] * query['numpositions']
    query['activeInteractions'] = ['cWW']
    query['userMessage']=[]
    query = make_search_files(query)

    query, ifedata = readPAI(query, query['searchFiles'][0])
    return(ifedata)

def create_flanking_bp_query(loop):
    unbulged = {}
    unbulged['strand'] = [[unitID for unitID in strand if unitID not in loop['bulged']] for strand in loop['strand']]
    flanking_bp_query = make_flanking_bp_query_structure(unbulged)
    flanking_bp_query = retrieveQueryInformation(flanking_bp_query)
    if 'errorStatus' in flanking_bp_query and flanking_bp_query['errorStatus'] == "write and exit":
        return None
    else:
        flanking_bp_query = calculateQueryConstraints(flanking_bp_query)
        flanking_bp_query['numStrands'] = len(loop['strand'])    
    return(flanking_bp_query)

def get_bulge(query,search_space):
    query_bulge = ''
    search_space_bulge = ''
    if len(query['loop_info']['bulged'])>0:
        query_bulge = query['loop_info']['bulged'][0].split("|")[3]
    if len(search_space['loop_info']['bulged'])>0:
        search_space_bulge = search_space['loop_info']['bulged'][0].split("|")[3]
    return(query_bulge,search_space_bulge)

def analyze_extra_nucleotides(candidates, search_space,query):
    '''
    Analyze extra nts of candidates that has no dq
    Check extra nts with core first,
    Then with other extra nts
    '''
    candidates_data = search_space['ifedata']
    interaction_with_extra = ''
    interaction_with_core = ''
    for candidate in candidates:
        if len(candidate["dq"]) > 0:
            continue
        
        core_nts = candidate["indices"]
        extra_nts = list(set(candidates_data["index_to_id"].keys()) - set(core_nts))
        if len(extra_nts) == 0:
            continue

        stack_counter = 0
        for i in range(len(extra_nts)):
            #Interaction with other extra nts
            for j in range(i+1,len(extra_nts)):
                interaction_with_extra = get_set_of_interactions(extra_nts[i],extra_nts[j],candidates_data)
                if interaction_with_extra.intersection(bptypes):
                    candidate['dq'].append(BP_PENALTY)
                if interaction_with_extra.intersection(near_bptypes) :
                    candidate['dq'].append(NEAR_BP_PENALTY)

            #Interaction with core nts
            for core_index in core_nts: 
                interaction_with_core = get_set_of_interactions(extra_nts[i],core_index,candidates_data)
                if interaction_with_core.intersection(bptypes) :
                    candidate['dq'].append(BP_PENALTY)
                if interaction_with_core.intersection(near_bptypes):
                    candidate['dq'].append(NEAR_BP_PENALTY)
                if interaction_with_core.intersection(stacks):
                    stack_counter +=1
        if stack_counter > 1:
            candidate['dq'].append(STACK_PENALTY)

        candidate['dq'] = list(set(candidate['dq']))
    
    return(candidates)

def filter_out_conflicting_basepairs_and_stacks(candidates, query, search_space):
    '''
    this looks for conflicting base pairs or stacking for each element
    in a list of matched candidates and returns the viable portion of the dictionary
    '''
    # candidates data is just the ifedata of the candidate

    query_ifedata = query['ifedata']

    # this is the call for easy reading
    candidates_data = search_space['ifedata']

    not_rejected = []

    if(len(candidates) > 1):
        # sort candidates by discrepancy
        candidates = sorted(candidates, key = lambda i: i['discrepancy'])

    for candidate in candidates:

        DQ_code = [] # change logic to use DQ_code as a set? saves lots of operations, Adam

        filtered_candidates = {}

        # REMEMBER TO CHECK IF KEYS IN DEFAULT DICT BEFORE CALLING #################################

        for i in range(len(candidate['indices'])): # goes like 0 to 7
        # next line utilizes Q not including bulges, and how its 'unitID's are in index order, to
        # return to its 'ifedata's indexing system
            index_i_of_query = query_ifedata['id_to_index'][ query["Q"]["unitID"][i] ]
            index_i_of_candidate = candidate['indices'][i]#The integer indentifying nucleotide in i position
            
            for j in range(i + 1, len(candidate['indices'])): # goes like i+1 to 7
                index_j_of_query = query_ifedata['id_to_index'][ query["Q"]["unitID"][j] ]
                index_j_of_candidate = candidate['indices'][j]
        
                query_interaction= get_set_of_interactions(index_i_of_query, index_j_of_query, query_ifedata) & (bptypes | stacks)
                candidate_interaction = get_set_of_interactions(index_i_of_candidate, index_j_of_candidate, candidates_data) & (bptypes | stacks)
                DQ_code += (are_conflicting(query_interaction, candidate_interaction))
        
        set_DQ_code = set(DQ_code)
        # proposed solution, in bottom of "for candidate"
        if len(set_DQ_code) == 0:
            filtered_candidates['dq'] = []
            filtered_candidates['query_id'] = query['loop_info']['loop_id']
            filtered_candidates['search_space_id'] = search_space['loop_info']['loop_id']
            filtered_candidates['indices'] = candidate['indices']
            filtered_candidates['query_unit_ids'] = query['Q']['unitID']
            filtered_candidates['target_unit_ids'] = candidate['unitids']
            filtered_candidates['discrepancy'] = candidate['discrepancy']
            filtered_candidates['numStrands'] = query['Q']['numStrands']

            filtered_candidates['query_rfam'] = query['loop_info']['rfam']
            filtered_candidates['search_space_rfam'] = search_space['loop_info']['rfam']

            not_rejected.append(filtered_candidates)

        else: # if has a real DQ_code
            filtered_candidates['dq'] = list(set_DQ_code)
            filtered_candidates['discrepancy'] = candidate['discrepancy']
            filtered_candidates['numStrands'] = query['Q']['numStrands']

            not_rejected.append(filtered_candidates)

    # DQ_code is short for disqualification code
    return(not_rejected)


def find_lowest_discrepancy(candidates):
    '''
    this takes a list of possible candidates and returns the one
    with the lowest discrepancy, unless all candidates have
    disqualification codes
    '''
    lowest_dq_discrepancy = 100 # purposely seeded high
    lowest_matched_discrepancy = 100
    positive_found = 0
    best_match = 0

    for candidate in candidates:
            if candidate['dq'] == []:
                if candidate['discrepancy'] < lowest_matched_discrepancy:
                    lowest_matched_discrepancy = candidate['discrepancy']
                    best_match = candidate
            else: # if a disqualified match
                if candidate['discrepancy'] < lowest_dq_discrepancy:
                    lowest_dq_discrepancy = candidate['discrepancy']
                    best_dq = candidate
    
    # if an error happens on these next few lines, something is likely hitting
    # a discrepancy ABOVE the lowest seeded value set above
    if best_match:
        return([best_match])
    else: # if all disqualified
        return([best_dq])

def get_set_of_interactions(i, j, ifedata):
    '''
    the definition of interaction here is hard stacks or hard basepairs,
    but not near stacks nor near basepairs
    '''
    if((i,j) in ifedata['pairToInteractions'] ):
        return set(ifedata['pairToInteractions'][(i,j)])
    else:
        return set([])

def are_conflicting(q_int, c_int):
    '''
    q_int is short for "interaction of query between nts (i, j)"
    c_int is short for "interaction of canditate between nts (i, j)"
    the definition of interaction here is hard stacks or hard basepairs,
    but not near stacks nor near basepairs
    '''
    # return options [-1 = "compatible", 8 = "basestack_mismatch",
    # 					7 = "basepair_mismatch"]
    q_int = set(q_int)
    c_int = set(c_int)
    BASESTACK_MISMATCH = 8
    BASEPAIR_MISMATCH = 7

    if(q_int == c_int):
        return []
    if( (q_int & stacks) and (c_int & stacks) ): # loose due to modeling flips
        return []
    if( not ( q_int&stacks) and (c_int&stacks) ):
        if(len(q_int) != 0):
            return [BASESTACK_MISMATCH] # base stack mismatch
        else:
            return []
    if( (q_int & stacks) and not (c_int & stacks)):
        if(len(c_int) != 0):
            return [BASESTACK_MISMATCH] # base stack mismatch
        else:
            return []
    # no stacks by this line
    if( len(q_int) == 0 or len(c_int) == 0 ): # one is empty
        return [] # compatible
    if( q_int != c_int ): # i dont believe any loops have made it this far yet (82x82 tested)
        return [BASEPAIR_MISMATCH] # base pair mismatch
    print("we hit the default case... inspect c_int and q_int")
    raise BaseException("error in `are_conflicting()` logic.\nq_int = " + q_int + "\nc_int = " + c_int)

def name_structure(dict_of_searches):
    '''
    this is a helper function for the naming of files
    '''
    number_of_strands = dict_of_searches[list(dict_of_searches.keys())[0]][0]['numStrands']

    if number_of_strands == 1:
        return("HL") # hairpin loop
    if number_of_strands == 2:
        return("IL") # internal loop
    if number_of_strands > 2:
        return("J" + str(number_of_strands))
    else:
        raise BaseException("number of strands is {}".format(number_of_strands))


def load_previous_search_results_one_pdb(loop_type,query_pdb_id, path = "./search_results/"):
    
    results = {}
    if not os.path.exists(path):
        os.mkdir(path)
        return(results)
    #file_name_and_path = path + "/search_results/" + structure_name + ".pickle"
    file_name = loop_type + "_" + query_pdb_id + "_search_results.pickle"
    file_path = os.path.join(path,file_name)
    
    if os.path.exists(file_path):
        if sys.version_info[0] < 3:
            results = pickle.load(open(file_path, "rb"))
        else:
            results = pickle.load(open(file_path, "rb"), encoding = 'latin1')

        for (query,search_space),value in results.items(): #Load No-match result as [{'dq': 0, 'discrepancy': 99}] instead of an integer
            if isinstance(value,int):
                results[(query,search_space)] = [{'dq': [value], 'discrepancy': 99}]

        return(results)
    return(results)

def save_search_results_one_pdb(loop_type,query_pdb_id, results, path = "./search_results/"):
    if not os.path.exists(path):
        os.mkdir(path)
        
    file_name = loop_type + '_' + query_pdb_id + '_search_results.pickle'
    file_path = os.path.join(path,file_name)

    pickle.dump(obj = results, file = open(file_path, "wb"), protocol = 2)
    return()

def create_queries_and_search_spaces(loops,loops_of_interest = None):
    queries = {}
    search_spaces = {}
    for loop in loops:
        if not loops_of_interest or loop['loop_id'] in loops_of_interest:
            ifedata = create_search_space(loop)
            Q = create_query(loop)
            loop['rfam'] = which_rfam_and_equiv(loop)
            if Q:
                queries[loop['loop_id']] = {}
                queries[loop['loop_id']]['loop_info'] = loop
                queries[loop['loop_id']]['Q'] = Q
                queries[loop['loop_id']]['ifedata'] = ifedata
            if ifedata:
                search_spaces[loop['loop_id']] = {}
                search_spaces[loop['loop_id']]['loop_info'] = loop
                search_spaces[loop['loop_id']]['ifedata'] = ifedata
    return(queries,search_spaces)

def create_flanking_bp_queries(loops,loops_of_interest = None):
    queries = {}
    for loop in loops:
        if not loops_of_interest or loop['loop_id'] in loops_of_interest:
            Q = create_flanking_bp_query(loop)
            if Q:
                queries[loop['loop_id']] = {}
                queries[loop['loop_id']]['loop_info'] = loop
                queries[loop['loop_id']]['Q'] = Q
                if loop['loop_id'] == "IL_5TBW_144" or loop['loop_id'] == "IL_6SY6_001":
                    queries[loop['loop_id']]['Q']['discrepancy']= 0.95
    return(queries)

'''Seyoung code from MotifAtlasPipeline.py'''
def average_disc(clique,disc_matrix):
    sum_disc = 0
    counter = 0
    clique.sort()
    for i in range(len(clique)-1):
        for j in clique[i+1:]:
            sum_disc += disc_matrix[clique[i]][j]
            counter += 1
    return sum_disc/counter

def find_best_clique(disc_m,lst,ratio,descending=True):
    if not descending:
        lst.sort(key=len,reverse=True)
    
    limit = round(len(lst[0])*ratio)
    large_cliques = [lst[0]]
    for i in lst:
        if len(i) >= limit:
            large_cliques.append(i)
        else:
            break
        
    min_disc = average_disc(large_cliques[0],disc_m)
    new_clique = large_cliques[0]
    for i in large_cliques:
        if len(i)>1:
            disc = average_disc(i,disc_m)
            if disc < min_disc:
                #print(f"Switching to clique of size {len(i)} with disc {disc} over {min_disc}")
                min_disc = disc
                new_clique = i            
    print(f"found clique of size{len(new_clique)} with disc{min_disc} going over {len(large_cliques)} \
    #cliques from size {len(large_cliques[0])} to {len(large_cliques[len(large_cliques)-1])}")

    
    return new_clique

def cluster_motifs3(cliques_lst,disc_m,ratio,loop_ids):
    groups = []
    
    while True:
        cliques_lst.sort(key=len,reverse=True)
        if len(cliques_lst[0]) == 1 or len(cliques_lst) == 0 or len(cliques_lst[0]) == 0:
            break
        best_clique = find_best_clique(disc_m,cliques_lst,ratio)
        new_clique_names = [x for x in loop_ids if loop_ids.index(x) in best_clique] 
        groups.append(new_clique_names)

        # remove row/column from both matrices 
        cliques_lst = [[x for x in sub if x not in best_clique] for sub in cliques_lst] 
    set_clique = set()
    for lst in cliques_lst:
        if len(lst)>0:
            set_clique.add(loop_ids[lst[0]])
    for singleton in set_clique:
        groups.append([singleton])
    return groups

def set_loop_type(queries):
    loop_id = list(queries.keys())[0]
    loop_type = loop_id.split('_')[0]
    return loop_type

def all_against_all_searches(loop_type,queries, search_spaces,flanking_bp_queries,load_previous_result = True,reversed_search=False, save_path = "./search_results/"):
    '''
    Search all loops against all loops
    Take in query strands and search space strands
    Flow:
    Set query_pbd_id to empty to check the first iteration
    Sort all pbd ids
    Load previous search results if query_pbd_id changes
    Save the results if query_pbd_id changes and new_results = true
    results is a dictionary store temporary search results
    search_results is the accumulated search results, get returned
    '''

    new_results = False
    timer_data= myTimer("All against all search")
    query_pdb_id = ""
    search_results = {} # search_results store the final, accumulated results
    #DEFINE LOOP TYPE

    loop_counter = 0
    
    for query_id in sorted(queries.keys(),reverse=reversed_search):
        loop_counter += 1
        print("Searching for loop %d, %s, in %d loops" % (loop_counter,query_id,len(search_spaces.keys())))

        #Only load previous result when there is a change in query_pdb_id.
        #Save the results if not the first iteration.
        if query_pdb_id != query_id.split("_")[1]:
            results = {} # results: {(query_id,search_space_id) : [dq,discrepancy]}
            query_pdb_id = query_id.split("_")[1]
            if load_previous_result: #To run tests without loading previous results
                timer_data = myTimer("Loading files")
                results = load_previous_search_results_one_pdb(loop_type,query_pdb_id, save_path)
            new_results = False
        
        for search_space_id in sorted(list(search_spaces.keys())):

            if(query_id != search_space_id):
                # Check if this search already done           
                if (query_id, search_space_id) not in results:
                    query_length = len(queries[query_id]['Q']['fullUnits'])
                    query_bulge_length = len(queries[query_id]['loop_info']['bulged'])
                    is_single_bulged_candidate = query_bulge_length ==1 and query_length==5          

                    search_space_length = len(search_spaces[search_space_id]['ifedata']['index_to_id'])
                    search_space_bulged_length = len(search_spaces[search_space_id]['loop_info']['bulged'])
                    is_single_bulged_search_space = search_space_bulged_length ==1 and search_space_length==5          
                    
                    if loop_type == 'IL' and is_single_bulged_candidate:
                        if is_single_bulged_search_space:
                            Q, candidates, elapsed_time = FR3D_search(Q = queries[query_id]['Q'],
                                        ifedata = search_spaces[search_space_id]['ifedata'], ifename = search_space_id,
                                        timerData = timer_data)
                            if candidates:
                                temp_result = analyze_single_bulged_base_loops(candidates=candidates,query=queries[query_id],search_space=search_spaces[search_space_id])
                                results[(query_id, search_space_id)] = temp_result
                            else:
                                results[(query_id, search_space_id)] =  MISMATCHED_BULGE

                        else:
                            results[(query_id, search_space_id)] =  MISMATCHED_BULGE
                        new_results=True
                    elif loop_type=="IL" and is_single_bulged_search_space:
                        results[(query_id, search_space_id)] =  MISMATCHED_BULGE
                        new_results= True
                    #Regular search
                    else:
                        if(query_length <= search_space_length):
                            #Checking flanking bp
                            if loop_type != 'HL':
                                Q,candidates, elapsed_time = FR3D_search(Q=flanking_bp_queries[query_id]['Q'],ifedata = search_spaces[search_space_id]['ifedata'], ifename = search_space_id,
                            timerData = timer_data)
                            else:
                                candidates = []

                            if loop_type=="HL" or candidates:
                                Q, candidates, elapsed_time = FR3D_search(Q = queries[query_id]['Q'],
                                        ifedata = search_spaces[search_space_id]['ifedata'], ifename = search_space_id,
                                        timerData = timer_data)
                                
                                if candidates: # this is to filter out empties AND get rid of all being list[0]
                                    temp_result = filter_out_conflicting_basepairs_and_stacks(candidates, queries[query_id], search_spaces[search_space_id])
                                    if temp_result:
                                        # For now, only save the lowest discrepancy match.
                                        temp_result = analyze_extra_nucleotides(temp_result,search_spaces[search_space_id],queries[query_id])
                                        temp_result = find_lowest_discrepancy(temp_result)
                                        # new_query_ids.append(query_id)
                                        results[(query_id,search_space_id)] = temp_result
                                        # results[query_pdb_id].append({(query_id,search_space):temp_result})        
                                        # try to differentiate core nts vs extra nts
                                        # run thru extra nt filter

                                        print('Found %5d candidates and %5d remain after filtering' % (len(candidates),len(temp_result)))

                                    else:
                                        results[(query_id, search_space_id)] =  CONFLICTING_BASEPAIRS_AND_STACKS

                                else: # no candidates from fred search
                                    #results[(query_id, search_space)] = [{'dq': 0, 'discrepancy': 99}] # hoping disqualification code of 0 makes sense for NO MATCH
                                    #Put 0 instead of list
                                    results[(query_id, search_space_id)] =  NO_CANDIDATES
                                # after candidate sorting, we will no longer use Find_lowest()
                                # run test_motif_atlas_code::check_interaction() after EACH FR3D search Adam
                            else:
                                results[(query_id, search_space_id)] =  FLANKING_BP_CONFLICT
                        else:
                            results[(query_id, search_space_id)] =  SEARCH_SPACE_CONFLICT
                        new_results = True
                if isinstance(results[(query_id, search_space_id)],int):
                    search_results[(query_id,search_space_id)] = [{'dq': [NO_CANDIDATES], 'discrepancy': 99}]
                else:
                    search_results[(query_id,search_space_id)] = results[(query_id,search_space_id)]
        if new_results:
            timer_data = myTimer('Saving files')
            save_search_results_one_pdb(loop_type,query_pdb_id, results, save_path)

    return(search_results)
    # save results from the last set of searches
    #may not need anymore

def cluster_loops(search_results,all_loops_ids,ratio=1,show_dq_codes = False):
    '''
    Code from Seyoung.
    Create a discrepancy matrix of 0 and 1
    0 means discrepancy >1, 1 means discrepancy <1.

    '''
    MM_dimension = len(all_loops_ids)

    MM = np.zeros(shape=(MM_dimension,MM_dimension))
    disc_m = np.zeros(shape=(MM_dimension,MM_dimension))
    dq_matrix = [[-1] * MM_dimension for i in range(MM_dimension)]

    for i in range(0,MM_dimension):
        loop_i_id = all_loops_ids[i]
        for j in range(i+1,MM_dimension):
            loop_j_id = all_loops_ids[j]
            #If the value of the tuple is an integer -> no match
            i_j_disqualification = search_results[(loop_i_id,loop_j_id)][0]['dq']
            j_i_disqualification = search_results[(loop_j_id,loop_i_id)][0]['dq']

            i_j_discrepancy = search_results[(loop_i_id,loop_j_id)][0]['discrepancy'] 
            j_i_discrepancy = search_results[(loop_j_id,loop_i_id)][0]['discrepancy']
            test=search_results[(loop_j_id,loop_i_id)]
            MM[i][j] = 1 if (len(i_j_disqualification) == 0 or len(j_i_disqualification) == 0) else 0
            MM[j][i] = MM[i][j]

            if MM[i][j] ==1:
                disc_m[i][j] = min(i_j_discrepancy,j_i_discrepancy)
                disc_m[j][i] = disc_m[i][j]
            else:
                dq_matrix[i][j] = i_j_disqualification[0]
                dq_matrix[j][i] = j_i_disqualification[0]
    G = nx.from_numpy_matrix(MM)
    cliques = list(nx.find_cliques(G))
    '''
    Check dq code between motif groups
    '''
    groups = []
    groups = cluster_motifs3(cliques,disc_m,ratio,all_loops_ids)
    #heat_map(cliques,groups,disc_m)
    # if show_dq_codes:
    #     get_dq_between_motif_groups(cliques,dq_matrix,groups)
    return groups

def analyze_single_bulged_base_loops(candidates,query, search_space):
    not_rejected = []
    for candidate in candidates:
        DQ_code = [] # change logic to use DQ_code as a set? saves lots of operations, Adam
        query_bulge,search_space_bulge = get_bulge(query,search_space)
        if query_bulge != search_space_bulge:
            DQ_code += [MISMATCHED_BULGE]

        filtered_candidates = {}
        
        set_DQ_code = set(DQ_code)
        # proposed solution, in bottom of "for candidate"
        if len(set_DQ_code) == 0:
            filtered_candidates['dq'] = []
            filtered_candidates['query_id'] = query['loop_info']['loop_id']
            filtered_candidates['search_space_id'] = search_space['loop_info']['loop_id']
            filtered_candidates['indices'] = candidate['indices']
            filtered_candidates['query_unit_ids'] = query['Q']['unitID']
            filtered_candidates['target_unit_ids'] = candidate['unitids']
            filtered_candidates['discrepancy'] = candidate['discrepancy']
            filtered_candidates['numStrands'] = query['Q']['numStrands']

            filtered_candidates['query_rfam'] = query['loop_info']['rfam']
            filtered_candidates['search_space_rfam'] = search_space['loop_info']['rfam']

            not_rejected.append(filtered_candidates)

        else: # if has a real DQ_code
            filtered_candidates['dq'] = list(set_DQ_code)
            filtered_candidates['discrepancy'] = candidate['discrepancy']
            filtered_candidates['numStrands'] = query['Q']['numStrands']

            not_rejected.append(filtered_candidates)

    temp_result = find_lowest_discrepancy(not_rejected)
    return(temp_result)

def orderBySimilarity(clusters,search_results):
    reordered_clusters = []
    reordered_dist_matrices =[]
    singletons = []
    for cluster in clusters:
        if len(cluster) >1:
            dist_matrix = np.zeros(shape=(len(cluster),len(cluster)))
            reordered_dist_matrix =np.zeros(shape=(len(cluster),len(cluster)))

            for i in range(0,len(cluster)):
                loop_i = cluster[i]
                for j in range(i+1,len(cluster)):
                    loop_j=cluster[j]
                    if len(search_results[(loop_i,loop_j)][0]['dq']) == 0:
                        dist_matrix[i][j] = search_results[(loop_i,loop_j)][0]['discrepancy']
                        dist_matrix[j][i] = dist_matrix[i][j]
                    else:
                        dist_matrix[j][i] = search_results[(loop_j,loop_i)][0]['discrepancy']
                        dist_matrix[i][j] = dist_matrix[j][i]
            order=treePenalizedPathLength(dist_matrix)
            for i in range(0,len(cluster)):
                for j in range(0,len(cluster)):
                    reordered_dist_matrix[i][j]=dist_matrix[order[i]][order[j]]
            reordered_cluster = [cluster[x] for x in order]
            reordered_clusters.append(reordered_cluster)
            reordered_dist_matrices.append(reordered_dist_matrix)
        else:
            reordered_clusters.append(cluster)
            reordered_dist_matrices.append(np.zeros(shape=(1,1)))

        #ax = sns.heatmap(dist_matrix)
        #plt.show()
        
    return(reordered_clusters,reordered_dist_matrices)

def align_nts(search_results,queries,clusters):
    core_positions = defaultdict()
    cluster_name = 0
    for cluster in clusters:
        core_positions[cluster_name] = {}

        min_nts = 1000
        centroid_candidates =[]
        centroid = ""
        #CENTROID CANDIDATES
        #FIND ALL CANDIDATES THAT HAVE LOWEST NUMBER OF NONBULGED NTS
        for candidate_id in cluster:
            numpositions = sum(len(nts) for nts in queries[candidate_id]["loop_info"]["strand"])
            bulged_nts = len(queries[candidate_id]["loop_info"]["bulged"])
            #SPECIAL CASE FOR SINGLE BULGED BASE:
            # if numpositions == 5 and bulged_nts ==1:
            #     centroid_candidates.append(candidate_id)
            #     break #BREAK OUT OF THE LOOP BECAUSE ANY CANDIDATE CAN BE CENTROID.
            non_bulged_nts = numpositions - bulged_nts
            if non_bulged_nts < min_nts:
                min_nts = non_bulged_nts
                centroid_candidates = [candidate_id]
            elif non_bulged_nts == min_nts:
                centroid_candidates.append(candidate_id)
        #CENTROID
        #FIND THE CANDIDATE THAT HAS THE LOWEST TOTAL DISCREPANCIES TO OTHER INSTANCES
        min_total_disc = float('inf')
        for query in centroid_candidates:
            total_disc = 0
            for target in cluster:
                if query == target:
                    continue
                if search_results[(query,target)][0]['discrepancy'] != 99:
                    total_disc+=search_results[(query,target)][0]['discrepancy']
                else:
                    total_disc+= search_results[(target,query)][0]['discrepancy'] + 1000
                    #TO RANK CANDIDATES
                
            if total_disc < min_total_disc:
                min_total_disc = total_disc
                centroid = query
        
        #STORE THE ALIGNMENT OF ALL INSTANCES BASED ON THE CENTROID'S SEARCH RESULT
        #TO SEARCH_RESULTS[(CANDIDATE,CANDIDATE)][0]['QUERY_UNIT_IDS']
        for candidate in cluster:
            if candidate == centroid:
                core_positions[cluster_name][centroid]=queries[centroid]["Q"]["unitID"]
                continue

            centroid_to_candidates={}
            a = search_results[(centroid,candidate)][0]
            if "target_unit_ids" in search_results[(centroid,candidate)][0]:
                target_unit_ids = search_results[(centroid,candidate)][0]["target_unit_ids"]
                query_unit_ids = search_results[(centroid,candidate)][0]["query_unit_ids"]
                for i in range (0,len(target_unit_ids)):
                    centroid_unit_id = query_unit_ids[i]
                    candidate_unit_id = target_unit_ids[i]
                    centroid_to_candidates[centroid_unit_id] = candidate_unit_id                    
            else: 
                target_unit_ids = search_results[(candidate,centroid)][0]["target_unit_ids"]
                query_unit_ids = search_results[(candidate,centroid)][0]["query_unit_ids"]
                for i in range (0,len(target_unit_ids)):
                    centroid_unit_id = target_unit_ids[i]
                    candidate_unit_id = query_unit_ids[i]
                    centroid_to_candidates[centroid_unit_id] = candidate_unit_id                

            core_positions[cluster_name][candidate] = []
            for centroid_unit_id in centroid_to_candidates:
                core_positions[cluster_name][candidate].append(centroid_to_candidates[centroid_unit_id])

        cluster_name+=1
            
    return(core_positions)

def main(loops = None, pair_to_interaction_list = None):
    #loops_and_strands=load_loops_and_strands()
    
    queries = {}
    search_spaces ={}
    flanking_bp_queries = {}

    load_all_previous_results = True
    timer_data = myTimer("start")
    timer_data = myTimer('Set up loops')
    
    #single bulged A and single bulged U
    #http://rna.bgsu.edu/rna3dhub/motif/view/IL_80007.15
    #http://rna.bgsu.edu/rna3dhub/motif/view/IL_83039.10
    loops_of_interest = [loop_id for loop_id,alignment in loops_and_strands.items() if len(alignment) == 5]
    loops_of_interest = ['IL_7RQB_111','IL_5J7L_302','IL_4WF9_002', 'IL_4V88_409', 'IL_4V9F_002']
    loops_of_interest = [loop_id for loop_id,alignment in loops_and_strands.items()]
    #download_and_save_loops_and_strands_json_file(3.57,"./data")
    loop_type = set_loop_type(loops_and_strands)

    if(load_all_previous_results):
        search_results,all_loops_ids = load_all_search_results(loop_type)
        queries = load_all_queries(loop_type)
        python_clustered_motif_groups = cluster_loops(search_results,all_loops_ids,0.1)
        reordered_python_clustered_motif_groups,reordered_dist_matrices = orderBySimilarity(python_clustered_motif_groups,search_results)

        #reordered_python_clustered_motif_groups = load_all_clusters(loop_type)
        #reordered_dist_matrices = load_dist_matrices(loop_type)
        #reordered_python_clustered_motif_groups,reordered_dist_matrices = orderBySimilarity(reordered_python_clustered_motif_groups,search_results)
        #validate_clusters(search_results,loops_of_interest,reordered_python_clustered_motif_groups)
        show_discrepancy(search_results, ['IL_6CZR_120'],['IL_1QBP_002','IL_1IHA_001','IL_1ICG_001'])
        core_positions = align_nts(search_results,queries,reordered_python_clustered_motif_groups)
        #save_all_search_results(search_results,all_loops_ids) #MAYBE LATER
        #validate_clusters(search_results,loops_of_interest,reordered_python_clustered_motif_groups)
        Q= defaultdict(list)
        Q["name"] = "testing"
        Q["numFilesSearched"] = 5
        Q["searchFiles"]= 'searchFiles'
        Q["elapsedCPUTime"] = 5
        for i in range(0,len(reordered_python_clustered_motif_groups)):
            if(len(reordered_python_clustered_motif_groups[i]) > 1):
                writeHTMLOutput(loop_type,search_results,reordered_python_clustered_motif_groups,core_positions[i],Q,reordered_python_clustered_motif_groups[i],queries,reordered_dist_matrices[i],i)
    else:
        print("Testing %d loops of interest" % len(loops_of_interest))
        loops, pair_to_interaction_list = startup_list_of_dictionaries(loops_and_strands,loops_of_interest)
        # for each loop, set it up as a query Q and as a search space ifedata
        # we are going to search for every loop inside of every other loop

        # sort the list of loops by loop id
        loops = sorted(loops, key = lambda x : x['loop_id'])

        timer_data = myTimer("Set up queries and ifedata",timer_data)
        print("Set up queries and search space data")

        queries,search_spaces = create_queries_and_search_spaces(loops,loops_of_interest)
        flanking_bp_queries = create_flanking_bp_queries(loops,loops_of_interest)
        save_all_queries(loop_type,queries)

        print(myTimer("summary"))

        print("Start all against all searches")
        timer_data = myTimer("All against all searches",timer_data)

        # to search faster, run two instances with True/False on next line
        reversed_search = True

        # load results of previous searches, must faster to use True
        load_previous_result = True

        search_results = all_against_all_searches(loop_type,queries,search_spaces,flanking_bp_queries,load_previous_result,reversed_search)
        all_loops_ids = list(queries.keys())

        timer_data = myTimer("Cluster loops",timer_data)
        print("Clustering loops")
        ### The last argument for cluter_loops is the ratio
        python_clustered_motif_groups = cluster_loops(search_results,all_loops_ids,0.1)
        reordered_python_clustered_motif_groups,reordered_dist_matrices = orderBySimilarity(python_clustered_motif_groups,search_results)

        # save .pickle files in DATAPATH/new_results dictory
        save_dist_matrices(loop_type,reordered_dist_matrices)
        save_all_clusters(loop_type,reordered_python_clustered_motif_groups)
        save_all_search_results(loop_type,search_results,all_loops_ids)

        print()

    # get statistics about agreement between Python clusters and Matlab motif groups
    # output is written to DATAPATH/new_results directory
    timer_data = myTimer("Validate clusters",timer_data)
    #show_discrepancy(search_results, ['IL_3IGI_011'],['IL_4V9F_007'])
    validate_clusters(loop_type,search_results,loops_of_interest,reordered_python_clustered_motif_groups)

    print()
    print(myTimer("summary")) 

if __name__ == '__main__':
    res = main()

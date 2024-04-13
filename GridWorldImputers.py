import numpy as np
import copy
from collections import Counter

import GridWorldHelpers as gwh



############################
# Little Helpers
############################
def shuffle(p):
    u = np.random.uniform()
    if u <= p:
        return True
    else:
        return False 

############################
# Principled Approach
############################


def init_Tstandard(d, colors = [0,1,2], init_value = 0): 
    """
    
    Parameters
    ----------
    d : dimension of grid
    
    colors : list of color codes
    
    init_value: count to initialize with
  
    
    Returns
    -------
    Tstandard: a dict where keys are (S_t,A_t) aka tuples of the form ((y,x,c),(a1,a2))
    and values are dictionaries with keys formed by (S_{t+1}) aka tuples of the form
    (y,x,c)
    
    Example:
        
        Tstandard[((1,1,1),(0,1))][(0,0,0)]
        
        accesses the count for times was at (1,1) on the grid with color 1, 
        took action "up" (0,1) and ended up at (0,0,0)
        
        *Note that in practice many transitions are impossible and should stay at 0
        but we initialize with all possibilities so that this matrix is (a) not world
        specific and (b) to reflect that we do not actually give the agent this logical
        information
    
    
    #TODO: think about consequences of how initialize this -- if do all to 0, then good
    because those that are truly 0 will stay 0 but bad because will never impute a transition
    until have observed it at least once. Maybe you want to initilize this to \epsilon?
    
    
    """
    # get possible actions
    action_descs = gwh.load_actions()
    
    inner_S_dict = {((i,j,c)) : init_value for i in range(d) for j in range(d) for c in colors}


    T = {((i, j, c), action) : copy.deepcopy(inner_S_dict) for i in range(d) for j in range(d) 
                for c in colors for action in list(action_descs.keys())}

    return(T)    
    

def sample_entry(dic):
    """
    Given dictionary dic with non-negative integers or floats
    values, draw a key at random with probabilities proportional to 
    normalized values
    
    If all counts are 0, select one at random
    """
    options = list(dic.keys()) #using fact this keeps order
    counts = np.array(list(dic.values()))  
    
    if all(counts == 0):
        counts = np.ones(len(counts))

    probs = counts/sum(counts)
    j = np.random.choice(range(len(options)),1,1,probs)[0]
    return(options[j])

def draw_Tstandard(Tstandard,S,A, pobs_state):
    """
    Given transition matrix and current state and action, draw at random a
    new state

    Parameters
    ----------
    Tstandard : dictionary as output by init_Tstandard()

    S : tuple describing state
    A : tuple describing action
    pobs_state : observed state

    Returns
    -------
    S' : tuple describing the next state
    
    """
    assert type(S) == tuple, "S is not tuple"
    assert type(A) == tuple, "A is not tuple"  
    assert type(pobs_state) == tuple, "pobs_state is not a tuple"
    
    #O state missingness information
    miss_vec = np.isnan(pobs_state)
    num_miss = np.sum(miss_vec)
    where_no_miss = np.where(~miss_vec)[0]
   
    # if fully observed, return state 
    if num_miss == 0:
        return pobs_state

    rel_dict = Tstandard[(S,A)]
    
    #filter based on observed states
    if num_miss  > 0:
        #Get relevant dictionary based on last state and action
        keys = [elem  for elem in rel_dict.keys() if all([elem[i] == pobs_state[i] for i in where_no_miss]) ]
        rel_dict = {k:v for (k,v) in rel_dict.items() if k in keys}
       
    Istate = sample_entry(rel_dict)
    #assert [pobs_state[i] == Istate[i] for i in where_no_miss], "failed to maintain observed state"

    return(Istate)
  

def Tstandard_update(Tstandard, Slist, A, new_Slist):
    """
    Updates Tstandard (S,A,S') transition counts
    using the i^{th} entry of Slist as S and the i^th entry
    of new_Slist as S' 
    
    Note that the amount added to each count is 1/K where
    K = len(Slist) so 1 is fractionally allocated according to how
    often each (S,'S) pair occurs 
                
    Modifies matrix in place
    
    """
    K = len(Slist)
    for k in range(K):
        #comment: if really going to do this addition many times
        #have to worry about numeric error accumulating
        Tstandard[(Slist[k],A)][new_Slist[k]] += 1/K
    
       
    

    


  


########################################################################33
# MICE approach that doesn't use joint info but is lower dimensional
########################################################################33

def init_Tmice(d, colors = [0,1,2], init_value = 0 ):
    """

    Parameters
    ----------
    d : dimension of grid
    
    colors : list of color codes
    
    init_value: count to initialize with
    
    Returns
    -------
    dict with keys 0,1,2 where
    
    
    0 stands for y coordinate and value is:
           
        dict where keys are tuples of the form ((y,x,c),(a1,a2),(y,c))
        and values are dictionaries with keys 0,...,d-1 for each of the possible
        values of x
        
    1 stands for "y" value.
        
        it is same as above only keys are tuples of form ((y,x,c),(a1,a2),(x,c))
    
    2 stands for "c" value
        
        it is same as above only keys are tuples of form ((y,x,c),(a1,a2),(y,x))


    Note: the raeson (y,x) are not the typical (x,y) order is because of how Numpy does
    (row,column) indexing 

    """
    action_descs = gwh.load_actions()
   
    x_dict = {((i, j, c), action, (u,v)) :  {i: init_value for i in range(d)} for i in range(d) for j in range(d) 
                for c in colors for action in list(action_descs.keys()) for u in range(d) for v in colors}

    y_dict = x_dict.copy() #same structure 
    
    c_dict = {((i, j, c), action, (u,v)) :  {c: init_value for c in colors} for i in range(d) for j in range(d) 
                for c in colors for action in list(action_descs.keys()) for u in range(d) for v in range(d)}

    return {0:y_dict,
            1:x_dict, 
            2:c_dict}


def draw_Tmice(Tmice, S, A, focal, Scomplete = None):
    """
    Given transition matrix, current state and action, 
    a complete next state vector Scomplete and a focal state
    (the one that was originally missing that we will now
      re-draw), draw S[focal] | S, A, S[focal] using the 
    probabilities implied by the counts in the Tmice[focal] matrix
    

    Parameters
    ----------
    Tmice : dictionary as output by init_Tmice()

    S : tuple describing state
    A : tuple describing action
    Scomplete: a tuple describing state
    focal : an integer in [0,1,2] 
  
    Returns
    -------
    if Scomplete is not None:
        Scomplete with Scomplete[focal] updated to a possibly new
        value drawn according to Tmice
    
    if Scomplete is None:
        
        a new draw of focal |S,A  (just the single element)
    
    """
    assert type(S) == tuple, "S is not tuple"
    assert type(A) == tuple, "A is not tuple"
    assert type(Scomplete) == tuple or Scomplete is None, "Scomplete is not a tuple"
    assert focal in [0,1,2], "focal out of range"
    
    #if given, you draw from focal given other entries in Scomplete    
    if Scomplete is not None:
        #note: OK for Scomplete to be a tuple but returns an array
        Snf = tuple(np.delete(Scomplete, focal)) #S non-focal
        rel_dict = Tmice[focal][(S,A, Snf)]
        new_entry = sample_entry(rel_dict)
        
        #replace old value with the new sampled entry
        Scomplete = np.array(Scomplete) #in case it is tuple
        Scomplete[focal] = new_entry
        return(tuple(Scomplete))
    
    #otherwise, draw from marginal x|S,A type distribution
    else:
        keys = [elem for elem in Tmice[focal].keys() if (S in elem and A in elem)]
        dicts = [v for (k,v) in Tmice[focal].items() if k in keys]
        counter = Counter()
        for d in dicts: 
            counter.update(d)
        rel_dict = dict(counter)
        new_entry = sample_entry(rel_dict)

        return(new_entry)

    
def draw_mouse(Tmice, S, A, pobs_state, num_cycles = 10):
    """
    Function for doing mice for a single K
    """
    miss_vec = np.isnan(pobs_state)
    num_miss = np.sum(miss_vec)
    where_miss = np.where(miss_vec)[0]
    where_no_miss = np.where(~miss_vec)[0]
   
    
    # if fully observed, return state 
    if num_miss == 0:
        return pobs_state
    
    # draw initially over marginal x | S,A type dist
    Istate = [0,0,0]
    for k in where_miss:
        Istate[k] = int(draw_Tmice(Tmice, S, A, focal = k))
    for k in where_no_miss:
        Istate[k] = pobs_state[k]

    # cycle through draws of conditionals
    Istate = tuple(Istate)    
    for n in range(num_cycles):
         for k in where_miss:
             Istate = draw_Tmice(Tmice, S, A, Scomplete = Istate, focal = k)
             if len(where_miss) == 1:
                 break #only need to draw once then
    
    check = [pobs_state[i] == Istate[i] for i in where_no_miss]
    if len(check) > 0:
        assert all(check), "failed to maintain observed state"
    return(Istate)


#########################################################
# Multiple Imputation Functions
#########################################################


def MI(method, Slist, A, pobs_state, shuffle = False,
                 Tstandard = None, Tmice = None, num_cycles = None):
    """
    Given K = len(Slist) imputations previous step, for each draw a new
    imputation using draw_Tstandard() or draw_mouse()
    
    Optionally randomly reshuffle the order. This will have the
    effect of mixing the chains in terms of how the Q gets updated
    but maybe won't have a huge effect since otherwise the 
    Slist is not used going forward


    """
    assert method in ["joint","mice"], "invalid method specified"
    if method == "joint":
        assert Tstandard is not None
    if method == "mice":
        assert Tmice is not None and num_cycles is not None
    
    K = len(Slist)
    NewSlist = [0]*K
    for i in range(K):
        if method == "joint":
            NewSlist[i] = draw_Tstandard(Tstandard,Slist[i],A, pobs_state)
        if method == "mice":
            NewSlist[i] = draw_mouse(Tmice, Slist[i], A, pobs_state, num_cycles = num_cycles)
                
    if shuffle:
        np.random.shuffle(NewSlist) #modifies in place
        

    return(NewSlist)

    
def Tmice_update(Tmice, Slist, A, new_Slist):
    """
    Updates Tmice marginal transition counts
    using the i^{th} entry of Slist as S and the i^th entry
    of new_Slist as S' 
    
    Note that the amount added to each count is 1/K where
    K = len(Slist) so 1 is fractionally allocated according to how
    often each (S,'S) pair occurs 
    
    """
    K = len(Slist)
    #for each of the conditionals
    for r in [0,1,2]:
        others = tuple(np.delete([0,1,2],r))
        #for each chain
        for k in range(K):
            partial = (new_Slist[k][others[0]],new_Slist[k][others[1]])
            Tmice[r][(Slist[k],A,partial)][new_Slist[k][r]] += 1/K   
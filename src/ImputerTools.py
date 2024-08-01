import numpy as np
import copy
import itertools
from collections import Counter


############################
# Little Helpers
############################
def shuffle(p):
    u = np.random.uniform()
    if u <= p:
        return True
    else:
        return False 
    
######################################
# Taking actions based on Q.
######################################


# function for initializing our Q matrix as all zeroes
def init_Q(
           state_value_lists, 
           action_list, 
           include_missing_as_state = False, 
           missing_as_state_value = -1):
    """
    Parameters
    ----------
    state_value_lists : list of lists
        each sublist corresponds to a dimension of state space
        elements of each sublist are the possible state values
        
        e.g. [[1,2], [1,2], [1,2,3]] reflects a 3-D state space where
        all combinations in {1,2} x {1,2} x {1,2,3} form the state space
        
    action_list : list of possible actions, encoded as integers or tuples

    include_missing_as_state : bool, optional, default False
        If True, includes 'missing' as a possible state for each
        dimension of state space. Missing takes value <missing_as_state_value>
        with default -1.
        
    Returns
    -------
    Q : dictionary where keys are a tuple representing a possible (s,a) state
    action pair and values are 0.0. Dictionary contains all state-action pairs
    as encoded by state_value_lists and action_list

    """
    if include_missing_as_state:
        state_value_lists = [elem + [missing_as_state_value] for elem 
                             in state_value_lists]
        
    Q = {(elem, a) : 0.0 
         for elem in itertools.product(*state_value_lists)
         for a in action_list}
        
    return Q



def select_action(state, action_list, Q, epsilon, option = "voting2"):
    """
    Function select actions based on an epsilon greedy policy 
    (or greedy if psilon = 0), maximizing over the Q function formatted
    as output by init_Q

    Parameters
    ----------
    state : EITHER 
        - option 1: tuple encoding state to select action for
        - option 2: list of states from multiple imputation to incorporate into
                    action selection procedure

    Q : Q matrix

    epsilon : int or float >= 0
    
    option : if state is of type typle, this is ignored. If state if of type 
             list, then two options
             
             (1) voting1 - get the Q-maximizing action for each state and then 
                 pick the most voted action, breaking ties at random
                 
             (2) voting2 - get the Q-maximizing action for each state and then 
                 pick among them at random
                 
            (3) averaging - for each action, calculate the mean Q function over
                states and then take the action with maximum mean Q
                 

    Returns
    -------
    an action encoded as a length 2 tuple
    """
    assert epsilon >= 0
    
    if type(state) is tuple:
        # get Q value for each action for that state
        Qvals = [Q[(state, a)] for a in action_list]
        # get where the max Q values are
        max_indices = np.where(Qvals == np.max(Qvals))[0]
        
        
    if type(state) is list:
        
        if "voting" in option:
            # get Q value for each action for each state 
            Qvals = [[Q[(s, a)] for a in action_list] for s in state]
            # get where max Q values are for each state, breaking ties randomly
            maxes = [np.where(v == np.max(v))[0] for v  in Qvals]
            maxes = [v[np.random.choice(len(v))] for v in maxes] 
            if option == "voting1":
                counts = Counter(maxes)
                max_count = counts.most_common(1)[0][1] #what is highest count?
                max_indices = [elem for elem, count in counts.items() if count == max_count]
            if option == "voting2": 
                max_indices = maxes 
          
        if option == "averaging":    
            action_means = [np.mean([Q[(s,a)] for s in state]) for a in action_list]
            max_indices = np.where(action_means == np.max(action_means))[0]
    
    # what is the "greedy" action index? break ties by randomly selecting one
    greedy_idx = max_indices[np.random.choice(len(max_indices))]

    # actually pick our action index based on epsilon greedy
    action_idx = greedy_idx if np.random.uniform() > epsilon else np.random.choice(len(action_list))

    # return our action
    return action_list[action_idx]
        
        


# function for updating Q
def update_Q(Q, state, action, action_list, reward, new_state, alpha, gamma):
    """
    Given Q function, state, action, reward and next state, do a 
    standard Q update
    """
  
    # make a copy of Q
    Qnew = copy.deepcopy(Q)
    
    # figure out optimal Q-value on S'
    optQ = np.max([Q[new_state, a] for a in action_list])
    
    # update our q-entry
    Qnew[state, action] += alpha * (reward + (gamma*optQ) - Q[state, action])
    
    # return our Q
    return Qnew
    

############################
# Principled Approach
############################


def init_Tstandard(state_value_lists, 
                   action_list, 
                   init_value = 0.0): 
    """
    
    Parameters
    ----------
    state_value_lists : list of lists
        each sublist corresponds to a dimension of state space
        elements of each sublist are the possible state values
        
        e.g. [[1,2], [1,2], [1,2,3]] reflects a 3-D state space where
        all combinations in {1,2} x {1,2} x {1,2,3} form the state space
        
    action_list : list of possible actions, encoded as integers or tuples
    
    init_value: count to initialize with
  
    
    Returns
    -------
    Tstandard: a dict where keys are (S_t,A_t) and
    values are dictionaries with keys formed by (S_{t+1})
    
    Example:
        
        Tstandard[((1,1,1),(0,1))][(0,0,0)]
        
        accesses the count for times was at (1,1,1), took action (0,1), and ended up
        at (0,0,0)
        
        *Note that in practice some transitions may be impossible and should stay at 0
        but we initialize with all possibilities so that this matrix is (a) not world
        specific and (b) to reflect that we do not actually give the agent this logical
        information    
        
        If you know that all transitions are in principle possible, you may want to pick
        init_value = \eta > 0 rather than the default of 0.        
    """
    inner_S_dict = {elem : init_value for elem in itertools.product(*state_value_lists)}    
    T = {(elem,a):copy.deepcopy(inner_S_dict) 
     for elem in itertools.product(*state_value_lists) 
     for a in action_list}
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
        
        
#########################################################
# Multiple Imputation Functions
#########################################################


def MI(method, last_state_list, last_A, pobs_state, shuffle = False,
                 Tstandard = None, Tmice = None, num_cycles = None):
    """
    Given K = len(last_state_list) imputations from previous step and the previous action,
    for each draw a new imputation using draw_Tstandard() or draw_mouse()
    
    Optionally randomly reshuffle the order. This will have the
    effect of mixing the chains in terms of how the Q gets updated
    but maybe won't have a huge effect since otherwise the 
    last_state_list is not used going forward

    """
    assert method in ["joint","mice", "joint-conservative"], "invalid method specified"
    if method == "joint" or method == "joint-conservative":
        assert Tstandard is not None
    if method == "mice":
        assert Tmice is not None and num_cycles is not None
    
    K = len(last_state_list)
    new_state_list = [0]*K
    for i in range(K):
        if method == "joint" or method == "joint-conservative":
            new_state_list[i] = draw_Tstandard(Tstandard,last_state_list[i], last_A, pobs_state)
        if method == "mice":
            new_state_list[i] = draw_mouse(Tmice, last_state_list[i], last_A, pobs_state, num_cycles = num_cycles)
                
    if shuffle:
        np.random.shuffle(new_state_list) #modifies in place
        

    return(new_state_list)
       
#######################################
# Q update in MI case
#######################################

def updateQ_MI(Q, Slist, new_Slist, A, action_list, reward, alpha, gamma):
    """
    Given multiple imputations, update Q fractionally allocating updates
    with alpha/K learning rate where K is length of Slist
    """
    assert len(Slist) == len(new_Slist)
    K = len(Slist)
    for k in range(K):
        Q = update_Q(Q, 
                     state = Slist[k],
                     action = A,
                     action_list = action_list,
                     reward = reward,
                     new_state = new_Slist[k],
                     alpha = alpha/K,
                     gamma = gamma)
    return(Q)    

    
########################################################################33
# MICE approach that doesn't use joint info but is lower dimensional
########################################################################33

def init_Tmice(state_value_lists, action_list,  init_value = 0):
    """
    Parameters
    ----------
    state_value_lists : list of lists
        each sublist corresponds to a dimension of state space
        elements of each sublist are the possible state values
        
        e.g. [[1,2], [1,2], [1,2,3]] reflects a 3-D state space where
        all combinations in {1,2} x {1,2} x {1,2,3} form the state space
        
    action_list : list of possible actions, encoded as integers or tuples
    
    init_value: count to initialize with

    Returns
    -------
    dict with a key for each element of state_value_lists. 
    values of that dict are dictionaries with all combos of (s,a,s') 
    excluding the dimension of s' that this outer key is for                                                       
    """ 
    T = {}
    for i, dim in enumerate(state_value_lists):

        sub_list = state_value_lists[:i] + state_value_lists[i+1:]
        target = state_value_lists[i]
        sub_dict = {(elem,a,elem2) : {e:0 for e in target}
                    for elem in itertools.product(*state_value_lists)
                    for a in action_list
                    for elem2 in itertools.product(*sub_list) 
                   }
        T[i] = sub_dict

    return(T)


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

    
def Tmice_update(Tmice, Slist, A, new_Slist):
    """
    Updates Tmice marginal transition counts
    using the i^{th} entry of Slist as S and the i^th entry
    of new_Slist as S' 
    
    Note that the amount added to each count is 1/K where
    K = len(Slist) so 1 is fractionally allocated according to how
    often each (S,'S) pair occurs 
    
    """
    assert len(Slist) == len(new_Slist)
    K = len(Slist)
    #for each of the conditionals
    for r in [0,1,2]:
        others = tuple(np.delete([0,1,2],r))
        #for each chain
        for k in range(K):
            partial = (new_Slist[k][others[0]],new_Slist[k][others[1]])
            Tmice[r][(Slist[k],A,partial)][new_Slist[k][r]] += 1/K   



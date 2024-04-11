import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, copy, os, shutil
from tqdm.notebook import tqdm
import seaborn as sns


# Global Parameters  #TODO: capitalize
action_descs = {(0, 0) : "stay", 
                (0, 1) : "up", 
                (1, 1) : "diag-right-up", 
                (1, 0) : "right", 
                (1, -1): "diag-right-down", 
                (0, -1) : "down", 
                (-1, -1) : "diag-left-down",
                (-1, 0) : "left", 
                (-1, 1): "diag-left-up"}

# what are the actual list of actions that are possible?
ACTIONS = list(action_descs.keys())

# just to make sure we can get it quickly
def load_actions():
    return action_descs.copy()

# function for initializing our Q matrix, assuming 3 colors
def init_Q(d, init_value=0.0):
    
    # start our Q
    Q = {((i, j, c), action) : init_value for i in range(d) for j in range(d) 
         for c in range(3) for action in list(action_descs.keys())}
    
    # encode our missing-state option, too
    for action in list(action_descs.keys()):
        Q[("missing", action)] = init_value
        
    # return our Q matrix
    return Q




####################
# Environment Set-up
####################



def build_grids(d, baseline_penalty = -1, 
                water_penalty = -10,
                end_reward = 10):
    """
    Build grid worlds with state vector (y,x,c) characterized by the presence
    of water (negative reward) and dry land (positive reward) as well
    as a terminal state that is the target place the agent needs to navigate to 
    
    - (y,x) are the location in a 2-D plane
        flipped because of Numpy's indexing by (row, column)
    
    - (c)  is color (a way of adding a  visualizable 3rd dimension). 
        The idea is that color does not affect geography but 
        gives a signal (red, orange, green) related to 
        the safety of the current state and the safegy or 
        danger to move right (but not any other direction)

    
    Parameters
    ----------
    d : int 
        dimension of square grid world

    Returns
    -------
    A tuple of mean reward grids


    #TODO 
    CAUTION:  this is not yet fully genearlized to any d. It works as expected for
    d=8  but maybe not fpr other values
    
    """
    # baseline grid
    gw0 = np.full(shape=(d, d), fill_value=-1.0); gw0[0, -1] = end_reward

    bridge_height = d-2

    # bridge grid world (no pond overflow)
    gw1 = np.full(shape=(d, d), fill_value= baseline_penalty)
    gw1[:, (d-5):(d-3)] = water_penalty  #the water - width
    gw1[bridge_height,(d-5):(d-3)] = baseline_penalty #the bridge
    gw1[0:3,:] = -1.0;  #clear water on top
    gw1[bridge_height, -1] = +end_reward #final spot

    # bridge grid world (yes pond overflow)
    gw2 = np.full(shape=(d, d), fill_value= baseline_penalty)
    gw2[:, (d-6):(d-2)] = water_penalty #the water - width - one wider
    gw2[bridge_height,(d-6):(d-2)] = baseline_penalty #the bridge
    gw2[0:2,:] = -1.  #clear water on top (make it one wider)
    gw2[bridge_height, -1] = +end_reward
    
    return(gw0, gw1, gw2)



def visualize_reward_grid(gw, ax):
    """
    Use heatmap to visualize (mean) rewards of the given grid on the 
    given axes
    """
    sns.heatmap(gw, ax = ax, cmap="viridis", annot=True, cbar=False)



def get_environment(ce, p, indices):
    """
    Function which takes current environment and
    
    * with probability p, from ce to 1-ce
    * with probability (1-p), keep environment constant
    
    Meant for toggling between overflow state and non-overflow state

    Parameters
    ----------
    ce : current environment - an index in indices
    
    p : float in [0,1]
    
    indices : np.array with possible indices - currently must be length 2
    
    Returns
    -------
    ce for no switch
    other index in indices for switch
    """
    u = np.random.uniform()
    if u < p: 
        return indices[indices != ce][0]
    else: 
        return ce
    
    #return 1 if np.random.uniform() < p_overflow else 2
    

def make_gw_colors(gw):
    """
    Encodes color of each state in grid according to following 
    principle:
    
    (0) Green = currently not in water and there is no water to the right
    
    (1) Orange = currently in water but moving right moves out of water
    
            OR
            
            currently not in water but moving right moves into water
            
            Note: idea is that agent will have to learn joint relationships
            where some (x, orange) mean go right and some (x,orange) mean
            don't go right. 
             
    (2) Red = currently in water and moving right stays in water
    

    Parameters
    ----------
    gw : a grid world as output by build_grids() 

    Returns
    -------
    gw_colors : a grid of same dimension as gw
    encoding colors for each (x,y) in grid. 
    0 = green, 1 = orange, 2 = red       
    """
    d  = gw.shape[0]

    # instantiate a color environment as all green.
    gw_colors = np.full(shape=(d,d), fill_value=0.0)

    # if-then rules for encoding colors
    for i in range(d):
        for j in range(d):

            # figure out what my location would be if I took a step right
            j_right = j+1 if j != d-1 else j

            # safe to go right: green
            if gw[i,j] >= -1 and gw[i,j_right] >= -1:
                gw_colors[i,j] = 0
            # two orange situations
            elif gw[i,j] == -10 and gw[i,j_right] == -1:
                gw_colors[i,j] = 1
            elif gw[i,j] == -1 and gw[i,j_right] == -10:
                gw_colors[i,j] = 1
            # red situation
            elif gw[i,j] == -10 and gw[i,j_right] == -10:
                gw_colors[i,j] = 2
                
    # return the colors
    return gw_colors



##############################################
# Implementing Actions in True State Space
##############################################

# to account for wind in both i and j directions
def wind(state, d, p_wind_i, p_wind_j):
    """
    Function which randomly adds some wind 
    pushing unit left/right/up/down an additional step
  
    p_wind_i is prob of a y direction move
    p_wind_j is prob of a x direction move
    
    given we do move, left vs right or up vs down is
    selected with probabilities (1/2,1/2)

    """
    
    # make sure state is an np.array
    state = np.array(state)
    
    # make a copy of our state
    wind_state = state.copy()
    
    # determine our i direction perturbation
    if np.random.uniform() < p_wind_i:
        wind_state[0] += np.random.choice([-1.0, 1.0])
        wind_state[0] = np.clip(a=wind_state[0], a_min=0, a_max=d-1)
        
    # independently, determine our j direction perturbation
    if np.random.uniform() < p_wind_j:
        wind_state[1] += np.random.choice([-1.0, 1.0])
        wind_state[1] = np.clip(a=wind_state[1], a_min=0, a_max=d-1)
        
    # just return our state
    return wind_state


def true_move(state, a, gw, gw_colors, p_wind_i, p_wind_j):
    """
    Parameters
    ----------
    state : (y,x,c) tuple - current true location

    a : an action encoded as a tuple of 0-1's 
        e.g., (0,1) is up, (1,1) is diagonally up (up then right)
    
    gw : grid world encoding rewards
    
    gw_colors : grid world encoding colors
    
    phi_wind : probability of wind

    Returns
    -------
    new_state : a (y,x,c) tuple encoding new state that results
    from action

    """
    
    # if we're at the terminal state, directly teleport to the origin regardless of action or wind
    if (state[0] == 6) and (state[1] == 7):
        return (7, 0, 0.0)
       
    d  = gw.shape[0]
    
    # extract the state quantities
    i, j, c = tuple(state)
    
    # what's our proposed movement?
    i_new = int(np.clip(a=i-a[1], a_min=0, a_max=d-1))
    j_new = int(np.clip(a=j+a[0], a_min=0, a_max=d-1))
    
        
    # compile the new state (not color yet)
    new_state = np.array([i_new, j_new, np.nan])
    
    # possibly add wind here
    new_state = wind(new_state, d, p_wind_i, p_wind_j)
    
    # get color of final new state
    new_state[2] = int(gw_colors[int(new_state[0]),
                                 int(new_state[1])]) #TODO: did I flip these?
   
    # convert to a tuple
    new_state = tuple(new_state)
    
    return new_state


    
######################################
# Taking actions based on Q.
######################################

def select_action(state, Q, epsilon):
    """
    Function select actions based on an epsilon greedy policy 
    (or greedy if psilon = 0), maxixmizing over the Q function formatted
    as output by init_Q

    Parameters
    ----------
    state : tuple encoding state to select action for

    Q : Q matrix

    epsilon : int or float >= 0

    Returns
    -------
    an action encoded as a length 2 tuple
    """
    assert epsilon >= 0

    # what is the "greedy" action index?
    greedy_idx = np.argmax([Q[(state, a)] for a in ACTIONS])

    # let's actually pick our action index based on epsilon greedy
    action_idx = greedy_idx if np.random.uniform() > epsilon else np.random.choice(len(ACTIONS))

    # return our action
    return ACTIONS[action_idx]


# function for updating Q
def update_Q(Q, state, action, reward, new_state, alpha, gamma):
    """
    Given Q function, state, action, reward and next state, do a 
    standard Q update
    """
    
    # what's our list of ACTIONS again?
    actions = [(0, 0), (0, 1), (1, 1), 
               (1, 0), (1, -1), (0, -1), 
               (-1, -1), (-1, 0), (-1, 1)]
    
    # make a copy of Q
    Qnew = copy.deepcopy(Q)
    
    # figure out optimal Q-value on S'
    optQ = np.max([Q[new_state, a] for a in actions])
    
    # update our q-entry
    Qnew[state, action] += alpha * (reward + (gamma*optQ) - Q[state, action])
    
    # return our Q
    return Qnew
    
    
######################################
# Missing Data Mechanism
######################################  

def MCAR(state, thetas):
    """
    This is the building block missing data function which, 
    given the current state and a vector of thetas of the same length,
    independently and randomly sets each element i of the vector to missing
    with Bernoulli(theta_i). That is, thetas represent the probability
    of being missing so 
    
        * \theta_i = 0   for always observed
        * \theta_i = 1   for never observed

    Parameters
    ----------
    state : np array
    thetas : np array of same length as state with elements in [0,1]

    Returns
    -------
    po_state : a copy of state, possibly with some elements set to np.nan
    """
    assert len(state) == len(thetas), "theta-state length mismatch"
    
    # make a copy to stay safe: "partially-observed state"
    po_state = np.array(state).copy().astype(float)
    
    # generate bernoullis and mask relevant elements
    mask = np.random.binomial(1, thetas).astype(bool)
    po_state[mask] = np.nan
    
    # return our missing-operated state
    return tuple(po_state)


def Mcolor(state, theta_dict): 
    """
    Applied MCAR function  (see above) with a different theta
    for each color
    
    If the third element of each theta entry is 0 so that
    color can never be missing, then this is MAR. Else, this is NMAR
    
    Examplantion: x,y|c are missing at random indepenent of the true
    value of (x,y) but c|x,y will depend on the missing value of c
    if color is missing. E.g., if red is has high missing rates
    and green has low missing rates, then we may assess the probability
    of red|x,y to be lower than it actually is just because of higher missing
    rate.
    
    Note: this becomes especially relevant when we have stochastic
    water so that color is random...some areas of our grid example have
    constant color, which shouldn't be hard to learn ( maybe depending
    a bit on initialization )
    
    
    Parameters
    ----------
    state : np.array
    
    theta_dict : dictionary with keys "0","1","2" and values
        each an np.array of same length as state containing
        elements in [0,1]

    Returns
    -------
    po_state : a copy of state, possibly with some elements set to np.nan
  
    """
    
    # query what our true color is + get the corresponding thetas_c vector
    c = int(state[2]); thetas_c = theta_dict[c]
    
    # use MCAR
    return MCAR(state, thetas_c)




# If you are in that region, 
# Mfog is MAR if theta_{1j} = theta_{2j} = 0 (i.e, x,y are observed for all timesteps), else NMAR
def Mfog(state, i_range, j_range, thetas_in, thetas_out):
    """
    This missing data mechanism 'casts a fog' over some
    rectangular region of the grid so that within this region, 
    missingness has one rate and outside this region, another.
    
    This is MCAR if (x,y) are always observed (first two elements of
         each thet aare 0) and only color is missing
    Else it is NMAR
    
    
    Parameters
    ----------
    state : np.array

    i_range : a tuple (a,b) for lower and upper bounds of fog in y direction
   
    j_range : a tuple (c,d) for lower and upper bounds of fog in x direction
        DESCRIPTION.
     
    thetas_in : np array of same length as state with elements in [0,1]
  
    thetas_out : np array of same length as state with elements in [0,1]

    Returns
    -------
    po_state : a copy of state, possibly with some elements set to np.nan
 
    """
    assert len(state) == len(thetas_in), "thetas_in length mismatch"
    assert len(state) == len(thetas_out), "thetas_out length mismatch"
    
    # default to not being in the region
    inregion = False
    
    # check if we're in the fog region or not
    if np.clip(a=state[0], a_min=i_range[0], a_max=i_range[1]) == state[0]:
        if np.clip(a=state[1], a_min=j_range[0], a_max=j_range[1]) == state[1]:
            inregion = True
            
    # figure out what thetas to use + apply the MCAR
    thetas = thetas_in if inregion else thetas_out
    return MCAR(state, thetas)
    





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap

import sys, copy, os, shutil
from tqdm.notebook import tqdm
import seaborn as sns


# Global Parameters 
ACTION_DESCS = {(0, 0) : "stay", 
                (0, 1) : "up", 
                (1, 1) : "diag-right-up", 
                (1, 0) : "right", 
                (1, -1): "diag-right-down", 
                (0, -1) : "down", 
                (-1, -1) : "diag-left-down",
                (-1, 0) : "left", 
                (-1, 1): "diag-left-up"}

# what are the actual list of actions that are possible?
#ACTIONS = list(ACTION_DESCS.keys())

# just to make sure we can get it quickly
def load_actions(allow_stay_action):
    
    if allow_stay_action:
        return ACTION_DESCS.copy()
    else:
        return {k:v for k,v in ACTION_DESCS.items() if k != (0,0)}
    

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
    
    # possibly add wind to the new state
    new_state = wind(new_state, d, p_wind_i, p_wind_j)
    
    # clip so that wind does not push outside of 3 x 3 grid from original state 
    i_new = int(np.clip(new_state[0], state[0]-1, state[0]+1))
    j_new = int(np.clip(new_state[1], state[1]-1, state[1]+1))
    new_state = np.array([i_new, j_new, np.nan])
    
    # get color of final new state
    new_state[2] = int(gw_colors[int(new_state[0]),
                                 int(new_state[1])])
   
    # convert to a tuple
    new_state = tuple(new_state)
    
    return new_state
    
    
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
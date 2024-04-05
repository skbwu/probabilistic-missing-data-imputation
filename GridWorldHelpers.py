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




####################
# Environment Set-up
####################



def build_grids():
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

    """
    d=10
    
    # baseline grid
    gw0 = np.full(shape=(d, d), fill_value=-1.0); gw0[0, -1] = +10.0

    # bridge grid world (no pond overflow)
    gw1 = np.full(shape=(d, d), fill_value=-1.0); gw1[:, 5:8] = -10.0
    gw1[6,:] = -1.0; gw1[6,4:9] = -1 #the bridge
    gw1[0:3,:] = -1.0; gw1[6, -1] = +10.0

    # bridge grid world (yes pond overflow)
    gw2 = np.full(shape=(d, d), fill_value=-1.0); gw2[:, 4:9] = -10.0
    gw2[6,:] = -1.0; gw2[6,4:9] = -1 #the bridge
    gw2[0:3,:] = -1.0; gw2[6, -1] = +10.0

    return(gw0, gw1, gw2)

  

def visualize_reward_grid(gw, ax):
    """
    Use heatmap to visualize (mean) rewards of the given grid on the 
    given axes
    """
    sns.heatmap(gw, ax = ax, cmap="viridis", annot=True, cbar=False)



def get_environment(ce, p):
    """
    Function which takes current environment and
    
    * with probability p, from ce to 1-ce
    * with probability (1-p), keep environment constant
    
    Meant for toggling between overflow state and non-overflow state

    Parameters
    ----------
    p : float in [0,1]

    Returns
    -------
    Mean reward grid
    """
    u = np.random.uniform()
    if u < p: 
        return 1-ce
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
    
    return new_state




def sync_color(state, gw_colors):
    """
    Makes sure the color of state (y,x,c) aligns with the
    current state of the environment, as captured in gw_colors

    """
    state[2] = int(gw_colors[int(state[0]),int(state[1])]) #TODO: did I flip these?
    return(state)
     
    
    
    
    
    
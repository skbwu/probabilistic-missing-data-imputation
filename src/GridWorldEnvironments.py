# created 7/16/2024 to move functions generating environment to new file.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap

import sys, copy, os, shutil
from tqdm.notebook import tqdm
import seaborn as sns


#TODO: class specifictation
#TODO: could make grid itself a class
class LakeWorld():
    """TODO: class specification """

    def __init__(self, d, colors = [0,1,2],
                 baseline_penalty = -1, 
                 water_penalty = -10,
                 end_reward = 100,
                 river_restart = False, 
                 fog_i_range = (0,2),
                 fog_j_range = (5,7),
                 p_wind_i = 0,
                 p_wind_j = 0,
                 p_switch = 0,
                 start_location = (7, 0),
                 terminal_location = (6, 7)):
        
        self.d = d
        self.colors = colors
        self.p_switch = p_switch
        self.start_location = start_location
        self.terminal_location = terminal_location
        self.baseline_penalty = baseline_penalty
        self.water_penalty = water_penalty
        self.end_reward = end_reward
        self.river_restart = river_restart 
        
        #set-up grids and their colors in a dictionary
        basegrid = build_grid(d = d, baseline_penalty = baseline_penalty,
                            water_penalty = water_penalty, end_reward = end_reward,
                            flood = False)
        floodgrid = build_grid(d = d, baseline_penalty = baseline_penalty,
                            water_penalty = water_penalty, end_reward = end_reward,
                            flood = True)
        basegridcol = make_gw_colors(basegrid)
        floodgridcol = make_gw_colors(floodgrid)
        self.environments = {0 : [basegrid, basegridcol],
                             1 : [floodgrid, floodgridcol]}
     
        self.environment_options = [0,1]#0 for basegrid, 1 for floodgrid
        self.current_environment = 1
        
        # states
        self.state_value_lists = [list(range(d)), list(range(d)), colors]
        
        # default to initializing at start
        i,j = start_location
        self.current_state = (i,j, self.environments[self.current_environment][1][i,j])
        
        # wind settings
        self.p_wind_i = p_wind_i
        self.p_wind_j = p_wind_j
        
        # missingness settings
        self.fog_i_range = fog_i_range
        self.fog_j_range = fog_j_range
        
    
    def set_state(self,state): 
        """Allow user to manually set the state."""
        self.current_state = state

    def get_reward(self):
        """ Get reward from current state """
        reward = self.environments[self.current_environment][0][self.current_state[0],
                                                                self.current_state[1]]
        return(reward)        

    def refresh_environment(self):
        """With probability self.p_switch, flip current environment to the other
        option. Otherwise, keep it the same. This creates a markov chain.
        """ 
        u = np.random.uniform()
        if u < self.p_switch: 
            self.current_environment = np.abs(1-self.current_environment) #if 0, 1. If 1, 0       

    def step(self, action): 
        """Implement an action by updating current state. 
        Also implements any stochasticity in the environment before 
        new state is returned""" 
        
        #implement stochasticity
        self.refresh_environment()

        #if current state is terminal state, telaport to origin before take action
        if self.current_state == self.terminal_location:
            self.current_state = self.start_location
        
        # extract the state quantities
        i, j, c = tuple(self.current_state)
        
        # proposed movement
        i_new = int(np.clip(a=i-action[1], a_min=0, a_max=self.d-1))
        j_new = int(np.clip(a=j+action[0], a_min=0, a_max=self.d-1))
     
        # compile the new state (not color yet)
        new_loc = np.array([i_new, j_new])
        
        # possibly add wind to the new state
        new_loc = wind(new_loc, self.d, self.p_wind_i, self.p_wind_j)
        
        # clip so that wind does not push outside of 3 x 3 grid from original state 
        i_new = int(np.clip(new_loc[0], self.current_state[0]-1, self.current_state[0]+1))
        j_new = int(np.clip(new_loc[1], self.current_state[1]-1, self.current_state[1]+1))
        
        # get color of final new state
        new_col = self.environments[self.current_environment][1][i_new, j_new]
       
        # Update state
        self.current_state = (i_new, j_new, new_col)
        
        # Get reward
        reward = self.get_reward()
        
        # Check if should be sent back to start
        if self.river_restart:
            if (reward == self.water_penalty):
                self.current_state = self.start_location 
        
        return(reward, self.current_state)



####################
# Environment Set-up
####################

def build_grid(d, baseline_penalty = -1, 
                water_penalty = -10,
                end_reward = 100,
                flood = False):
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
        
    flood : whether lake in grid should be flooding

    Returns
    -------
    A tuple of mean reward grids


    #TODO 
    CAUTION:  this is not yet fully genearlized to any d. It works as expected for
    d=8  but maybe not for other values
    
    """
    # baseline grid
    #gw0 = np.full(shape=(d, d), fill_value=-1.0); gw0[0, -1] = end_reward
    bridge_height = d-2

    # bridge grid world (no pond flood)
    if not flood:
        gw = np.full(shape=(d, d), fill_value= baseline_penalty)
        gw[:, (d-5):(d-3)] = water_penalty  #the water - width
        gw[bridge_height,(d-5):(d-3)] = baseline_penalty #the bridge
        gw[0:3,:] = -1.0;  #clear water on top
        gw[bridge_height, -1] = +end_reward #final spot
        
    if flood:
        # bridge grid world (yes pond flood)
        gw = np.full(shape=(d, d), fill_value= baseline_penalty)
        gw[:, (d-6):(d-2)] = water_penalty #the water - width - one wider
        gw[bridge_height,(d-6):(d-2)] = baseline_penalty #the bridge
        gw[0:2,:] = -1.  #clear water on top (make it one wider)
        gw[bridge_height, -1] = +end_reward
    
    return(gw)


def get_state_value_lists(d, colors):  #TODO: deprecate
    """
    Given dimension of grid and a list of colors,
    create state_value_lists, a list of lists where each sublist
    gives the possible values of that dimension of the state
    
    In particular, order is 1:d, 1:d and then the list of colors
    """
    state_value_lists = [list(range(d)),
                      list(range(d)),
                      colors]
    
    return(state_value_lists)

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
    
    Meant for toggling between flood state and non-flood state

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
    
    #return 1 if np.random.uniform() < p_flood else 2
    

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


####################
# Action Set-up
####################

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

# just to make sure we can get it quickly
def load_actions(allow_stay_action):
    if allow_stay_action:
        return ACTION_DESCS.copy()
    else:
        return {k:v for k,v in ACTION_DESCS.items() if k != (0,0)}


##############################################
# Functions which implement consequence of actions in the True State Space
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

    Note: state can be just a location (length 2 tuple) and in that case,
    function also returns a location (length 2 tuple)

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
    
    
    
#################################################################
# Environment-Specific Missing Data Mechanism Functions
#################################################################

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








###################################
# Plotter
###################################


def plot_grid(rewards_grid, color_grid, show_fog = False):
    """
    Visualize the grid rewards and color together
    
    Warning: This function is pretty specific to our particular grid set-up with how it draws rectangle
    """
    d = rewards_grid.shape[0]

    # Pick colorblind friendly colors
    custom_colors = [(0, 1, 0, 0.3), (1, 0.5, 0, 0.4), (1, 0, 0, 0.45), (1,1,0,.5) ] 
    green = sns.color_palette("colorblind", 8)[2]
    orange = sns.color_palette("colorblind", 8)[1]
    red = sns.color_palette("colorblind", 8)[4]
    target = sns.color_palette("colorblind", 8)[2] #make it darker green below
    
    # Make colors lighter by setting alpha
    green = (green, .5)
    orange = (orange, .6)
    red = (red,.9)
    target = (target, 1)
    white = (1,1,1)
    
    # Create color map
    if show_fog:
        custom_colors = [green, orange, red, target, white]
    else:
        custom_colors = [green, orange, red, target]

    cmap = ListedColormap(custom_colors)  

    # Assign the terminal state its own color
    color_grid = color_grid.copy()
    color_grid[6,7] = 3
    color_grid[7,0] = 3
    
    if show_fog:
        color_grid[0:(2+1),5:(7+1)] = 4
   
    # Set-up grid
    fig, ax = plt.subplots(figsize=(d, d))
    ax.set_yticks([])
    ax.set_xticks([])
    ax.imshow(color_grid, cmap=cmap, interpolation='nearest')

    # Add number labels to each square
    for i in range(rewards_grid.shape[0]):
        for j in range(rewards_grid.shape[1]):
            plt.text(j, i, str(rewards_grid[i, j]), 
                     ha='center', va='center', color='black', 
                     fontweight = 'bold', fontsize = 24)

    # Set tick positions
    plt.gca().set_xticks(np.arange(-.5, 8, 1), minor=True) 
    plt.gca().set_yticks(np.arange(-.5, 8, 1), minor=True) 
    
    # get rid of extra tick marks
    plt.gca().tick_params(which='minor', size=0)

    # Plot a grid of lines at the ticks
    plt.grid(which='minor', color='black', linestyle='-', linewidth=1.5)
    

    # Add thicker border around the water
    l = 8
    x,y = np.where(rewards_grid == -10)
    width = np.sum(rewards_grid[y[0],:] == -10)
    height = np.sum(rewards_grid[:,x[0]] == -10) - 1
    rect = Rectangle((y[0] - 0.5, x[0] - 0.5), width, height, linewidth=l, edgecolor='blue', facecolor='none')
    ax.add_patch(rect)

    height = 1
    rect = Rectangle((y[0] - 0.5, x[-1] - 0.5), width, height, linewidth=l, edgecolor='blue', facecolor='none')
    ax.add_patch(rect)

    # add border around end state 
    rect = Rectangle((7 - 0.5, 6- 0.5), 1,1, linewidth=l, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    
    # add border around end state 
    rect = Rectangle((0 - 0.5, 7- 0.5), 1,1, linewidth=l, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
import MissingMechanisms as mm


class LakeWorld():
    """TODO: class specification """

    def __init__(self, d, colors = [0,1,2],
                 baseline_penalty = -1,  #grid configuration
                 water_penalty = -10,
                 end_reward = 100,
                 start_location = (7, 0),
                 terminal_location = (6, 7),
                 river_restart = False, 
                 p_wind_i = 0,  #environment stochasticity
                 p_wind_j = 0,
                 p_switch = 0,
                 MCAR_theta = np.array([0,0,0]), # MCAR parameter
                 fog_i_range = (0,2), #MFOG parameters
                 fog_j_range = (5,7),
                 theta_in = np.array([0,0,0]),  
                 theta_out = np.array([0,0,0]),
                 color_theta_dict = {0: np.array([0,0,0]), #MCOLOR parameters
                               1: np.array([0,0,0]),
                               2: np.array([0,0,0])},
                 action_dict = "default",
                 allow_stay_action = True
                 ):
        
        # Save basic grid parameters
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
        
        # set-up possible state list
        self.state_value_lists = [list(range(d)), list(range(d)), colors] #Note: order matters
        
        # set starting state
        i,j = start_location
        self.current_state = (i,j, self.environments[self.current_environment][1][i,j])
        
        # wind settings
        assert p_wind_i >= 0 and p_wind_i <= 1, "p_wind_i not a probability"
        assert p_wind_j >= 0 and p_wind_j <= 1, "p_wind_i not a probability"
        self.p_wind_i = p_wind_i
        self.p_wind_j = p_wind_j
        
        # missingness settings
        assert len(self.current_state) == len(theta_in), "theta_in length mismatch"
        assert len(self.current_state) == len(theta_out), "theta_out length mismatch"
        self.MCAR_theta = MCAR_theta
        self.fog_i_range = fog_i_range
        self.fog_j_range = fog_j_range
        self.theta_in = theta_in
        self.theta_out = theta_out
        self.color_theta_dict = color_theta_dict
       
        # actions: initialize default full range of actions with descriptions or use whatever user gives
        self.allow_stay_action = allow_stay_action
        if action_dict == "default":
            self.action_dict = {
                                (0, 1) : "up", 
                                (1, 1) : "diag-right-up", 
                                (1, 0) : "right", 
                                (1, -1): "diag-right-down", 
                                (0, -1) : "down", 
                                (-1, -1) : "diag-left-down",
                                (-1, 0) : "left", 
                                (-1, 1): "diag-left-up"}
            if allow_stay_action:
                self.action_dict[(0,0)] = "stay"
            
        else: 
            self.action_dict = action_dict
            
        # how to encode missing as state
        self.missing_as_state_value = -1
            
      
    def set_state(self,state): 
        """Allow user to manually set the state."""
        self.current_state = state
        
    def get_action_list(self):
        """Get possible actions in a list"""
        return list(self.action_dict.keys())

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
                
        # Terminal flag #TODO: test this
        if (self.current_state[0],self.current_state[1]) == self.terminal_location:
            terminal = True
        else: 
            terminal = False
        
        return(reward, self.current_state, terminal)
    
    
    def MCAR(self):
        """Wrapper which applies MCAR function to current state"""
        return(mm.MCAR(self.current_state, self.MCAR_theta))
    
    def Mfog(self):
        """
        This missing data mechanism 'casts a fog' over some rectangular region 
        of the grid so that within this region, missingness has one rate and 
        outside this region, another. This is MCAR if (x,y) are always observed 
        (first two elements of each theta are 0) and only color is missing.
        Else it is NMAR.
        
        Parameters
        ----------
        state : np.array

        i_range : a tuple (a,b) for lower and upper bounds of fog in y direction
       
        j_range : a tuple (c,d) for lower and upper bounds of fog in x direction
            DESCRIPTION.
         
        theta_in : np array of same length as state with elements in [0,1]
      
        theta_out : np array of same length as state with elements in [0,1]

        Returns
        -------
        po_state : a copy of state, possibly with some elements set to np.nan
     
        """
        # default to not being in the region
        inregion = False
        
        i_check = np.clip(a=self.current_state[0], 
                   a_min=self.fog_i_range[0], 
                   a_max=self.fog_i_range[1]) == self.current_state[0]
        j_check = np.clip(a=self.current_state[1], 
                   a_min=self.fog_j_range[0], 
                   a_max=self.fog_j_range[1]) == self.current_state[1]
        
        # check if we're in the fog region or not
        if i_check and j_check:
            inregion = True
            
        # figure out what theta to use + apply the MCAR
        theta = self.theta_in if inregion else self.theta_out
        return mm.MCAR(self.current_state, theta)
    
    
    def Mcolor(self): 
        """
        Applies MCAR function with a different theta vector for each color
        
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
        # query true color is + get the corresponding theta_c vector
        c_ix = int(self.current_state[2])
        theta_c = self.color_theta_dict[c_ix]
        # apply MCAR
        return mm.MCAR(self.current_state, theta_c)
    
    
    def get_filename(self, env_missing):
        """ Produce the part of the filename that concerns environment settings
        or missingness mechanism settings, which may be specific to environment"""
        
        # start our filename: p_switch = PS, PW = p_wind_{i,j}, MM = missingness mechanism
        fname = f"PS={self.p_switch}_PW={self.p_wind_i}_MM={env_missing}"
    
        # record whether stay in place action was allowed or not
        if self.allow_stay_action:
            fname += "_ASA=T"
        else:
            fname += "_ASA=F"
    
        # MISSING MECHANISM
        # record the MCAR variables
        if env_missing == "MCAR":
            # all theta_i are the same, just record what the theta was.
            fname += f"_MCAR_theta={self.MCAR_theta[0]}"
        # record the Mcolor variables
        elif env_missing == "Mcolor":
            # only thing that is differential/changing is whether the last value is 0.0 or something else.
            fname += f"_t-color={self.color_theta_dict[0][1]}+{self.color_theta_dict[0][2]}"
        # record the Mfog variables    
        elif env_missing == "Mfog":
            # record fog-in and fog-out theta values (equal for each component)
            fname += f"_t-in={self.theta_in[0]}_t-out={self.theta_out[0]}"
        # else, throw a hissy fit
        else:
            raise Exception("Missingness mechanism is not supported.")
            
    
        
            
    
    
    

##################################
# Logger 
##################################

class LakeWorldLogger():
    '''
    Creates and maintains a DataFrame to log results
    
    Per episode: log makes it possible to calculate
        1. Mean reward per episode, # of times we landed in the river per episode, # of steps per episode.
        2. Counts of fully-observed, 1-missing, 2-missing, and 3-missing states per episode.       
        3. Wall clock time per episode.
        
        
    '''
    def __init__(self, per_timestep = True):
        
        # for episode logs
        self.logs = pd.DataFrame(data=None, 
                            columns=["total_reward",
                                     "steps_river",
                                     "num_steps", 
                                     "counts_0miss", 
                                     "counts_1miss",
                                     "counts_2miss", 
                                     "counts_3miss",
                                     "wall_clock_time"])
        
        if per_timestep:
            self.t_step_logs = pd.DataFrame(data=None, columns=["t_step",
                                                                "action_i", 
                                                                "action_j", 
                                                                "true_i", 
                                                                "true_j", 
                                                                "true_c", 
                                                                "obs_i", 
                                                                "obs_j",
                                                                "obs_c", 
                                                                "reward", 
                                                                "wall_clock_time"])
    ##########################
    # EPISODE TRACKING
    ##########################
    def start_epsiode(self):
        """Resets all counters to 0"""
        self.total_reward = 0
        self.steps_river = 0
        self.num_steps = 0,
        self.counts_0miss = 0,
        self.counts_1miss = 0,
        self.counts_3miss = 0,
        self.start_time = time.time()
        
    def update_epsisode_log(self, env, new_pobs_state, reward):
        """
        Update the various trackers that have new info per step
        """
        
        self.num_steps += 1
        
        self.total_reward += reward
        
        if reward == env.water_penalty:
            self.steps_river += 1
            
        num_nan = np.isnan(new_pobs_state).sum() 
        if num_nan == 0:
            self.counts_0miss += 1
        elif num_nan == 1:
            self.counts_1miss += 1
        elif num_nan == 2:
            self.counts_2miss += 1
        elif num_nan == 3:
            self.counts_3miss += 1
            
                    
    def finish_and_reset_epsiode(self):
        """Add this episode to overall dataframe and reset things so that
        can start logging a new episode"""
        
        # get duration
        wall_clock_time = time.time() - self.start_time 
        
        # update our dataframe
        row = [self.total_reward, self.steps_river, self.num_steps, 
               self.counts_0miss, self.counts_1miss, self.counts_2miss, self.counts_3miss,
               wall_clock_time]
        
        # add to dataframe
        self.logs.loc[len(self.logs.index)] = row
        
        #start a new episode
        self.start_episode()
        
        
    ##########################
    # TIMESTEP TRACKING
    ##########################
    def start_t_step(self, t_step): 
        self.start_t_step = time.time()
        self.t_step_row = [t_step]
    
    def finish_t_step(self, env, action, new_true_state, new_pobs_state, reward):
        
        self.t_step_row += [action[0], action[1]]
        self.t_step_row += [elem for elem in new_true_state]
        self.t_step_row += [new_pobs_state[0], new_pobs_state[1], new_pobs_state[2], reward]
        self.t_step_row += [time.time() - self.start_t_step]
        self.t_step_logs.loc[len(self.t_step_logs.index)] = self.t_step_row
        
      
   
            
            
        
 
        
            



####################################
# Helpers used above
####################################

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
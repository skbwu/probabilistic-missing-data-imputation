# created 7/16/2024 to move functions generating environment to new file.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap

####################
# Environment Set-up
####################

def build_grids(d, baseline_penalty = -1, 
                water_penalty = -10,
                end_reward = 100):
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
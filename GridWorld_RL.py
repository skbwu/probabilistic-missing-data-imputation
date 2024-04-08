# -*- coding: utf-8 -*-
"""
This is the final destination of main process for running our 
learning algorithm
"""
import numpy as np

import GridWorldHelpers as gwh


############################################
# Set-up Environments
############################################

# Initialize reward grids
gw0, gw1, gw2 = gwh.build_grids(d=8, 
                               baseline_penalty = -1,
                               water_penalty = -10,
                               end_reward = 10)

# Generate their color grids
gw0_colors = gwh.make_gw_colors(gw0) 
gw1_colors = gwh.make_gw_colors(gw1)
gw2_colors = gwh.make_gw_colors(gw2)


# set-up the possible environments
environments = {
                0: [gw0, gw0_colors], #baseline
                1: [gw1, gw2_colors],
                2: [gw2, gw2_colors] #flooding
               }


#TODO
#-------------------

# for baseline with no water
ce = 0; p_switch = 0; indices = np.array([0, np.nan])

# for no stochastic water
ce = 1; p_switch = 0; indices = np.array([1, np.nan])

# for stochastic water
ce = 1; p_switch = None; indices = np.array([1, 2])

ce = gwh.get_environment(ce, 
                         p_switch = p_switch, 
                         indices = indices) #switch mechanism
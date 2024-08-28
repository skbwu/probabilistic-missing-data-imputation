# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 22:19:43 2024
@author: Kyla Chasalow
"""

import numpy as np



def MCAR(state : tuple, 
         thetas):
    """
    This is a building block missing data function which, 
    given a state and a vector of thetas of the same length,
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

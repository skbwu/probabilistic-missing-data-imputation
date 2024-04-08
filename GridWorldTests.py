# -*- coding: utf-8 -*-
"""
Script for tests of our functions
"""
import numpy as np

import GridWorldHelpers as gwh


def test_miss_mech():
    """
    Some very basic sanity check tests. Could do more to make sure
    actually doing things at right rate but have in any case done some
    repeated draws to make sure things looked right
    """

    # set-up
    theta_dict = {0: np.array([1,1,1]),
                       1: np.array([0.0,0.0,0.0]),
                       2: np.array([.4,.4,.4])}
    
    
    i_range = (3,5); j_range = (3,5)
    thetas_in = np.array([1,1,1])
    thetas_out = np.array([0.0,0.0,0.0]) 
    
    
    # test of basic MCAR
    state = np.array([1,1,1])
    assert (gwh.MCAR(state,np.array([0,0,0])) == state).all()
    assert np.isnan(gwh.MCAR(state,np.array([1,1,1]))).all()

    
    # test of color
    assert (gwh.Mcolor(state,theta_dict) == state).all()
    state[2] = 0
    assert np.isnan(gwh.Mcolor(state,theta_dict)).all()
    
    # test of fog - out region  
    out = gwh.Mfog(state, i_range = i_range, j_range = j_range, thetas_in = thetas_in, thetas_out = thetas_out)
    assert (out == state).all()
    
    # test of fog - in region
    out = gwh.Mfog(np.array([4,4,1]), i_range = i_range, j_range = j_range, thetas_in = thetas_in, thetas_out = thetas_out)
    assert np.isnan(out).all()
   
    
    print("All tests passed")
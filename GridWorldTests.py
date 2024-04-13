# -*- coding: utf-8 -*-
"""
Script for tests of our functions
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import GridWorldHelpers as gwh
import GridWorldImputers as gwi


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
    
    
    
    
def test_actions():
    """
    Try all possible actions and examine visusally if has done right thing

    """
    d = 5
    for a in list(gwh.action_descs.keys()):
            
        # try a test-case
        gw = np.zeros((d,d))
        gw_colors = gwh.make_gw_colors(gw)
    
        # initialize our state randomly
        state = np.array([2, 2, 1])
        gw[state[0], state[1]] = +100
        
        # get our new state
        new_state = gwh.true_move(state, a, gw, gw_colors, p_wind_i = 0, p_wind_j = 0)
        gw[int(new_state[0]), int(new_state[1])] = 50
        
        sns.heatmap(gw, cbar=False, cmap="viridis")
        plt.title(gwh.action_descs[a])
        plt.show()
    


def test_imputers():
    
    Tstandard = gwi.init_Tstandard(2,[4,5], 0.5)
    Tmice = gwi.init_Tmice(2,[4,5],0.5)

    S = (1,1,4)
    A = (0,1)
    
    #Make sure that if nothing missing, recovers original state
    pobs_state = (1,1,4)
    out = gwi.draw_mouse(Tmice, S, A, pobs_state = pobs_state, num_cycles = 3)
    assert out == pobs_state
    out = gwi.draw_Tstandard(Tstandard,S, A, pobs_state)
    assert out == pobs_state 

    #Probabilistic Tests that are very unlikely to fail though it is possible
    count = 0
    pobs_state = (1,np.nan,np.nan)
    for i in range(100):
        out = gwi.draw_Tstandard(Tstandard,S, A, pobs_state)
        assert out[0] == pobs_state[0]
        if out[1] != pobs_state[1]:
            count += 1
    assert count > 1
    
    count = 0
    pobs_state = (1,np.nan,np.nan)
    for i in range(100):
        out = gwi.draw_mouse(Tmice, S, A, pobs_state = pobs_state, num_cycles = 3)
        assert out[0] == pobs_state[0]
        if out[1] != pobs_state[1]:
            count += 1
    assert count > 1
    
    
    Tstandard[(S,A)][(1,0,4)] = 1000 #make this dominate
    count = 0
    pobs_state = (1,np.nan,np.nan)
    for i in range(100):
        out = gwi.draw_Tstandard(Tstandard,S, A, pobs_state)
        assert out[0] == pobs_state[0]
        if out == (1,0,4):
            count += 1
    assert count > 90
    
    
    Tmice[2][(S, A,(1,0))][5] = 1000 #make color 5 dominate over 4
    count = 0
    pobs_state = (1,0,np.nan)
    for i in range(100):
        out = gwi.draw_mouse(Tmice, S, A, pobs_state = pobs_state, num_cycles = 3)
        assert out[0] == pobs_state[0]
        assert out[1] == pobs_state[1]
        if out == (1,0,5):
            count += 1
    assert count > 90


    #check that elsehwere it's still 50-50 (5 does not dominate over 4)
    count = 0
    S = (0,0,4)
    pobs_state = (1,0,np.nan)
    for i in range(100):
        out = gwi.draw_mouse(Tmice, S, A, pobs_state = pobs_state, num_cycles = 3)
        assert out[0] == pobs_state[0]
        assert out[1] == pobs_state[1]
        if out == (1,0,5):
            count += 1
    assert count < 90, "still 50-50"
    
    count = 0
    pobs_state = (1,0,np.nan)
    for i in range(100):
        out = gwi.draw_Tstandard(Tstandard,S, A, pobs_state)
        assert out[0] == pobs_state[0]
        assert out[1] == pobs_state[1]
        if out == (1,0,4):
            count += 1
    assert count < 90
    
    
    #check it works in case where everything is missing
    count = 0
    S = (0,0,4)
    Tstandard[(S,A)][(1,0,5)] = 1 #make this dominate a little
    pobs_state = (np.nan,np.nan,np.nan)
    for i in range(100):
        out = gwi.draw_mouse(Tmice, S, A, pobs_state = pobs_state, num_cycles = 3)
        if out == (1,0,5):
            count += 1
    assert count < 95 and count > 5
    
    print("Imputation method tests passed")
    
    
    
if __name__ == "__main__":
    test_miss_mech()
    test_actions()
    test_imputers()
    
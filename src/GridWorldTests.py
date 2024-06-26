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
   
    
    print("Test miss mech passed")
    
    
    
    
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
    
 
def test_Tupdaters():
    
    #set-up
    Tstandard = gwi.init_Tstandard(2,[4,5], 0)
    Tmice = gwi.init_Tmice(2,[4,5],0)
    true_state = (0,0,4)
    pobs_state = (0, np.nan, 4)
    A = (0,1)
    
    #get vector of S' imputations based on S vector and A
    K = 10
    Slist = [true_state] * K
    new_Slist = gwi.MI(method = "joint",
           Slist = Slist,
           A = A,
           pobs_state = pobs_state,
           shuffle = False,
           Tstandard = Tstandard)

    assert Tstandard[((0,0,4),A)][(0,0,4)] == 0
    assert Tstandard[((0,0,4),A)][(0,0,4)] == 0
    assert Tstandard[((0,1,4),A)][(0,0,4)] == 0
    
    gwi.Tstandard_update(Tstandard, Slist, A, new_Slist)
    
    assert Tstandard[((0,1,4),A)][(0,0,4)] == 0
    #these should hold with very high probability
    assert Tstandard[((0,0,4),A)][(0,1,4)] > 0
    assert Tstandard[((0,0,4),A)][(0,0,4)] > 0
    
    print("Tupdater Tstandard updater passed")
    
    
    #conditional of color, which is 4, given (0, ?)    
    assert Tmice[2][((0,0,4),A, (1,0))][4] == 0
    assert Tmice[2][((0,0,4),A, (0,0))][4] == 0
    assert Tmice[2][((0,0,4),A, (0,1))][4] == 0

    #conditional of x coordinate, which is ?, given (0, 4)    
    assert Tmice[1][((0,0,4),A, (0,5))][1] == 0
    assert Tmice[1][((0,0,4),A, (0,4))][0] == 0
    assert Tmice[1][((0,0,4),A, (0,4))][1] == 0
    
    gwi.Tmice_update(Tmice, Slist, A, new_Slist)
    
    #conditional of color, which is 4, given (0, ?)    
    assert Tmice[2][((0,0,4),A, (1,0))][4] == 0
    #these should hold with very high probability
    assert Tmice[2][((0,0,4),A, (0,0))][4] > 0
    assert Tmice[2][((0,0,4),A, (0,1))][4] > 0
    
    #conditional of x coordinate, which is ?, given (0, 4)    
    assert Tmice[1][((0,0,4),A, (0,5))][1] == 0
    assert Tmice[1][((0,0,4),A, (0,4))][0] > 0
    assert Tmice[1][((0,0,4),A, (0,4))][1] > 0
    
   
    print("Tupdater Tmice updater passed")
    
 
def test_Qupdate():
    
    Q = gwh.init_Q(3)
    alpha = 1; gamma = 1
    assert Q[(0,0,0),(0,0)] == 0
    assert Q[(1,1,1),(0,0)] == 0
    
    Q = gwh.update_Q(Q, state = (0,0,0), action = (0,0),
                 reward = 10, new_state = (1,1,1), 
                 alpha = alpha, gamma = gamma)
    Q = gwh.update_Q(Q, state = (1,1,1), action = (0,0),
                 reward = 10, new_state = (0,0,0), 
                 alpha = alpha, gamma = gamma)
    
    assert Q[(0,0,0),(0,0)] == 10  #0 + 1[10 + 1*0 - 0]
    assert Q[(1,1,1),(0,0)] == 20   #0 + 1[10 + 1*10 - 0]
    
    alpha = .5; gamma = .5
    Q = gwh.update_Q(Q, state = (0,0,0), action = (0,0),
                 reward = 10, new_state = (1,1,1), 
                 alpha = alpha, gamma = gamma)
    assert Q[(0,0,0),(0,0)] == 15  #10 + .5[10 + .5*20 - 10]
    assert Q[(1,1,1),(0,0)] == 20   #unchanged
    print("Basic update Q test passed")


def test_miss_pipeline(impute_method):
    """
    A dummy run of pipeline
    """
    #set some parameters
    d = 2
    colors = [4,5]
    init_T_val = 0
    p_shuffle = .2
    K = 10
    num_cycles = 10
    alpha = .5; gamma = .25
    
    #init stuff
    Q = gwh.init_Q(d, colors = colors)
    Tstandard = gwi.init_Tstandard(d,colors, init_T_val)
    Tmice = gwi.init_Tmice(d,colors,init_T_val)
    
    #set dummy examples of states, rewards etc
    true_state = (0,0,4)
    A = (0,1)
    reward = 10
    Slist = [true_state] * K
    Slist[0] = (0,1,4) 
    pobs_state = (0, np.nan, np.nan)

    # draw whether to shuffle - won't matter here though
    shuffle = gwi.shuffle(p_shuffle)
    
    #get new state vector
    new_Slist = gwi.MI(method = impute_method,
       Slist = Slist,
       A = A,
       pobs_state = pobs_state,
       shuffle = shuffle,
       Tmice = Tmice,
       Tstandard = Tstandard,
       num_cycles = num_cycles)
    
    #Update T matrix 
    if impute_method == "mice":
        gwi.Tmice_update(Tmice, Slist, A, new_Slist)
    if impute_method == "joint":
        gwi.Tstandard_update(Tstandard, Slist, A, new_Slist)
        
        
    #Update Q matrix
    Q  = gwi.updateQ_MI(Q, Slist, new_Slist, A, reward, alpha, gamma)
    
    
    Slist = new_Slist 
    
    print(f"A dummy example of the MI pipeline ran without error for imp method {impute_method}")
          
      



    




 
    
if __name__ == "__main__":
    test_miss_mech()
    #test_actions() - produces visuals
    test_imputers()
    test_Tupdaters()
    test_Qupdate()
    test_miss_pipeline("joint")
    test_miss_pipeline("mice")
    
    
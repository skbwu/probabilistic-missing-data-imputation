# -*- coding: utf-8 -*-
"""
Script for tests of our functions
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

import ImputerTools as impt
import LakeWorldEnvironments as lwe
import RLTools as rlt
import MissingMechanisms as mm


def get_state_value_lists(d, colors):  #don't need this in main anymore but useful here
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


def test_lakeworld(print_action_plots = False, with_wind = False):
    """
    Some very basic sanity check tests. Could do more to make sure
    actually doing things at right rate but have in any case done some
    repeated draws to make sure things looked right
    """
    
    # set-up
    theta_dict = {0: np.array([1,1,1]),
                       1: np.array([0.0,0.0,0.0]),
                       2: np.array([.4,.4,.4])}
    
    
    env = lwe.LakeWorld(d=8,
                        color_theta_dict= theta_dict,
                        theta_in= np.array([1,1,1]),
                        theta_out = np.array([0,0,0]),
                        fog_i_range = (3,5),
                        fog_j_range = (3,5),
                        )

  
    env.set_state((1,1,1))
    assert env.current_state[0] == 1
    assert env.current_state[1] == 1
    
    # test of basic MCAR
    env.MCAR_theta = [0,0,0]
    assert all(elem == 1 for elem in env.MCAR())
    env.MCAR_theta = [1,1,1]
    assert all(np.isnan(env.MCAR()))
    print("Test MCAR passed")
    
    
    
    # test of color
    assert all(elem == 1 for elem in env.Mcolor())
    env.set_state((1,1,0))
    assert all(np.isnan(elem) for elem in env.Mcolor())
    print("Test MCOLOR passed")
    
    
    
    # test of fog - out region, in region 
    env.set_state((1,1,1))
    assert all(elem == 1 for elem in env.Mfog())
    env.set_state((4,4,1))
    assert all(np.isnan(elem) for elem in env.Mfog())
    print("Test MFOG passed")
    
    # try taking steps
    env.set_state((1,1,1))
    env.step((1,1))
    assert env.current_state[0] == 0
    assert env.current_state[1] == 2
    env.step((1,1))
    assert env.current_state[0] == 0
    assert env.current_state[1] == 3
    env.step((-1,1))
    
    env.set_state((1,1,1))
    env.step((-1,-1))
    assert env.current_state[0] == 2
    assert env.current_state[1] == 0
    
    # with wind  
    
    if print_action_plots:
        
        if with_wind:
            env.p_wind_i = .5; env.p_wind_j = .5
      
        state = (2,2,1)
        env.set_state(state)

        # dummy grid with highlight at current location
        gw = np.zeros((8,8))
        gw[state[0], state[1]] = +100      
        gw_colors = lwe.make_gw_colors(gw)
        env.environments[2] = [gw, gw_colors] 
        env.current_environment = 2
        
        for a in env.action_list:  
            env.step(a)
            gw[int(env.current_state[0]), int(env.current_state[1])] = 50 #mark on map where are
            sns.heatmap(gw, cbar = False, cmap= 'viridis')
            plt.title(env.action_dict[a])
            plt.show()
            #reset
            gw[int(env.current_state[0]), int(env.current_state[1])] = 25 #old
            env.set_state(state)
         
    
    
    
def test_actions():
    """
    Try all possible actions and examine visusally if has done right thing

    """
    d = 5
    for a in list(lwe.action_descs.keys()):
            
        # try a test-case
        gw = np.zeros((d,d))
        gw_colors = lwe.make_gw_colors(gw)
    
        # initialize our state randomly
        state = np.array([2, 2, 1])
        gw[state[0], state[1]] = +100
        
        # get our new state
        new_state = lwe.true_move(state, a, gw, gw_colors, p_wind_i = 0, p_wind_j = 0)
        gw[int(new_state[0]), int(new_state[1])] = 50
        
        sns.heatmap(gw, cbar=False, cmap="viridis")
        plt.title(lwe.action_descs[a])
        plt.show()
    


def test_imputers():
    
    action_list = [(0,0),(0,1)]
   
    Tstandard = impt.init_Tstandard(get_state_value_lists(2, [4,5]),
                                   action_list,  0.5)
    Tmice = impt.init_Tmice(get_state_value_lists(2, [4,5]),
                                   action_list)

    S = (1,1,4)
    A = (0,1)
    
    #Make sure that if nothing missing, recovers original state
    pobs_state = (1,1,4)
    out = impt.draw_mouse(Tmice, S, A, pobs_state = pobs_state, num_cycles = 3)
    assert out == pobs_state
    out = impt.draw_Tstandard(Tstandard,S, A, pobs_state)
    assert out == pobs_state 

    #Probabilistic Tests that are very unlikely to fail though it is possible
    count = 0
    pobs_state = (1,np.nan,np.nan)
    for i in range(100):
        out = impt.draw_Tstandard(Tstandard,S, A, pobs_state)
        assert out[0] == pobs_state[0]
        if out[1] != pobs_state[1]:
            count += 1
    assert count > 1
    
    count = 0
    pobs_state = (1,np.nan,np.nan)
    for i in range(100):
        out = impt.draw_mouse(Tmice, S, A, pobs_state = pobs_state, num_cycles = 3)
        assert out[0] == pobs_state[0]
        if out[1] != pobs_state[1]:
            count += 1
    assert count > 1
    
    
    Tstandard[(S,A)][(1,0,4)] = 1000 #make this dominate
    count = 0
    pobs_state = (1,np.nan,np.nan)
    for i in range(100):
        out = impt.draw_Tstandard(Tstandard,S, A, pobs_state)
        assert out[0] == pobs_state[0]
        if out == (1,0,4):
            count += 1
    assert count > 90
    
    
    Tmice[2][(S, A,(1,0))][5] = 1000 #make color 5 dominate over 4
    count = 0
    pobs_state = (1,0,np.nan)
    for i in range(100):
        out = impt.draw_mouse(Tmice, S, A, pobs_state = pobs_state, num_cycles = 3)
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
        out = impt.draw_mouse(Tmice, S, A, pobs_state = pobs_state, num_cycles = 3)
        assert out[0] == pobs_state[0]
        assert out[1] == pobs_state[1]
        if out == (1,0,5):
            count += 1
    assert count < 90, "still 50-50"
    
    count = 0
    pobs_state = (1,0,np.nan)
    for i in range(100):
        out = impt.draw_Tstandard(Tstandard,S, A, pobs_state)
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
        out = impt.draw_mouse(Tmice, S, A, pobs_state = pobs_state, num_cycles = 3)
        if out == (1,0,5):
            count += 1
    assert count < 95 and count > 5
    
    print("Imputation method tests passed")
    
 
def test_Tupdaters():
    
   
    #set-up
    action_list = [(0,0),(0,1)]
    Tstandard = impt.init_Tstandard(get_state_value_lists(2,[4,5]),
                                   action_list, 0)
    Tmice = impt.init_Tmice(get_state_value_lists(2,[4,5]),
                                   action_list, 0)
    true_state = (0,0,4)
    pobs_state = (0, np.nan, 4)
    last_A = (0,1)
    
    #get vector of S' imputations based on S vector and A
    K = 10
    last_state_list = [true_state] * K
    new_last_state_list = impt.MI(method = "joint",
           last_state_list = last_state_list,
           last_A = last_A,
           pobs_state = pobs_state,
           shuffle = False,
           Tstandard = Tstandard)

    assert Tstandard[((0,0,4),last_A)][(0,0,4)] == 0
    assert Tstandard[((0,0,4),last_A)][(0,0,4)] == 0
    assert Tstandard[((0,1,4),last_A)][(0,0,4)] == 0
    
    impt.Tstandard_update(Tstandard, last_state_list, last_A, new_last_state_list)
    
    assert Tstandard[((0,1,4),last_A)][(0,0,4)] == 0
    #these should hold with very high probability
    assert Tstandard[((0,0,4),last_A)][(0,1,4)] > 0
    assert Tstandard[((0,0,4),last_A)][(0,0,4)] > 0
    
    print("Tupdater: Tstandard updater passed")
    
    
    #conditional of color, which is 4, given (0, ?)    
    assert Tmice[2][((0,0,4),last_A, (1,0))][4] == 0
    assert Tmice[2][((0,0,4),last_A, (0,0))][4] == 0
    assert Tmice[2][((0,0,4),last_A, (0,1))][4] == 0

    #conditional of x coordinate, which is ?, given (0, 4)    
    assert Tmice[1][((0,0,4),last_A, (0,5))][1] == 0
    assert Tmice[1][((0,0,4),last_A, (0,4))][0] == 0
    assert Tmice[1][((0,0,4),last_A, (0,4))][1] == 0
    
    impt.Tmice_update(Tmice, last_state_list, last_A, new_last_state_list)
    
    #conditional of color, which is 4, given (0, ?)    
    assert Tmice[2][((0,0,4),last_A, (1,0))][4] == 0
    #these should hold with very high probability
    assert Tmice[2][((0,0,4),last_A, (0,0))][4] > 0
    assert Tmice[2][((0,0,4),last_A, (0,1))][4] > 0
    
    #conditional of x coordinate, which is ?, given (0, 4)    
    assert Tmice[1][((0,0,4),last_A, (0,5))][1] == 0
    assert Tmice[1][((0,0,4),last_A, (0,4))][0] > 0
    assert Tmice[1][((0,0,4),last_A, (0,4))][1] > 0
    
   
    print("Tupdater: Tmice updater passed")
    
 
def test_Qupdate():
    
    Q = impt.init_Q(get_state_value_lists(3, [0,1,2]),
                   [(0,0),(0,1)]
                   )
    alpha = 1; gamma = 1
    action_list = [(0,0),(0,1)]
    assert Q[(0,0,0),(0,0)] == 0
    assert Q[(1,1,1),(0,0)] == 0
    
    Q = impt.update_Q(Q, state = (0,0,0), action = (0,0),
                     action_list = action_list,
                 reward = 10, new_state = (1,1,1), 
                 alpha = alpha, gamma = gamma)
    Q = impt.update_Q(Q, state = (1,1,1), action = (0,0),
                     action_list = action_list,
                 reward = 10, new_state = (0,0,0), 
                 alpha = alpha, gamma = gamma)
    
    assert Q[(0,0,0),(0,0)] == 10  #0 + 1[10 + 1*0 - 0]
    assert Q[(1,1,1),(0,0)] == 20   #0 + 1[10 + 1*10 - 0]
    
    alpha = .5; gamma = .5
    Q = impt.update_Q(Q, state = (0,0,0), action = (0,0),
                     action_list = action_list,
                 reward = 10, new_state = (1,1,1), 
                 alpha = alpha, gamma = gamma)
    assert Q[(0,0,0),(0,0)] == 15  #10 + .5[10 + .5*20 - 10]
    assert Q[(1,1,1),(0,0)] == 20   #unchanged
    print("Basic update Q test passed")


def test_select_action():
    
    d = 2
    state_value_lists = [list(range(d)),
                      list(range(d)),
                      [3,4]]

    action_list = [(0,0), (1,1)]
    
    # (1,1) should win
    Q = impt.init_Q(state_value_lists, action_list, True)
    Q[((0,0,3),(1,1))] = 46
    Q[((0,0,4),(1,1))] = 6
    Q[((0,0,3),(0,0))] = 45
    Q[((0,0,4),(0,0))] = 5
    
    action = impt.select_action((0,0,3), action_list, Q, epsilon = 0)
    assert action == (1,1)
   
    # (0,0) should win
    for o in ["voting1", "voting2", "averaging"]:
        action = impt.select_action([(0,0,3),(0,0,4)], action_list, Q, epsilon = 0, option = o)
        assert action == (1,1)


    Q[((0,0,3),(1,1))] = 20
    Q[((0,0,4),(1,1))] = 20
    Q[((0,0,3),(0,0))] = 40
    Q[((0,0,4),(0,0))] = 50
    for o in ["voting1", "voting2", "averaging"]:
        action = impt.select_action([(0,0,3),(0,0,4)], action_list, Q, epsilon = 0, option = o)
        assert action == (0,0)
   
   
    # some methods have random tie breaking and some don't
    Q[((0,0,3),(1,1))] = 50
    Q[((0,0,4),(1,1))] = 10
    Q[((0,0,3),(0,0))] = 10
    Q[((0,0,4),(0,0))] = 20
    count = 0
    for i in range(200):
        action = impt.select_action([(0,0,3),(0,0,4)], action_list, Q, epsilon = 0, option = "voting1")
        if action == (1,1):
            count += 1
    assert count > 10  and count < 190, "test1"
    count = 0
    for i in range(100):
        action = impt.select_action([(0,0,3),(0,0,4)], action_list, Q, epsilon = 0, option = "voting2")
        if action == (1,1):
            count += 1
    assert count > 10 and count < 190, "test2"
    
    action = impt.select_action([(0,0,3),(0,0,4)], action_list, Q, epsilon = 0,
                                option = "averaging")
    assert action == (1,1)
    
    
    
    # all random tie breaking
    Q[((0,0,3),(1,1))] = 20
    Q[((0,0,4),(1,1))] = 20
    Q[((0,0,3),(0,0))] = 20
    Q[((0,0,4),(0,0))] = 20
    
    for o in ["voting1", "voting2", "averaging"]:
        count = 0
        for i in range(200):
            action = impt.select_action((0,0,3), action_list, Q, epsilon = 0, option = o)
            if action == (1,1):
                count += 1
        assert count > 10 and count < 190
            
    
    print("Basic select_action tests passed")
    
    rlt.get_action(last_imp_state = (0,0,3),
                   last_imp_state_list = [(0,0,1),(0,0,2)],
                  impute_method = "last_fobs",
                  action_list = action_list,
                  Q = Q, 
                  epsilon = .05,
                  action_option = "voting2")
        
    print("get_action function runs")
    
    
    
def test_get_imputation():
    
    
    new_pobs_state = (1,2,1,np.nan)
    last_fobs_state = (1,1,0,2)
    last_obs_state_comp = (1,5,0,4)
    last_A = (1,1)
    last_state_list = [(1,1,1,2),(1,1,2,2)]
    K = len(last_state_list)
    missing_as_state_value = -2
    
    #just for MI methods
    d = 3
    state_value_lists = [list(range(d)), list(range(d)),
                     list(range(d)), list(range(d))] 

    action_list = [(1,1),(0,0)]
    Tstandard = impt.init_Tstandard(state_value_lists,
                        action_list, 
                       init_value = 0.0)
    Tmice = impt.init_Tmice(state_value_lists,
                        action_list, 
                       init_value = 0.0)


    impute_method = "random_action"
    new_imp_state, new_imp_state_list = rlt.get_imputation(impute_method,
                   new_pobs_state, last_fobs_state, last_obs_state_comp,
                   last_A, last_state_list, K)
    
    assert new_imp_state == None and new_imp_state_list == None
    print(f"test of {impute_method} impute method passed")
    
    
    impute_method = "last_fobs"
    new_imp_state, new_imp_state_list = rlt.get_imputation(impute_method,
                   new_pobs_state, last_fobs_state, last_obs_state_comp,
                   last_A, last_state_list, K)
    assert new_imp_state == last_fobs_state and new_imp_state_list == None
    print(f"test of {impute_method} impute method passed")
    
    new_pobs_state_temp = (1,2,1,1)
    new_imp_state, new_imp_state_list = rlt.get_imputation(impute_method,
                   new_pobs_state_temp, last_fobs_state, last_obs_state_comp,
                   last_A, last_state_list, K)
    assert new_imp_state == new_pobs_state_temp and new_imp_state_list == None
    print(f"test of {impute_method} impute method passed - nothing missing case")
    
    
    impute_method = "last_obs"
    new_imp_state, new_imp_state_list = rlt.get_imputation(impute_method,
                   new_pobs_state, last_fobs_state, last_obs_state_comp,
                   last_A, last_state_list, K)
    assert new_imp_state == (1,2,1,4) and new_imp_state_list == None
    print(f"test of {impute_method} impute method passed")
    
    impute_method = "missing_state"
    new_imp_state, new_imp_state_list = rlt.get_imputation(impute_method,
                   new_pobs_state, last_fobs_state, last_obs_state_comp,
                   last_A, last_state_list, K,
                   missing_as_state_value = missing_as_state_value)
    assert new_imp_state == (1,2,1,missing_as_state_value) and new_imp_state_list == None
    print(f"test of {impute_method} impute method passed")
    
    impute_method = "joint"
    new_imp_state, new_imp_state_list = rlt.get_imputation(impute_method,
                   new_pobs_state, last_fobs_state, last_obs_state_comp,
                   last_A, last_state_list, K,
                   Tstandard = Tstandard)
    assert new_imp_state == None 
    assert len(new_imp_state_list) == K
    assert all(elem[0] == 1 and elem[1] == 2 and elem[2] == 1 for elem in new_imp_state_list)
    assert all(elem[3] != -1 for elem in new_imp_state_list)
    print(f"(minimal) test of {impute_method} impute method passed")

    impute_method = "joint-conservative"
    new_imp_state, new_imp_state_list = rlt.get_imputation(impute_method,
                   new_pobs_state, last_fobs_state, last_obs_state_comp,
                   last_A, last_state_list, K,
                   Tstandard = Tstandard)
    assert new_imp_state == None 
    assert len(new_imp_state_list) == K
    assert all(elem[0] == 1 and elem[1] == 2 and elem[2] == 1 for elem in new_imp_state_list)
    assert all(elem[3] != -1 for elem in new_imp_state_list)
    print(f"(minimal) test of {impute_method} impute method passed")

    impute_method = "mice"
    new_imp_state, new_imp_state_list = rlt.get_imputation(impute_method,
                    new_pobs_state, last_fobs_state, last_obs_state_comp,
                    last_A, last_state_list, K,
                    Tmice = Tmice, 
                    num_cycles = 2)
    assert new_imp_state == None 
    assert len(new_imp_state_list) == K
    assert all(elem[0] == 1 and elem[1] == 2 and elem[2] == 1 for elem in new_imp_state_list)
    assert all(elem[3] != -1 for elem in new_imp_state_list)
    print(f"(minimal) test of {impute_method} impute method passed")
    
    
    new_pobs_state_temp = (1,2,1,1)
    new_imp_state, new_imp_state_list = rlt.get_imputation(impute_method,
                   new_pobs_state_temp, last_fobs_state, last_obs_state_comp,
                   last_A, last_state_list, K,Tmice = Tmice, 
                   num_cycles = 2)
    assert new_imp_state == new_pobs_state_temp 
    assert new_imp_state_list[0] == new_pobs_state_temp
    assert new_imp_state_list[1] == new_pobs_state_temp
    print(f"test of {impute_method} impute method passed - nothing missing case")
    
        
    
    
    
    
    
    
    
    

def test_dummy_miss_pipeline(impute_method):
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
    action_list = [(0,0),(0,1)]
    
    #init stuff
    Q = impt.init_Q(get_state_value_lists(d, colors),
                   action_list
                   )
    Tstandard = impt.init_Tstandard(get_state_value_lists(d, colors),
                                   action_list,
                                   init_T_val)
    Tmice = impt.init_Tmice(get_state_value_lists(d, colors),
                                   action_list,
                                   init_T_val)
    
    #set dummy examples of states, rewards etc
    true_state = (0,0,4)
    last_A = (0,1)
    reward = 10
    last_state_list = [true_state] * K
    last_state_list[0] = (0,1,4) 
    pobs_state = (0, np.nan, np.nan)

    # draw whether to shuffle - won't matter here though
    shuffle = impt.shuffle(p_shuffle)
    
    #get new state vector
    new_last_state_list = impt.MI(method = impute_method,
       last_state_list = last_state_list,
       last_A = last_A,
       pobs_state = pobs_state,
       shuffle = shuffle,
       Tmice = Tmice,
       Tstandard = Tstandard,
       num_cycles = num_cycles)
    
    #Update T matrix 
    if impute_method == "mice":
        impt.Tmice_update(Tmice, last_state_list, last_A, new_last_state_list)
    if impute_method == "joint":
        impt.Tstandard_update(Tstandard, last_state_list, last_A, new_last_state_list)
        
        
    #Update Q matrix
    Q  = impt.updateQ_MI(Q, last_state_list, new_last_state_list, last_A, action_list, reward, alpha, gamma)
    
    
    last_state_list = new_last_state_list 
    
    print(f"A dummy example of the MI pipeline ran without error for imp method {impute_method}")
          

        
def test_main_runRL():
    
    
    env = lwe.LakeWorld(d = 8,
                        colors = [0,1,2],
                        baseline_penalty = -1, 
                        water_penalty = -10,
                        end_reward = 100,
                        start_location = (7, 0),
                        terminal_location = (6, 7),
                        p_wind_i = .5,
                        p_wind_j = .5,
                        p_switch = .1,
                        fog_i_range = (0,2),
                        fog_j_range = (5,7),
                        MCAR_theta = [0,0,0],
                        theta_in = [0,0,0],
                        theta_out = [0,0,0],
                        color_theta_dict = {0:[0,0,0],1:[0,0,0],2:[0,0,0]},
                        action_dict = "default",
                        allow_stay_action = True
                        )
    
    logger = lwe.LakeWorldLogger() 

    t = 100

    s = time.time()
    rlt.run_RL(env,
           logger,
           miss_mech = "MCAR", # environment-missingness governor "MCAR", "Mcolor", "Mfog"
           impute_method = "last_fobs", # "last_fobs", "random_action", "missing_state", "joint", "mice"
               action_option = None, # voting1, voting2, averaging
               K = None, #number of multiple imputation chains
               num_cycles = None, #number of cycles used in Mice
               epsilon = .1, # epsilon-greedy governor 
               alpha = 1, # learning rate 
               gamma = .8, # discount factor 
               max_iters = t, # how many iterations are we going for?
               seed = 1, # randomization seed
               verbose=True, # intermediate outputs or nah?
               missing_as_state_value = -1,
               save_Q = True,
               log_per_t_step = True)
    e = time.time()
    print(f"The RL pipeline ran for {t} iterations in {e-s} seconds with per-timestep logging and save Q")

    s = time.time()
    rlt.run_RL(env,
           logger,
           miss_mech = "MCAR", # environment-missingness governor "MCAR", "Mcolor", "Mfog"
           impute_method = "last_fobs", # "last_fobs", "random_action", "missing_state", "joint", "mice"
               action_option = None, # voting1, voting2, averaging
               K = None, #number of multiple imputation chains
               num_cycles = None, #number of cycles used in Mice
               epsilon = .1, # epsilon-greedy governor 
               alpha = 1, # learning rate 
               gamma = .8, # discount factor 
               max_iters = t, # how many iterations are we going for?
               seed = 1, # randomization seed
               verbose=True, # intermediate outputs or nah?
               missing_as_state_value = -1,
               save_Q = False,
               log_per_t_step = False)
    e = time.time()

    print(f"The RL pipeline ran for {t} iterations in {e-s} seconds without per-timestep logging or save Q")


    




 
    
if __name__ == "__main__":
    print("--")
    test_lakeworld(print_action_plots = False, with_wind = False)
    #test_actions() - produces visuals
    print("--")
    test_imputers()
    print("--")
    test_Tupdaters()
    print("--")
    test_Qupdate()
    print("--")
    test_select_action()
    print("--")
    test_dummy_miss_pipeline("joint")
    print("--")
    test_dummy_miss_pipeline("mice")
    print("--")
    test_get_imputation()
    print("--")
    test_main_runRL()
    
    
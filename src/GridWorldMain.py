""" 
This script contains the main functions for running the grid world environment and
the missing data methods on them
"""
import numpy as np
import pandas as pd
from IPython.display import clear_output
import pickle
import copy
import time
import os

import GridWorldEnvironments as gwe # added 7/16/2024
import ImputerTools as impt
import RLTools as rlt
import MissingMechanisms as mm


#TODO: document requirements for loggers and environments
#TODO: make this document general and move specifics of LakeWorld environment to a run script

# create a master function
def runner(p_switch, # float, flooding Markov chain parameter, {0.0, 0.1}
           p_wind_i, p_wind_j, # float, up-down/left-right wind frequency, {0.0, 0.1, 0.2}. INTENDED EQUAL!
           allow_stay_action, #4/16/2024 addition 
           env_missing, # environment-missingness governor "MCAR", "Mcolor", "Mfog"
           MCAR_theta, # np.array, MCAR, same theta_i values {0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5}
           theta_in, # np.array, Mfog, in: (0.5, 0.5, 0.5) + (0.25, 0.25, 0.25)
           theta_out, # np.array, Mfog, out: (0.0, 0.0, 0.0) + (0.1, 0.1, 0.1)
           color_theta_dict, # dict with keys {0, 1, 2} corresponding to a np.array each.
           impute_method, # "last_fobs", "random_action", "missing_state", "joint", "mice"
           action_option, # voting1, voting2, averaging
           K, #number of multiple imputation chains
           p_shuffle, #shuffle rate for chains
           num_cycles, #number of cycles used in Mice
           epsilon, # epsilon-greedy governor {0.0, 0.01, 0.05}
           alpha, # learning rate (0.1, 1.0)
           gamma, # discount factor (0.0, 0.25, 0.5, 0.75, 1.0)
           max_iters, # how many iterations are we going for?
           seed, # randomization seed
           verbose=False, # intermediate outputs or nah?
           river_restart=False,
           missing_as_state_value = -1,
           testmode = False): # option to force agent back to starting point if fall into river. 7/16/2024. 

    if testmode:
        assert K >= 1 or K is None
        assert num_cycles >= 1 or num_cycles is None
        assert epsilon >= 0
        #TODO - add other checks?

    # For convenience
    MImethods = ["joint", "mice", "joint-conservative"]
  
    # initialize environment
    env = gwe.LakeWorld(d = 8,
                        colors = [0,1,2],
                        baseline_penalty = -1, 
                        water_penalty = -10,
                        end_reward = 100,
                        start_location = (7, 0),
                        terminal_location = (6, 7),
                        p_wind_i = p_wind_i,
                        p_wind_j = p_wind_j,
                        p_switch = p_switch,
                        fog_i_range = (0,2),
                        fog_j_range = (5,7),
                        MCAR_theta = MCAR_theta,
                        theta_in = theta_in,
                        theta_out = theta_out,
                        color_theta_dict = color_theta_dict,
                        action_dict = "default",
                        allow_stay_action = allow_stay_action
                        )
    
    # Get other environment attributes
    action_list = env.get_action_list()
    state_value_lists = env.state_value_lists
    
    # set our seed for use in multiple trials
    np.random.seed(seed)
    
    # Set-up logger
    logger = gwe.LakeWorldLogger() #***
   
    # initialize Q matrices
    if impute_method == "missing_state":
        Q = impt.init_Q(state_value_lists,
                       action_list, 
                       include_missing_as_state=True,
                       missing_as_state_value = -1)
    else:
        Q = impt.init_Q(state_value_lists,
                       action_list, 
                       include_missing_as_state=False)

    # Initialize Transition matrices
    Tstandard = impt.init_Tstandard(state_value_lists = state_value_lists,
                                   action_list = action_list, init_value = 0.0)
    Tmice = impt.init_Tmice(state_value_lists = state_value_lists,
                            action_list = action_list, init_value = 0.0)

    # Assume fully-observed initial state and initialize first obs state and first imp state
    # List version is only used if doing MI method
    last_pobs_state, last_imp_state = env.current_state, env.current_state
    last_imp_state_list = [env.current_state] * int(K) 
    last_fobs_state = env.current_state

  
    logger.start_epsiode() #start episode 1

    ###########################################################################
    for t_step in range(max_iters):
        
        logger.start_t_step()
    
        # Choose action A from S using policy-derived from Q, possibly e-greedy
        action = rlt.get_action(last_imp_state = last_imp_state, 
                       last_imp_state_list = last_imp_state_list,
                       impute_method = impute_method,
                       action_list = action_list, 
                       Q = Q, 
                       epsilon = epsilon,
                       action_option = action_option)
             
        # Take action A, observe R, S'
        reward, new_true_state, terminal = env.step(action) #environment stochasticity handled internally
        assert new_true_state == env.current_state  #TODO: temp - just to check when run this
       
        # Apply missingness mechanism to generate new partially observed state
        if hasattr(env, env_missing):
            miss_method = getattr(env, env_missing)
            new_pobs_state = miss_method()
        else:
            raise Exception(f"env does not have a missing method called {env_missing}")
          
        # Impute for the new_pobs_state, if needed
        new_imp_state, new_imp_state_list = rlt.get_imputation(impute_method = impute_method,
                           new_pobs_state  = new_pobs_state, last_fobs_state = last_fobs_state, 
                           last_A = action, 
                           last_state_list = last_imp_state_list,
                           K = K, Tstandard = Tstandard, Tmice = Tmice, num_cycles = num_cycles,
                           missing_as_state_value = missing_as_state_value)
       
        # Q update (if permitted)
        if impute_method in MImethods:
            Q  = impt.updateQ_MI(Q, 
                                Slist = last_imp_state_list, 
                                new_Slist = new_imp_state_list, 
                                A = action, action_list = action_list,
                                reward = reward, alpha = alpha, gamma = gamma)
        elif impute_method != "random_action":
            Q = impt.update_Q(Q, last_imp_state, action, action_list,
                              reward, new_imp_state, alpha, gamma)
        elif impute_method == "random_action":
              if ~np.any(np.isnan(new_pobs_state)):
                  if ~np.any(np.isnan(last_pobs_state)):
                     Q = impt.update_Q(Q, last_imp_state, action, action_list, reward, new_imp_state, alpha, gamma)

        # T update (if needed)
        if impute_method in MImethods:
            if impute_method == "mice":
                impt.Tmice_update(Tmice, 
                                 Slist = last_imp_state_list, 
                                 A = action, 
                                 newSlist = new_imp_state_list)
            if impute_method == "joint":
                impt.Tstandard_update(Tstandard, 
                                     Slist = last_imp_state_list,
                                     A = action,
                                     new_Slist = new_imp_state_list)
            if impute_method == "joint-conservative":
                #only update if previous and current state are fully observed
                if ~np.any(np.isnan(new_pobs_state)):
                    if ~np.any(np.isnan(last_pobs_state)):
                        impt.Tstandard_update(Tstandard, 
                                            Slist = last_imp_state_list,
                                            A = action,
                                            new_Slist = new_imp_state_list)

        
        # check whether last_fobs_state can be updated
        if ~np.any(np.isnan(new_pobs_state)):
            last_fobs_state = copy.deepcopy(new_pobs_state)

        # now that we have updated Q and T, update 'lasts' for next round
        last_pobs_state = copy.deepcopy(new_pobs_state)
        last_imp_state = copy.deepcopy(new_imp_state)
        if impute_method in MImethods:
            last_imp_state_list = copy.deepcopy(new_imp_state_list)
        

        # LOGGING 
        logger.update_epsisode_log(env, new_pobs_state, reward)
        logger.finish_t_step(env, action, new_true_state, new_pobs_state, reward)
        
        # status update?
        if verbose == True:
            if (t_step+1) % 5 == 0 and len(logger.logs.index) >= 20:
                clear_output(wait=True)
                print(f"""Timestep: {t_step+1}, Past 20 Mean Epi. Sum Reward: {np.round(logger.logs.loc[-20:].total_reward.mean(), 3)}, Fin. Episodes: {len(logger.logs.index)}, Past 20 Mean Path Length: {np.round(logger.logs.loc[-20:].num_steps.mean(), 3)}""")
            elif (t_step+1) % 5 == 0:
                clear_output(wait=True)
                print(f"Timestep: {t_step+1}")
                print(f"Total Reward So Far This Episode: {logger.total_reward}")
                
        if terminal:
            logger.finish_and_reset_epsiode()            
        
    
    ###############################################################
    ##### SAVING OUR LOGS FILE TO A .CSV OUTPUT ###################
    ###############################################################
    
    # check if we have a results folder
    if "results" not in os.listdir():
        os.mkdir("results")

    # start filename with environment-specific aspects
    fname = env.get_filename(env_missing)

    # add in our imputation mechanism
    IM_desc = impute_method.replace("_", "-")
    fname += f"_IM={IM_desc}"

    if impute_method in MImethods:
        # encode NC: num_cycles, K: no. of imputations, PS: p_shuffle
        fname += f"_NC={num_cycles}_K={K}_p-shuf={p_shuffle}"
        
    # add in action option
    if action_option == "voting1":
        fname += "_v1"
    if action_option == "voting2":
        fname += "_v2"
    if action_option == "averaging":
        fname += "_avg"

    # add in number of maximum iterations
    fname += f"_max-iters={max_iters}"

    # add in final hyperparameters: epsilon, alpha, gamma
    fname += f"_eps={epsilon}_a={alpha}_g={gamma}"

    # make a directory for this foldername
    if fname not in os.listdir("results"):
        os.mkdir(f"results/{fname}")

    # save the EPISODES + STEPWISE log files to a .csv
    logger.logs.to_csv(f"results/{fname}/episodic_seed={seed}.csv", index=False)
    logger.t_step_logs.to_csv(f"results/{fname}/stepwise_seed={seed}.csv", index=False)

    # save the Q matrix to a .pickle
    with open(f"results/{fname}/Q_seed={seed}.pickle", "wb") as file:
        pickle.dump(Q, file)
        
    # just for kicks
    return 1
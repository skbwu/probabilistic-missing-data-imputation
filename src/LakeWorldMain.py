""" 
This script contains the main functions for running the grid world environment and
the missing data methods on them
"""
import numpy as np
import LakeWorldEnvironments as lwe 
import RLTools as rlt


def run_LakeWorld(p_switch, # float, flooding Markov chain parameter, {0.0, 0.1}
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
           num_cycles, #number of cycles used in Mice
           epsilon, # epsilon-greedy governor {0.0, 0.01, 0.05}
           alpha, # learning rate (0.1, 1.0)
           gamma, # discount factor (0.0, 0.25, 0.5, 0.75, 1.0)
           max_iters, # how many iterations are we going for?
           seed, # randomization seed
           verbose=False, # intermediate outputs or nah?
           river_restart=False, # option to force agent back to starting point if fall into river. 
           testmode = False): 

    
    # initialize environment
    env = lwe.LakeWorld(d = 8,
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
    
    # set our seed for use in multiple trials
    np.random.seed(seed)
    
    # Set-up logger
    logger = lwe.LakeWorldLogger() 
    
    rlt.run_RL(env, logger, env_missing, 
           impute_method, action_option, K, num_cycles, 
           epsilon = epsilon, alpha = alpha, gamma = gamma, 
           max_iters = max_iters, seed = seed, 
           verbose = verbose, missing_as_state_value = env.missing_as_state_value,
           testmode = testmode)
    
    # just for kicks
    return 1
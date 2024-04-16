#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sys, copy, os, shutil
import copy, time
from IPython.display import clear_output
import pickle

# our helper functions for the gridworlds
import GridWorldHelpers as gwh
import GridWorldImputers as gwi
import SimulationHelpers as shelpers

# fix our verbose and max_iters variable (we'll deal with seed at the end!)
verbose, max_iters = False, 50000


# In[2]:

# get all our settings
settings = shelpers.get_settings()  #4/16/2024 moved this to a helper so easier to test


# In[3]:


# create a master function
def runner(p_switch, # float, flooding Markov chain parameter, {0.0, 0.1}
           p_wind_i, p_wind_j, # float, up-down/left-right wind frequency, {0.0, 0.1, 0.2}. INTENDED EQUAL!
           allow_stay_action, #4/16/2024 addition 
           env_missing, # environment-missingness governor "MCAR", "Mcolor", "Mfog"
           thetas, # np.array, MCAR, same theta_i values {0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5}
           thetas_in, # np.array, Mfog, in: (0.5, 0.5, 0.5) + (0.25, 0.25, 0.25)
           thetas_out, # np.array, Mfog, out: (0.0, 0.0, 0.0) + (0.1, 0.1, 0.1)
           theta_dict, # dict with keys {0, 1, 2} corresponding to a np.array each.
           impute_method, # "last_fobs", "random_action", "missing_state", "joint", "mice"
           K, #number of multiple imputation chains
           p_shuffle, #shuffle rate for chains
           num_cycles, #number of cycles used in Mice
           epsilon, # epsilon-greedy governor {0.0, 0.01, 0.05}
           alpha, # learning rate (0.1, 1.0)
           gamma, # discount factor (0.0, 0.25, 0.5, 0.75, 1.0)
           max_iters, # how many iterations are we going for?
           seed, # randomization seed
           verbose=False): # intermediate outputs or nah?

    # For convenience
    MImethods = ["joint", "mice", "joint-conservative"]
  
    ###############################################################
    ##### CREATING ENVIRONMENT + SETTING THE SEED #################
    ###############################################################
    #assert K >= 1 or K is None
    #assert num_cycles >= 1 or num_cycles is None
    
    # initializing our environments + corresponding colors
    d = 8 # dimension of our gridworld
    colors = [0,1,2] #colors encoded with 0,1,2
    gw0, gw1, gw2 = gwh.build_grids(d=8, baseline_penalty = -1, 
                                    water_penalty = -10, 
                                    end_reward = 100)
    gw0_colors = gwh.make_gw_colors(gw0)
    gw1_colors = gwh.make_gw_colors(gw1)
    gw2_colors = gwh.make_gw_colors(gw2)

    # store quick-access indices for the environment
    environments = {
                    0: [gw0, gw0_colors], # baseline
                    1: [gw1, gw1_colors], # non-flooding CORRECTED 4/16/2024
                    2: [gw2, gw2_colors] # flooding
                   }
    
    # fog range - fixed.
    i_range, j_range = (0, 2), (5, 7)

    # what is our starting "current environment"
    ce = 1

    # which environments are we flipping through?
    indices = np.array([1, 2]) # the two to be flipping between, if any. If just one, make first element
    
    # set our seed for use in multiple trials
    np.random.seed(seed)
    
    ###############################################################
    ##### INITIALIZING START OF SIMULATIONS + DATA STRUCTURES #####
    ###############################################################
    
    # load the possible actions list, specifying whether stay in place allowed
    action_descs = gwh.load_actions(allow_stay_action = allow_stay_action)
    ACTIONS = list(action_descs.keys())
    
    # initialize our Q matrix: {((i, j, color), (a1, a2))}
    if impute_method == "missing_state":
        Q = gwh.init_Q(d, ACTIONS, include_missing_as_state=True, colors = colors)
    else:
        Q = gwh.init_Q(d, ACTIONS, include_missing_as_state=False, colors = colors)

    
    # initialize Transition matrices
    Tstandard = gwi.init_Tstandard(d = d, action_list = ACTIONS, colors = colors, init_value = 0)
    Tmice = gwi.init_Tmice(d = d, action_list = ACTIONS, colors = colors, init_value = 0)

    # initialize our starting environment + corresponding colors
    env, env_colors = environments[ce][0], environments[ce][1]

    # initialize our true initial state to be the bottom left corner.
    true_state = (d-1, 0, env_colors[d-1, 0])
    #  Assume fully-observed initial state and initialize first obs
    #  state and first imp state
    pobs_state, impu_state = true_state, true_state

    # if doing multiple imputation method, initilize state list
    if impute_method in MImethods:
        imp_state_list = [true_state] * int(K)

    # initialize variable for our last fully-obs-state
    last_fobs_state = copy.deepcopy(true_state)

    '''
    DataFrame to log our results for this simulation:
    1. Mean reward per episode, # of times we landed in the river per episode, # of steps per episode.
    2. Counts of fully-observed, 1-missing, 2-missing, and 3-missing states per episode.
    3. Wall clock time per episode.
    '''
    # ALL METRICS ARE PER EPISODE!
    logs = pd.DataFrame(data=None, columns=["total_reward", "steps_river", "path_length", 
                                            "counts_0miss", "counts_1miss", "counts_2miss", "counts_3miss",
                                            "wall_clock_time"])

    # things we want to store PER EPISODE
    total_reward, steps_river, path_length = 0, 0, 0
    counts_0miss, counts_1miss, counts_2miss, counts_3miss = 0, 0, 0, 0
    wall_clock_time = None

    # start our timer FOR THIS EPISODE
    start_time = time.time()

    ###############################################################
    ##### RUNNING SIMULATIONS FOR EACH TIMESTEP ###################
    ###############################################################

    # for each timestep ...
    for t_step in range(max_iters):


        #############################################################
        # Action selection based on last state(s) or random selection 
        #############################################################
        # "choose action A from S using policy-derived from Q (e.g., \epsilon-greedy)"

        # do we have any missing state values?
        if np.any(np.isnan(pobs_state).mean()):
            # deal with it accordingly to get imputed actions
            if impute_method == "last_fobs":
                action = gwh.select_action(last_fobs_state, ACTIONS, Q, epsilon)
            elif impute_method == "random_action":
                action = ACTIONS[np.random.choice(a=len(ACTIONS))]
            elif impute_method == "missing_state":
                # for this method only, we need to convert np.nan to -1
                pobs_state_temp = tuple([val if ~np.isnan(val) else -1 for val in pobs_state])
                action = gwh.select_action(pobs_state_temp, ACTIONS, Q, epsilon)
            elif impute_method in MImethods:
                
                # vote on action. note: not taking most-selected action because suspect not enough exploration
                action_options = [gwh.select_action(s, ACTIONS, Q, epsilon) for s in imp_state_list]
                action = action_options[np.random.choice(len(action_options))]                   
            else:
                raise Exception("impute_method choice is not currently supported.")
        
        # if no missingness, select an action by standard epsilon greedy 
        else:
            action = gwh.select_action(pobs_state, ACTIONS, Q, epsilon)


        ###############################################
        # Take action A, observe R, S'
        # Taking action affects underlying TRUE state, even if
        # we won't observe it!!!
        ###############################################

        # toggle our environment potentially!
        env, env_colors = environments[gwh.get_environment(ce, p_switch, indices)]

        # figure out what our new state is, which tells us our reward
        new_true_state = gwh.true_move(true_state, action, env, env_colors, p_wind_i, p_wind_j)
        reward = env[int(new_true_state[0]), int(new_true_state[1])]

        # update our reward counter + river counters
        total_reward += reward
        if reward == -10:
            steps_river += 1

        ###############################################
        # Apply missingness mechanism to generate our new partially observed state
        ###############################################

        # simulate our partially-observed mechanism.
        if env_missing == "MCAR":
            new_pobs_state = gwh.MCAR(new_true_state, thetas)
        elif env_missing == "Mcolor":
            new_pobs_state = gwh.Mcolor(new_true_state, theta_dict)
        elif env_missing == "Mfog":
            new_pobs_state = gwh.Mfog(new_true_state, i_range, j_range, thetas_in, thetas_out)
        else:
            raise Exception("The given env_missing mode is not supported.")

        print(action)
        print(new_pobs_state)
        print("---")
        time.sleep(3)
        
        ###############################################
        # IMPUTATION
        # make our imputation for the new_pobs_state, if not everything is observed.
        ###############################################

        if np.any(np.isnan(np.array(new_pobs_state)).mean()):

            if impute_method == "last_fobs":
                new_impu_state = copy.deepcopy(last_fobs_state)
            
            elif impute_method == "random_action":
                new_impu_state = None # we're not imputing any states!
            
            elif impute_method == "missing_state":

                # swapping np.nan to -1 to play nicer with dictionary indexing.
                new_impu_state = tuple([val if ~np.isnan(val) else -1 for val in new_pobs_state])

            elif impute_method in MImethods:

                #decide if we will shuffle (affects Q and T updates below)
                shuffle = gwi.shuffle(p_shuffle)

                #generate list of imputed values
                #note: because first state already observed, will only
                #get here when already have defined action variable 
                new_imp_state_list = gwi.MI(
                       method = impute_method,
                       Slist = imp_state_list,
                       A = action, #previous action?
                       pobs_state = new_pobs_state,
                       shuffle = shuffle,
                       Tmice = Tmice,
                       Tstandard = Tstandard,
                       num_cycles = num_cycles)

                new_impu_state = None #don't need this
            else:
                raise Exception("impute_method choice is not currently supported.")

        # if nothing is missing, just set new_impu_state equal to the new_pobs_state
        else:
            # just make a deepcopy!
            new_impu_state = copy.deepcopy(new_pobs_state)
            if impute_method in MImethods:
                new_imp_state_list = [new_pobs_state] * int(K)

        ######################################
        # Q update (if permitted)
        ######################################
        # multiple imputation way of updating Q with fractional allocation   
        if impute_method in MImethods:
            Q  = gwi.updateQ_MI(Q, 
                                Slist = imp_state_list, 
                                new_Slist = new_imp_state_list, 
                                A = action, action_list = ACTIONS,
                                reward = reward, alpha = alpha, gamma = gamma)
            
        # if we have random_action method, then we cannot update 
        elif impute_method != "random_action":
            Q = gwh.update_Q(Q, impu_state, action, ACTIONS, reward, new_impu_state, alpha, gamma)
    
        #if nothing is missing in last or current state, then we can
        #update Q under random_action
        elif ~np.any(np.isnan(new_pobs_state)):
            if ~np.any(np.isnan(pobs_state)):
                Q = gwh.update_Q(Q, pobs_state, action, ACTIONS, reward, new_pobs_state, alpha, gamma)

        ######################################
        # T update (if needed)
        ######################################
        if impute_method in MImethods:
            if impute_method == "mice":
                gwi.Tmice_update(Tmice, 
                                 Slist = imp_state_list, 
                                 A = action, 
                                 newSlist = new_imp_state_list)
            if impute_method == "joint":
                gwi.Tstandard_update(Tstandard, 
                                     Slist = imp_state_list,
                                     A = action,
                                     new_Slist = new_imp_state_list)
            if impute_method == "joint-conservative":
                if ~np.any(np.isnan(new_pobs_state)):
                    gwi.Tstandard_update(Tstandard, 
                                         Slist = imp_state_list,
                                         A = action,
                                         new_Slist = new_imp_state_list)

        # check whether our last_fobs_state can be updated
        if ~np.any(np.isnan(pobs_state).mean()):
            last_fobs_state = copy.deepcopy(pobs_state)

        # now that we have updated Q and T functions
        # update true_state, pobs_state, impu_state, imp_state_list
        # as 'current state' for for the next round
        true_state = copy.deepcopy(new_true_state)
        pobs_state = copy.deepcopy(new_pobs_state)
        impu_state = copy.deepcopy(new_impu_state)
        if impute_method in MImethods:
            imp_state_list = copy.deepcopy(new_imp_state_list)
        
        # update our missing data counters
        if np.isnan(pobs_state).sum() == 0:
            counts_0miss += 1
        elif np.isnan(pobs_state).sum() == 1:
            counts_1miss += 1
        elif np.isnan(pobs_state).sum() == 2:
            counts_2miss += 1
        elif np.isnan(pobs_state).sum() == 3:
            counts_3miss += 1

        # update our path-length counter
        path_length += 1

        # also see if we hit the terminal state
        if (true_state[0] == 6) and (true_state[1] == 7):

            # end our timer + record time elapsed FOR THIS EPISODE!
            end_time = time.time()
            wall_clock_time = end_time - start_time

            # update our dataframe
            row = [total_reward, steps_river, path_length, 
                   counts_0miss, counts_1miss, counts_2miss, counts_3miss,
                   wall_clock_time]
            logs.loc[len(logs.index)] = row

            # reset our counter variables per EPISODE
            total_reward, steps_river, path_length = 0, 0, 0
            counts_0miss, counts_1miss, counts_2miss, counts_3miss = 0, 0, 0, 0
            wall_clock_time = None

            # reset our timer, too
            start_time = time.time()

        # status update?
        if verbose == True:
            s = 20
            if (t_step+1) % 5 == 0 and len(logs.index) >= 20:
                clear_output(wait=True)
                print(f"Timestep: {t_step+1}, Past 20 Mean Epi. Sum Reward: {np.round(logs.loc[-20:].total_reward.mean(), 3)}, Fin. Episodes: {len(logs.index)}, Past 20 Mean Path Length: {np.round(logs.loc[-20:].path_length.mean(), 3)}")
            elif (t_step+1) % 5 == 0:
                clear_output(wait=True)
                print(f"Timestep: {t_step+1}")
                print(f"Reward This Episode: {total_reward}")
    
    ###############################################################
    ##### SAVING OUR LOGS FILE TO A .CSV OUTPUT ###################
    ###############################################################
    
    # check if we have a results folder
    if "results" not in os.listdir():
        os.mkdir("results")

    # start our filename: p_switch = PS, PW = p_wind_{i,j}, MM = missingness mechanism
    fname = f"PS={p_switch}_PW={p_wind_i}_MM={env_missing}"

    # record whether stay in place action was allowed or not
    if allow_stay_action:
        fname += "_ASA=T"
    else:
        fname += "_ASA=F"

    # record the MCAR variables
    if env_missing == "MCAR":

        # all theta_i are the same, just record what the theta was.
        fname += f"_theta={thetas[0]}"

    # record the Mcolor variables
    elif env_missing == "Mcolor":

        # only thing that is differential/changing is whether the last value is 0.0 or something else.
        fname += f"_t-color={theta_dict[0][2]}"

    # record the Mfog variables    
    elif env_missing == "Mfog":

        # record fog-in and fog-out theta values (equal for each component)
        fname += f"_t-in={thetas_in[0]}_t-out={thetas_out[0]}"

    # else, throw a hissy fit
    else:
        raise Exception("Missingness mechanism is not supported.")

    # add in our imputation mechanism
    IM_desc = impute_method.replace("_", "-")
    fname += f"_IM={IM_desc}"

    if impute_method in MImethods:
        # encode NC: num_cycles, K: no. of imputations, PS: p_shuffle
        fname += f"_NC={num_cycles}_K={K}_p-shuf={p_shuffle}"

    # add in number of maximum iterations
    fname += f"_max-iters={max_iters}"

    # add in final hyperparameters: epsilon, alpha, gamma
    fname += f"_eps={epsilon}_a={alpha}_g={gamma}"

    # make a directory for this foldername
    if fname not in os.listdir("results"):
        os.mkdir(f"results/{fname}")

    # save the log file to a .csv
    logs.to_csv(f"results/{fname}/seed={seed}.csv", index=False)

    # save the Q matrix to a .pickle
    with open(f"results/{fname}/seed={seed}.pickle", "wb") as file:
        pickle.dump(Q, file)
        
    # just for kicks
    return 1


# In[4]:


# which index are we starting at?
start_idx = int(sys.argv[1])

# which settings are we working on?
for i in range(start_idx*15, (start_idx*15)+15): # REVISED 4/16/2024 to account for the K=5 added settings.
    
    # unpack our settings
    gamma, alpha, epsilon, p_switch, p_wind_i, p_wind_j, allow_stay_action, env_missing, thetas, thetas_in, thetas_out, theta_dict, impute_method, p_shuffle, num_cycles, K = settings[i]
    
    # do our random seeding
    for seed in range(3):
        
        # just call our runner function
        output = runner(p_switch=p_switch, p_wind_i=p_wind_i, p_wind_j=p_wind_j,
                        allow_stay_action = allow_stay_action, env_missing=env_missing, 
                        thetas=thetas, thetas_in=thetas_in, thetas_out=thetas_out, theta_dict=theta_dict, 
                        impute_method=impute_method,
                        p_shuffle=p_shuffle, num_cycles=num_cycles, K=K, 
                        epsilon=epsilon, alpha=alpha, gamma=gamma, 
                        max_iters=max_iters, seed=seed, verbose=verbose)


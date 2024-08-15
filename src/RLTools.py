import numpy as np
from IPython.display import clear_output
import pickle
import copy
import os

import ImputerTools as impt
import RLTools as rlt




SingleImpMethods = ["random_action", #this ultimately an attribute
             "last_fobs",
             "last_obs",
             "missing_state"]

MImethods = ["joint", #TODO: better names for this? joint-synthetic?
             "mice",
             "joint-conservative"]


def get_imputation(impute_method : str,
                   new_pobs_state : tuple, 
                   last_fobs_state : tuple,
                   last_obs_state_comp : tuple,
                   last_A,
                   last_state_list : list, 
                   K : int,
                   Tstandard = None, 
                   Tmice = None, 
                   num_cycles = None,
                   missing_as_state_value = None):
    """
    Parameters
    ----------
    impute_method : str 
        specifies which imputation method to use. 
        
    new_pobs_state : tuple
        partially observed state to impute for. If fully observed, 
        then returned as is
    
    last_fobs_state : tuple
        the last fully observed state. This is used in some impute methods 
        
    last_obs_state_comp : tuple
        the ith element of this tuple contains the last observed instance of 
        that dimension of the state space, even if when that dimension was
        observed, others were not. If the last time step was fully observed,
        then this is the same as last_fobs_state, but otherwise, it may differ
        
        e.g. if the states are (1,2,1), (1,4,?),(2,?,3) then last_fobs_state is
        (1,2,1) while last_obs_state_comp is (1,4,3)
    
    last_A : the most recent action taken (the one before new_pobs_state was 
             observed)
        
    last_state_list : list
            if imputation method is a multiple imputation method, expect this.
            Contains list of last observed states with their imputations
            
    K : int
           if imputation method is multiple imputation method, this is number
           of imputations drawn at each step. Should match length of 
           last_state_list
           
    Tstandard : dictionary encoding transition matrix, optional
        Required for MI method "joint" and "joint-conservative"
        The default is None.
        
    Tmice : dictionary for encoding transitions, optional
        Required for MI method "mice"
    
    num_cycles : int, optional
        Required for MI method "mice" - specifices number of MICE algorithm cycles
        to do at each step. Warning: quickly becomes computationally expensive if this is high
            
    missing_as_state_value : optional
        If imputation method is 'missing_state', then all this function does is replace
        any np.nan's with this value.

    Returns
    -------
    If method is a non-multiple imputation method: an imputed state
    If method is a multiple imputation method: a list of K imputed states
    
    If there is no missingness in the new_pobs_state (the current state), then
    the imputed state returned is simply a copy of the current state

    """

    new_imp_state_list = None
    
    if np.any(np.isnan(new_pobs_state)):
        
        #Case where impute nothing
        if impute_method == "random_action":
            return None, None

        if impute_method == "last_fobs":
            new_imp_state = copy.deepcopy(last_fobs_state)
            
        elif impute_method == "last_obs":
            new_imp_state = tuple([i if ~np.isnan(i) else j for (i,j) in zip(new_pobs_state, last_obs_state_comp)])
               
        elif impute_method == "missing_state":

            # swapping np.nan to -1 to play nicer with dictionary indexing.
            new_imp_state = tuple([val if ~np.isnan(val) else missing_as_state_value for val in new_pobs_state])
      
        elif impute_method in MImethods:
            new_imp_state_list = impt.MI(
                   method = impute_method,
                   last_state_list = last_state_list,
                   last_A = last_A, 
                   pobs_state = new_pobs_state,
                   shuffle = False, #deprecated
                   Tmice = Tmice,
                   Tstandard = Tstandard,
                   num_cycles = num_cycles)        
            new_imp_state = None #don't need this
        else:
            raise Exception("impute_method choice is not currently supported.")

    # if nothing is missing, just set new_imp_state equal to the new_pobs_state
    else:
        # just make a deepcopy!
        new_imp_state = copy.deepcopy(new_pobs_state)
        if impute_method in MImethods:
            new_imp_state_list = [new_pobs_state] * int(K)
    
    #if there is missingness and get here, impute method is not "random action"
    #so new_imp_state is defined and test works
    return new_imp_state, new_imp_state_list


def get_action(last_imp_state : tuple, 
               last_imp_state_list : list,
               impute_method : str,
               action_list : list, 
               Q : dict, 
               epsilon : float,
               action_option : str):
    """
    Parameters
    ----------
    last_imp_state : tuple
        the state that should be used to determine the action
        if this argument is None, that means impute_method must be "random_action"
        and a random action is taken
        
    last_imp_state_list : list of tuples
        if impute_method is an MI method, this list of states is used to determine
        action with combination method as specified by action_option
    
    impute_method : str
        method of imputation being used. Only matters if this is "random_action",
        a multiple imputation method, or something else
    
    action_list : list
        list of possible actions
        
    Q : dict
        dictionary as of format as initialized by init_Q in ImputerTools
        
    epsilon : float between 0 and 1
        set to 0 for greedy action selection 
    
    action_option : str
        if impute_method is a MI method, this specifies how to do action selection
        using the list of states in last_imp_state_list

    Returns
    -------
    An element of action_list representing next action to take

    """
    if impute_method == "random_action" and last_imp_state is None:
        action = action_list[np.random.choice(a=len(action_list))]
                      
    elif impute_method in MImethods:
        action = impt.select_action(state = last_imp_state_list, 
                          action_list = action_list,
                          Q = Q, 
                          epsilon = epsilon, 
                          option = action_option)
    else:
        #for missing_state, get_imputation will already have converted np.nan to whatever 
        #filler value is being used and Q will already have incorporated that value
        action = impt.select_action(last_imp_state, action_list, Q, epsilon)
        
    return(action)
    
    


###############################################################################
# MAIN FUNCTION FOR RUNNING METHOD
###############################################################################

def run_RL(env, logger,
           miss_mech, # environment-missingness governor "MCAR", "Mcolor", "Mfog"
           impute_method, # "last_fobs", "random_action", "missing_state", "joint", "mice"
           action_option, # voting1, voting2, averaging
           K, #number of multiple imputation chains
           num_cycles, #number of cycles used in Mice
           epsilon, # epsilon-greedy governor 
           alpha, # learning rate 
           gamma, # discount factor 
           max_iters, # how many iterations are we going for?
           seed, # randomization seed
           verbose=False, # intermediate outputs or nah?
           missing_as_state_value = -1,
           save_Q = True,
           log_per_episode = True,
           log_per_t_step = False):  #always logs per episode
    """
    Runs the following overall pipeline for max_iters time steps.
    At each time step
        1. Take an action according to action_method
        2. Observe next state, partially with missingness, and observe reward
        3. Run an impute method if needed, possibly with multiple imputation
        4. Learning: Update Q matrix, T matrix
        (5. take an action...)
    
    env specifications:
        
        env must be an instance of a class for running an RL environment. 
        It must have the following methods and attributes
        
            env.action_list - returns list of possible actions
            
            env.state_value_lists - returns a list of lists where the i^th sublist
                                    is all the possible values of the i^th element of
                                    the state vector. order matters.
                                    e.g., for a 3-dimensional state, each component binary,
                                    this would be [[0,1],[0,1],[0,1]]
                                   
            env.step(action) - takes an action in the environment. <action> must
                                be contained in env.action_list. Must update the 
                                current state.
            
            env.current_state - current state of the environment. Must be updated by
                                env.step()
         
            env.<miss_mech> - miss_mech argument of this function should match the name of a
                   missingness mechanism method which returns the current state only with possible
                   missingness
                               
            env.get_filename(miss_mech) - returns a part of a filename to be used in saving
                            files related to this instance of the environment. This should
                            for example encode any parameter settings specific to this run.
                            This does not need to include any imputation settings as those are
                            automatically added to the file name by this function.
                            
         
        logger specifications:
        
            #TODO: this may need to be revised to fit gymnasium set-up 
        
            if logger is None, then no tracking of per-episode quantities from the 
            RL run are saved. (separately, if save_Q is saved, the final Q function estimate
                               will be)
            
            if logger is given, enables saving of trajectories in per-step or per-episodes logs
        
            if log_per_episode = True, logger instance should have:
                logger.start_epsiode() 
                logger.update_epsisode_log(env, new_pobs_state, reward)
                logger.finish_and_reset_epsiode()  
                logger.episode_log  - a pandas data frame where logging is stored

        
            if log_per_t_step = True, logger instance should have:
                logger.start_t_step(t_step)
                logger.finish_t_step(env, action, new_true_state, new_pobs_state, reward)
                logger.t_step_logs - a pandas data frame where logging is stored
                
            see LakeWorldEnvironments.py for examples of what these functions might do
        
    """
    SingleImpMethods = ["random_action","last_fobs",  "last_obs", "missing_state"]
    MImethods = ["joint", "mice", "joint-conservative"]
    if impute_method in MImethods:
        assert K is not None
        assert K >=1
    else:
        K = 0 #default to make below work, does nothing
    if impute_method == "mice":
        assert num_cycles is not None and num_cycles >= 1
    assert epsilon >= 0
    assert alpha >= 0
    assert gamma >= 0
    if log_per_episode:
        assert logger is not None, "logger object must be specified if log_per_episode = True"
    if log_per_t_step:
        assert logger is not None, "logger object must be specified if log_per_t_step = True"
        
    #--------------------------------------------------------------- 
   
    # For convenience
    MImethods = ["joint", "mice", "joint-conservative"]
  
    # Get other environment attributes
    action_list = env.action_list
    state_value_lists = env.state_value_lists
    
    # set our seed for use in multiple trials
    np.random.seed(seed)
    
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
    last_obs_state_comp = env.current_state

    if log_per_episode:
        logger.start_epsiode() #start episode 1

    ###########################################################################
    for t_step in range(max_iters):
        
        if log_per_t_step:
            logger.start_t_step(t_step)
    
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
        if hasattr(env, miss_mech):
            miss_method = getattr(env, miss_mech)
            new_pobs_state = miss_method()
        else:
            raise Exception(f"env does not have a missing method called {miss_mech}")
          
        # Impute for the new_pobs_state, if needed
        new_imp_state, new_imp_state_list = rlt.get_imputation(impute_method = impute_method,
                           new_pobs_state  = new_pobs_state, 
                           last_fobs_state = last_fobs_state,
                           last_obs_state_comp = last_obs_state_comp,
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
            
        # Update last_obs_state_comp with any parts that were observed  
        last_obs_state_comp = [i if ~np.isnan(i) else j for (i,j) in zip(new_pobs_state, last_obs_state_comp)]      

        # now that we have updated Q and T, update 'lasts' for next round
        last_pobs_state = copy.deepcopy(new_pobs_state)
        last_imp_state = copy.deepcopy(new_imp_state)
        if impute_method in MImethods:
            last_imp_state_list = copy.deepcopy(new_imp_state_list)
        

        # LOGGING 
        if log_per_episode: 
            logger.update_epsisode_log(env, new_pobs_state, reward)
        if log_per_t_step:
            logger.finish_t_step(env, action, new_true_state, new_pobs_state, reward)
        
        # status update?
        if verbose == True and log_per_episode:
            if (t_step+1) % 10 == 0 and len(logger.episode_logs.index) >= 20:
                clear_output(wait=True)
                print(f"""Timestep: {t_step+1}, Past 20 Mean Epi. Sum Reward: {np.round(logger.episode_logs.loc[-20:].total_reward.mean(), 3)}, Fin. Episodes: {len(logger.episode_logs.index)}, Past 20 Mean Path Length: {np.round(logger.episode_logs.loc[-20:].num_steps.mean(), 3)}""")
            elif (t_step+1) % 10 == 0:
                clear_output(wait=True)
                print(f"Timestep: {t_step+1}")
                print(f"Total Reward So Far This Episode: {logger.total_reward}")
        elif verbose == True and log_per_t_step:
            pass
            #TODO: add some alternative printing if only doing tstep logging
                
        if terminal and log_per_episode:
            logger.finish_and_reset_epsiode()            
        
    
    ###############################################################
    ##### SAVING OUR LOGS FILE TO A .CSV OUTPUT ###################
    ###############################################################
    
    # check if we have a results folder
    if "results" not in os.listdir():
        os.makedirs("results")

    # start filename with environment-specific aspects
    fname = env.get_filename(miss_mech)

    # add in our imputation mechanism
    IM_desc = impute_method.replace("_", "-")
    fname += f"_IM={IM_desc}"

    if impute_method in MImethods:
        # encode NC: num_cycles, K: no. of imputations, PS: p_shuffle
        fname += f"_NC={num_cycles}_K={K}"
        
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
        os.makedirs(f"results/{fname}")
  
    # save the EPISODES + STEPWISE log files to a .csv 
    if log_per_episode:
        logger.episode_logs.to_csv(f"results/{fname}/episodic_seed={seed}.csv", index=False)
    if log_per_t_step:
        logger.t_step_logs.to_csv(f"results/{fname}/stepwise_seed={seed}.csv", index=False)

    # save the Q matrix to a .pickle
    if save_Q:
        with open(f"results/{fname}/Q_seed={seed}.pickle", "wb") as file:
            pickle.dump(Q, file)
        
    # just for kicks
    print("Tada!")
    return 1
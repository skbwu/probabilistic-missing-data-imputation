import numpy as np
import copy
import ImputerTools as impt


SingleImpMethods = ["random_action", #this ultimately an attribute
             "last_fobs1",
             "last_fobs2",
             "missing_state"]

MImethods = ["joint", #TODO: better names for this? joint-synthetic?
             "mice",
             "joint-conservative"]


def get_imputation(impute_method : str,
                   new_pobs_state : tuple, 
                   last_fobs_state : tuple,
                   last_A,
                   last_state_list : list, 
                   K : int,
                   Tstandard = None, 
                   Tmice = None, 
                   num_cycles = None,
                   missing_as_state_value = None):
    """
    TODO: fill this in
    TOOD: test this function


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

        if impute_method == "last_fobs1":
            new_imp_state = copy.deepcopy(last_fobs_state)
            
        elif impute_method == "last_fobs2":
            new_imp_state = tuple([i if ~np.isnan(i) else j for (i,j) in zip(new_pobs_state, last_fobs_state)])
               
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


def get_action(pobs_state : tuple,
               last_fobs_state : tuple,
               last_imp_state_list : list,
               impute_method : str,
               action_list : list, 
               Q : dict, 
               epsilon : float,
               action_option : str,
               missing_as_state_value = -1):
    
    # do we have any missing state values?
    if np.any(np.isnan(pobs_state)):
        
        # deal with it accordingly to get imputed actions
        if impute_method == "last_fobs1":
            action = impt.select_action(last_fobs_state, action_list, Q, epsilon) #TODO: fix
            
        elif impute_method == "last_fobs2":
            action = impt.select_action(last_fobs_state, action_list, Q, epsilon) #TODO: fix
            
        elif impute_method == "random_action":
            action = action_list[np.random.choice(a=len(action_list))]
            
        elif impute_method == "missing_state":
            # for this method only, we need to convert np.nan to -1
            pobs_state_temp = tuple([val if ~np.isnan(val) else missing_as_state_value for val in pobs_state])
            action = impt.select_action(pobs_state_temp, action_list, Q, epsilon)
            
        elif impute_method in MImethods:
            action = impt.select_action(state = last_imp_state_list, 
                          action_list = action_list,
                          Q = Q, 
                          epsilon = epsilon, 
                          option = action_option)
        else:
            raise Exception("impute_method choice is not currently supported.")
    
    # if no missingness, select an action by standard epsilon greedy 
    else:
        action = impt.select_action(pobs_state, action_list, Q, epsilon)
        
    return(action)
    
    

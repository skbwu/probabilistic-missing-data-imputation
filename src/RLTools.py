import numpy as np
import copy
import ImputerTools as impt



def get_imputation(impute_method, new_pobs_state, 
                   last_fobs_state, last_A, last_state_list, 
                   K, MImethods,
                   Tstandard = None, Tmice = None, num_cycles = None):
    """
    TODO: fill this in
    TOOD: test this function

    Parameters
    ----------
    impute_method : TYPE
        DESCRIPTION.
    new_pobs_state : TYPE
        DESCRIPTION.
    last_fobs_state : TYPE
        DESCRIPTION.
    last_A : TYPE
        DESCRIPTION.
    last_state_list : TYPE
        DESCRIPTION.
    K : TYPE
        DESCRIPTION.
    MImethods : TYPE
        DESCRIPTION.
    Tstandard : TYPE, optional
        DESCRIPTION. The default is None.
    Tmice : TYPE, optional
        DESCRIPTION. The default is None.
    num_cycles : TYPE, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    If method is a non-multiple imputation method: an imputed state
    If method is a multiple imputation method: a list of K imputed states
    
    If there is no missingness in the new_pobs_state (the current state), then
    the imputed state returned is simply a copy of the current state

    """
    new_imp_state_list = None
    
    if np.any(np.isnan(np.array(new_pobs_state)).mean()):
        
        #Case where impute nothing
        if impute_method == "random_action":
            return None, None

        if impute_method == "last_fobs1":
            new_imp_state = copy.deepcopy(last_fobs_state)
            
        elif impute_method == "last_fobs2":
            new_imp_state = tuple([i if ~np.isnan(i) else j for (i,j) in zip(new_pobs_state, last_fobs_state)])
               
        elif impute_method == "missing_state":

            # swapping np.nan to -1 to play nicer with dictionary indexing.
            new_imp_state = tuple([val if ~np.isnan(val) else -1 for val in new_pobs_state])
      
        elif impute_method in MImethods:

            #generate list of imputed values
            #note: because first state already observed, will only
            #get here when already have defined action variable 
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



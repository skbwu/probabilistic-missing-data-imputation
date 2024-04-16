# -*- coding: utf-8 -*-
"""
Functions for generating settings and things like that
"""
import numpy as np


def get_settings():
    
    # a list to store all of our settings
    settings = []
    
    # create a list of settings that we want to try
    for gamma in [1.0, 0.5, 0.0]: # save 0.75 + 0.25 for wave 2.
        for alpha in [1.0, 0.1]: # boobooed with 0.1. NOW CORRECTED!
            for epsilon in [0.0, 0.05]:
                for p_switch in [0.1, 0.0]:
                    for p_wind in [0.0, 0.1]: # save 0.2 for wave 2.
                        for allow_stay_action in [True, False]:
                        
                            # assign the i+j dimensions
                            p_wind_i, p_wind_j = p_wind, p_wind
                            
                            # now, let's be careful and set irrelevant parameters to None
                            for env_missing in ["MCAR", "Mcolor", "Mfog"]:
                                
                                # MCAR-specific "thetas" parameter
                                if env_missing != "MCAR":
                                    thetas_list = [None]
                                elif env_missing == "MCAR":
                                    thetas_list = [np.ones(3) * theta for theta in [0.0, 0.05, 0.1, 0.2, 0.4]]
                                else:
                                    raise Exception("Something went wrong with env_missing.")
                                    
                                # iterate through the "thetas" intended ONLY FOR MCAR, will None-out otherwise.
                                for thetas in thetas_list:
                                    
                                    # let's work thru thetas_in and thetas_out accordingly
                                    if env_missing != "Mfog":
                                        thetas_IO_list = [(None, None)]
                                    elif env_missing == "Mfog":
                                        thetas_IO_list = [(np.array([0.5, 0.5, 0.5]), np.array([0.0, 0.0, 0.0])),
                                                          (np.array([0.25, 0.25, 0.25]), np.array([0.1, 0.1, 0.1]))]
                                    else:
                                        raise Exception("Something went wrong with env_missing.")
                                        
                                    # iterate through the thetas_IO intended only for Mfog
                                    for thetas_IO in thetas_IO_list:
                                        
                                        # just unpack from the tuple
                                        thetas_in, thetas_out = thetas_IO
                                        
                                        # theta_dict for "Mcolor" option
                                        if env_missing != "Mcolor":
                                            theta_dict_list = [None]
                                        elif env_missing == "Mcolor":
                                            theta_dict_list = [{0 : np.array([0.1, 0.1, 0.1]),
                                                                1 : np.array([0.2, 0.2, 0.2]),
                                                                2 : np.array([0.3, 0.3, 0.3])},
                                                               {0 : np.array([0.1, 0.1, 0.0]),
                                                                1 : np.array([0.2, 0.2, 0.0]),
                                                                2 : np.array([0.3, 0.3, 0.0])},]
                                        else:
                                            raise Exception("Something went wrong with env_missing.")
                                        
                                        # iterate through the theta_dict parameter
                                        for theta_dict in theta_dict_list:
                                            
                                            # let's work with the impute_method
                                            for impute_method in ["last_fobs", "random_action", 
                                                                  "missing_state", "joint", "joint-conservative"]: # MICE WAS YEETED CUZ TOO SLOW
                                                
                                                # start joint/mice specific settigns
                                                if impute_method not in ["joint", "joint-conservative", "mice"]:
                                                    
                                                    # create a bunch of lists of Nones
                                                    p_shuffle_list = [None]
                                                    num_cycles_list = [None]
                                                    K_list = [None]
                                                    
                                                # joint + mice specific matters
                                                # MICE WAS EXTREMELY EXTREMELY SLOW. THUS, YEETED.
                                                elif impute_method in ["joint", "joint-conservative", "mice"]:
                                                    
                                                    # create a bunch of lists of Nones
                                                    p_shuffle_list = [0.0, 0.1] # saving 0.05 for wave2
                                                    num_cycles_list = [1] # [10, 20]
                                                    K_list = [1, 5, 10] # 4/16/2024: ADDING IN K=5!
                                                    
                                                # as usual, throw a hissy fit
                                                else:
                                                    raise Exception("Something went wrong with env_missing.")
                                                    
                                                # iterate through these MICE specific settings
                                                for p_shuffle in p_shuffle_list:
                                                    for num_cycles in num_cycles_list:
                                                        for K in K_list:
                                                            
                                                            # create our setting-tuple + add to our list
                                                            setting = (gamma, alpha, epsilon, p_switch, 
                                                                       p_wind_i, p_wind_j, allow_stay_action, env_missing, 
                                                                       thetas, thetas_in, thetas_out, theta_dict, 
                                                                       impute_method, p_shuffle, num_cycles, K)
                                                            settings.append(setting)
                                                            
    return(settings)
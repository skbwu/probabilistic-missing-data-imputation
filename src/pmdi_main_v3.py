# 7/16/2024: these are the only modules we have to load
import SimulationHelpers as shelpers
import GridWorldMain as gwm

# fix our verbose and max_iters variable (we'll deal with seed at the end!)
verbose, max_iters = False, 50000

# get all our settings
settings = shelpers.get_settings()  #4/16/2024 moved this to a helper so easier to test

# which index are we starting at?
start_idx = int(sys.argv[1])

# which settings are we working on?
for i in range(start_idx*16, (start_idx*16)+16): # REVISED 4/16/2024 to account for the K=5 added settings.
    
    # unpack our settings
    gamma, alpha, epsilon, p_switch, p_wind_i, p_wind_j, allow_stay_action, env_missing, thetas, thetas_in, thetas_out, theta_dict, impute_method, p_shuffle, num_cycles, K = settings[i]
    
    # do our random seeding
    for seed in range(5):
        
        # just call our runner function
        output = gwm.runner(p_switch=p_switch, p_wind_i=p_wind_i, p_wind_j=p_wind_j,
                            allow_stay_action = allow_stay_action, env_missing=env_missing, 
                            thetas=thetas, thetas_in=thetas_in, thetas_out=thetas_out, theta_dict=theta_dict,
                            impute_method=impute_method, p_shuffle=p_shuffle, num_cycles=num_cycles, K=K, 
                            epsilon=epsilon, alpha=alpha, gamma=gamma, max_iters=max_iters, seed=seed, verbose=verbose)
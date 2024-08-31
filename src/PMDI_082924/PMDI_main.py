from LakeWorldMain import run_LakeWorld
import numpy as np
import sys, os, pickle
from collections import ChainMap

# use command-line arguments to figure out which of 1728x combined settings we're running
with open("combined_settings.pickle", "rb") as file:
    combined_settings = pickle.load(file)
combined_setting = combined_settings[int(sys.argv[1])] # between 0 and 1727, inclusive

# also load in our env settings
with open("env_settings.pickle", "rb") as file:
    env_settings = pickle.load(file)
    
# go thru each set of env_setting + our 5x seeds + run our simulation
for env_setting in env_settings:
    for seed in range(5):
        
        # concatenate all of our settings together + manually add in our seed
        setting = dict(ChainMap(combined_setting.copy(), env_setting.copy()))
        setting["seed"] = seed
        
        # run our payload experiment
        output = run_LakeWorld(**setting)
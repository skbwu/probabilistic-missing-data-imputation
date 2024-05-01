import numpy as np
import pandas as pd
import sys, copy, os, shutil
    
# cmd-line argument that tells us which piece of the pie we will be working on
split = int(sys.argv[1]) # let's go for 32 splits

# get our column names for our dataframe + create the dataframe
# t-color-p is the position missing probability, t-color-c is the color's missing probability
columns = ["PS", "PW", "MM", "ASA", "theta", "t-color-p", "t-color-c", "t-in", "t-out", 
           "IM", "NC", "K", "p-shuf", "max-iters", "eps", "a", "g", "seed", "num_episodes", 
           "mean_total_reward", "mean_steps_river", "mean_path_length", "mean_wallclock_time",
           "mean20_total_reward", "mean20_steps_river", "mean20_path_length", "mean20_wallclock_time"]
df = pd.DataFrame(columns=columns, data=None)

# only looking at the EPISODIC LOGS when building aggregate logs
fnames = [f for f in sorted(os.listdir("results")) if "PS" in f][int(split*864) : int(split*864) + 864]

# iterate through all of our file names
for fname in fnames:
    
    # load in each seed - WE ONLY CARE ABOUT EPISODIC LOGS WHEN LOOKING AT AGGREGATE PERFORMANCE!
    trial_names = sorted([f for f in os.listdir(f"results/{fname}") if "episodic" in f])
    for trial_name in trial_names:
        
        # start the row with our settings
        row = [fname.split("PS")[1].split("=")[1].split("_")[0]]
        for col in columns[1:17]:
            
            # split into two cases!
            if col not in ["t-color-p", "t-color-c"]:
                
                # add the value if it exists, else just append a None
                if ("_" + col + "=") in fname:
                    row.append(fname.split("_" + col + "=")[1].split("_")[0])
                else:
                    row.append(None)
            
            # deal with t-color on its own:
            else:
                
                # need to unpack a pair of arguments!
                if "_t-color=" in fname:
                    
                    # get the t-color-p and t-color-c + add to our row
                    t_color_p, t_color_c = fname.split("_t-color=")[1].split("_")[0].split("+")
                    t_color_p, t_color_c = float(t_color_p), float(t_color_c)
                    
                    # add in either t-color-p or t-color-c
                    if col == "t-color-p":
                        row.append(t_color_p)
                    elif col == "t-color-c":\
                        row.append(t_color_c)
                    else:
                        raise Exception("Something went wrong.")
                    
                # if we don't have it, just put a None
                else:
                    row += [None]
        
        # get our seed + add it to our columns
        seed = int(trial_name.split(".csv")[0].split("=")[1])
        row.append(seed)
        
        # load in our files + record our metrics of interest
        logs = pd.read_csv(f"results/{fname}/{trial_name}")
        
        # number of successfully completed episodes
        num_episodes = len(logs.index)
        
        # total means over all episodes
        mean_total_reward = logs.total_reward.mean()
        mean_steps_river = logs.steps_river.mean()
        mean_path_length = logs.path_length.mean()
        mean_wallclock_time = logs.wall_clock_time.mean()
        
        # let's look at last 20 episodes too, if applicable
        if num_episodes >= 20:
            mean20_total_reward = logs.total_reward.values[-20:].mean()
            mean20_steps_river = logs.steps_river.values[-20:].mean()
            mean20_path_length = logs.path_length.values[-20:].mean()
            mean20_wallclock_time = logs.wall_clock_time.values[-20:].mean()
        else:
            mean20_total_reward = logs.total_reward.mean()
            mean20_steps_river = logs.steps_river.mean()
            mean20_path_length = logs.path_length.mean()
            mean20_wallclock_time = logs.wall_clock_time.mean()
        
        # add to our row
        row += [num_episodes, mean_total_reward, mean_steps_river, mean_path_length, mean_wallclock_time, 
                mean20_total_reward, mean20_steps_river, mean20_path_length, mean20_wallclock_time]
        
        # add to our dataframe
        df.loc[len(df.index)] = row
        
# casting to various types as necessary
df.PS = df.PS.astype(float)
df.PW = df.PW.astype(float)

# ASA cast as True or False
df.ASA = [True if val == "T" else False for val in df.ASA.values]

# continue with the other columns
df.theta = df.theta.astype(float)
df["t-color-p"] = df["t-color-p"].astype(float)
df["t-color-c"] = df["t-color-c"].astype(float)
df["t-in"] = df["t-in"].astype(float)
df["t-out"] = df["t-out"].astype(float)
df.NC = df.NC.astype(float)
df.K = df.K.astype(float)
df["p-shuf"] = df["p-shuf"].astype(float)
df["max-iters"] = df["max-iters"].astype(float)
df.eps = df.eps.astype(float)
df.a = df.a.astype(float)
df.g = df.g.astype(float)

# save our results
split_str = str(split).zfill(2)
df.to_csv(f"logs/aggregate_logs_v3_split={split_str}.csv", index=False)
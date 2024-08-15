# Missing Data Multiple Imputation for Tabular Q-Learning in Online RL
This repo accompanies the paper "Missing Data Multiple Imputation for Tabular Q-Learning in Online RL" written for Susan Murphy's [Stat 234](https://people.seas.harvard.edu/~samurphy/teaching/stat234spring2024/syllabus.htm)  class on reinforcement learning at Harvard in Spring 2024 by Kyla Chasalow and Skyler Wu.

Within the `src` directory:

* `LakeWorldEnvironment` contains functions for setting up an instance of our LakeWorld RL environment, instances of which include methods for generating missingness in their states
* `LakeWorldMain` contains the main function for running the Lake World Environment
* `ImputerTools` contains functions for implementing imputation ensembles
* `RLTools` contains the functions for running our RL pipeline
* `MissingMechanisms` contains general functions for generating missingness that are used in multiple other places
* `SimulationHelpers` contains functions that aid in running our simulations
* `LakeWorldTests` contains some tests of various functions in the above scripts

**To run our simulation:** run `pmdi_main_v3_runscript_driver.sh` after making appropriate updates to filepaths. **TODO** update this

**To generate analysis files**: run `analyzer_main_runscript_driver.sh` and use `analyzer.ipynb` to combine the resulting files.

**To generate figures**: run `figures.ipynb`

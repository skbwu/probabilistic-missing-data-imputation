# Missing Data Multiple Imputation for Tabular Q-Learning in Online RL
This repo accompanies the paper "Missing Data Multiple Imputation for Tabular Q-Learning in Online RL" written for Susan Murphy's [Stat 234](https://people.seas.harvard.edu/~samurphy/teaching/stat234spring2024/syllabus.htm)  class on reinforcement learning at Harvard in Spring 2024.

Within the `src` directory:

* `GridWorldMain` contains the main function for running various methods in our environment
* `GridWorldEnvironment` contains functions for setting up and visualizing the environment, taking an action in that environment, and generating the missingness under various mechanisms
* `ImputerTools' contains functions for implementing imputation ensembles
* `GridWorldTests` contains some tests of various functions in the above scripts
* `SimulationHelpers` contains functions that aid in running our simulations

**To run our simulation:** run `pmdi_main_v3_runscript_driver.sh` after making appropriate updates to filepaths.

**To generate analysis files**: run `analyzer_main_runscript_driver.sh` and use `analyzer.ipynb` to combine the resulting files.

**To generate figures**: run `figures.ipynb`

# Missing Data Multiple Imputation for Tabular Q-Learning in Online RL
This repo accompanies the paper "Missing Data Multiple Imputation for Tabular Q-Learning in Online RL" written for Susan Murphy's [Stat 234](https://people.seas.harvard.edu/~samurphy/teaching/stat234spring2024/syllabus.htm)  class at Harvard in Spring 2024.

Within the `src` directory:

* `GridWorldMain` contains the main function for running various methods in our environment
* `GridWorldHelpers` contains functions for setting up and visualizing the environment, taking an action in that environment, generating the missingness under various mechanisms, and doing Q learning
* `GridWorldImputers` contains functions for implementing MI
* `SimulationHelpers` contains functions that aid in running our simulations
* `GridWorldTests` contains some tests of various functions in the above scripts

**To run our simulation:** run `pmdi_main_v3_runscript_driver.sh' after making appropriate updates to filepaths.

**To generate analysis files**: run ``analyzer_main_runscript_driver.sh` and use `analyzer.ipynb' to combine the resulting files.

**To generate figures**: run `figures.ipynb`

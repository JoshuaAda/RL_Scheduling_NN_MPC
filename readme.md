### Readme for the repositiory RL_Scheduling_NN_MPC
 This code is related to the paper 
 "Multi-system cloud control with embedded neural
 network approximators using safe reinforcement
 scheduling" by Joshua Adamek and Sergio Lucia,
 under revision for the European Conference 
 of Control, 2024.
 -----------------

It is is written in Python 3.7.16 and uses the 
[DO-MPC](https://www.do-mpc.com/en/latest/) framework for control and simulation purposes.
For the scheduling of cloud tasks using reinforcement learning, the
[Stable Baselines 3](https://stable-baselines3.readthedocs.io/en/master/) framework along with the [GYM](https://www.gymlibrary.dev) environment
is used. The necessary packages can be installed using the _environment.yml_ file presented in the repository. 
-------------------
Currently, this code is under development and is not yet ready for external use.
You find the training results of the models used in the paper in the _results/images_ folder.
The models have not yet been uploaded due to the size, in theory you can train them yourself using the _ppo_learn.py_ script.
Evaluating the results both on mockup classes and with the "real" cloud simulation can be done using _evaluate.py_.
-------------------
The _changing6.pkl_ file contains the scenario used for the control case study. In the respective _trackingXY.npz_
files you find the results for the relative tracking error for all three comparisons.

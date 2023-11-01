### Readme for the repositiory RL_Scheduling_NN_MPC
 #
This repository is related to the paper 
 "Multi-system cloud control with embedded neural
 network approximators using safe reinforcement
 scheduling" by Joshua Adamek and Sergio Lucia,
 under revision for the European Conference 
 of Control, 2024.
#

 -----------------
The code is written in Python 3.7.16 and uses the 
[DO-MPC](https://www.do-mpc.com/en/latest/) framework for control and simulation purposes.
For the scheduling of cloud tasks using reinforcement learning, the
[Stable Baselines 3](https://stable-baselines3.readthedocs.io/en/master/) framework along with a self-created [GYM](https://www.gymlibrary.dev) environment
is used. The necessary packages can be installed using the _requirements.txt_ file presented in the repository. 
#

-------------------

You find the training results of the models used in the paper in the _results/images_ folder.
The _changing6.pkl_ file contains the scenario used for the control case study. In the respective _trackingXY.npz_
files you find the results for the relative tracking error for all three comparisons. As it is not yet possible to upload the PPO model
used in the case studies due to storage size, you cannot run the case studies yourself in the exact manner. However, you
can already train your own model and evaluate it
#

-------------------
These are two tasks you can perform with this repository:
1. Train your own model for the scheduling of cloud tasks. For example the smaller scenario model in the paper
has been derived using the following command:
   ```
   train.py --t_s_length=16, --max_load=2, --failure_prob=5, --system_size=16, --num_workers=8
   ```
   The default parameters are already set to train for the larger scenario. In case you want to train only the PPO agent without
   the safe RL approach set ```--use_both=False```.
2. Evaluate a model. For this you can use the _evaluate.py_ file. The default parameters are set to evaluate the larger scenario model.
   This script is also used for the control case study. With the following flags you can rerun the results for the safe reinforcement learning scheduler
   ``` 
   evaluate.py --use_true=True, --num_iter=5000
   ```

   Setting the flag ```--use_mpc=True``` will give you the results for controlling only with online MPC's. Setting the flags
   ```--use_both=False, --use_priority=True``` leads to an evaluation of the heuristic scheduler.

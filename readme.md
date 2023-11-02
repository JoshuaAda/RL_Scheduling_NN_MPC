# ECC 2024: Multi-system cloud control with embedded neural network approximators using safe reinforcement scheduling
## Abstract
With the increased complexity of optimal control algorithms demanding more real-time computational power for systems with limited embedded hardware
, cloud control also known as Control as a Service (CaaS) is becoming of interest in recent research.
The need for a backup controller on-premise for safety purposes resulted in the controlling algorithm
presented in [[1]](https://ieeexplore.ieee.org/document/10284471). A neural network approximation of the explicit MPC serves as the embedded
controller, while the cloud or central computing unit (CCU) only controls in case of errors or model changes.
In this paper, the algorithm is extended to multiple systems by proposing a scheduling algorithm for the performed control
task, to reduce the CCU makespan. To improve from a simple heuristic, a safe reinforcement learning method
is proposed to deal with the real-time decision needs of the scheduling problem.
Finally, a case study shows a possible application of the proposed control structure
along with the benefits of the method.
 ## About the repository
This repository is related to the paper 
 "Multi-system cloud control with embedded neural
 network approximators using safe reinforcement
 scheduling" by Joshua Adamek and Sergio Lucia,
 under revision for the European Conference 
 of Control, 2024.

 ## Dependencies
 
The code is written in Python 3.7.16 and uses the 
[DO-MPC](https://www.do-mpc.com/en/latest/) framework for control and simulation purposes.
For the scheduling of cloud tasks using reinforcement learning, the
[Stable Baselines 3](https://stable-baselines3.readthedocs.io/en/master/) framework along with a self-created [GYM](https://www.gymlibrary.dev) environment
is used. The necessary packages can be installed using the _requirements.txt_ file presented in the repository. 

## Reconstruction of the results

You find the training curves of the RL models used in the paper in the _results/images_ folder.
The RL models for the comparison study can be found in the result folder and used as instructed below using the specific model parameters in the ```evaluation.py``` file.
The _changing_scenario.pkl_ file as well as _x_tilde_scenario.pkl_ contain the scenario used for the control case study and can be run as instructed below with the flag ```--run_first=False```.
Furthermore, you find the results for the relative tracking error for all three case studies directly in _results/Tracking_XY.npz_.
## Instructions

Be careful in using the default parameters for your own models as it will override the paper results and models given in the _results_ folder.
These are the two tasks you can perform with this repository:
1. Train your own model for the scheduling of cloud tasks. For example the smaller scenario model in the paper
has been derived using the following command:
   ```
   train.py --t_s_length=16, --max_load=2, --failure_prob=5, --system_size=16, --num_workers=8
   ```
   The default parameters are already set to train for the larger scenario. In case you want to train only the PPO agent without
   the safe RL approach set ```--use_both=False```.
2. Evaluate a model. For this you can use the _evaluate.py_ file. The default parameters are set to evaluate the larger scenario model.
   This script is also used for the control case study. With the following flags you can rerun the control case study with the safe reinforcement learning scheduler
   ``` 
   evaluate.py --use_true=True, --num_iter=5000
   ```

   Setting the flag ```--use_mpc=True``` will give you the results for controlling only with online MPC's. Setting the flags
   ```--use_both=False, --use_priority=True``` leads to an evaluation of the heuristic scheduler.


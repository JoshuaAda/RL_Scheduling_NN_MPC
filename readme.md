# ECC 2024:Safe and efficient multi-system neural controllers via reinforcement learning-based scheduling
## Abstract
With the increasing use of advanced control methods
such as model predictive control, the demand for real-time
computational power continues to increase. Meeting this demand
can be especially challenging when multiple systems need
to be controlled with the potentially limited embedded hardware
of the individual systems. Utilizing a central computing unit,
such as a cloud, can provide the required additional computing
power at the expense of increased economic and energetic costs.
In this work, we consider a scenario where embedded
controllers based on neural networks that imitate a model
predictive controller need to be supported by a central computing
unit in case of errors or model changes. In order to
minimize the necessary amount of central computing used to
guarantee a safe simultaneous operation of multiple systems,
we propose a reinforcement learning-based scheduling of the
different necessary computing tasks. A case study illustrates
the benefits of the proposed control structure.
 ## About the repository
This repository is related to the paper 
 "Multi-system cloud control with embedded neural
 network approximators using safe reinforcement
 scheduling" by Joshua Adamek and Sergio Lucia,
 under revision for the European Conference 
 of Control, 2024.
For a first introduction to the topic, we propose to read the paper first. Furthermore, we attached the full description of 
the MDP used as a model paradigm for the scheduling of cloud tasks in this subchapter:
![alt text](https://github.com/JoshuaAda/RL_Scheduling_NN_MPC/blob/main/gesamt_mdp_2.pdf)
Markov Decision Process for a single system. The states are described over its current performed task (\textbf{H}ealthy state, \textbf{U}nhealthy state, \textbf{S}ampling task, \textbf{L}oad data task, \textbf{C}ontrol task, \textbf{T}raining task), the deadline and the number of workers assigned to the task. For simplicity, the multi-step prediction states and the uncertain deadlines are omitted. The action determines whether to move on with a process or not and how many workers are assigned to a task. After completion, the respective system returns to the healthy state, where the system is controlled by the embedded controller.
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


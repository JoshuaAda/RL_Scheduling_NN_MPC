import sys
import os
from callback import SaveOnBestTrainingRewardCallback
import gym
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common import results_plotter
import tikzplotlib
from stable_baselines3.common.results_plotter import plot_results,load_results,ts2xy,window_func
from stable_baselines3.common.monitor import Monitor
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from typing import Callable, List, Optional, Tuple
EPISODES_WINDOW = 1000
sys.path.append('./gym-examples')
import gym_examples

def plot_curves(
    xy_list: List[Tuple[np.ndarray, np.ndarray]], x_axis: str, title: str, figsize: Tuple[int, int] = (8, 2)
) -> None:
    """
    plot the curves

    :param xy_list: the x and y coordinates to plot
    :param x_axis: the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param title: the title of the plot
    :param figsize: Size of the figure (width, height)
    """

    plt.figure(title, figsize=figsize)
    max_x = max(xy[0][-1] for xy in xy_list)
    min_x = 0
    for _, (x, y) in enumerate(xy_list):
        #plt.scatter(x, y, s=2)
        # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
        if x.shape[0] >= EPISODES_WINDOW:
            # Compute and plot rolling mean with window of size EPISODE_WINDOW
            x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean)
            plt.plot(x, y_mean)
    plt.xlim(min_x, max_x)
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel("Episode Rewards")
    plt.tight_layout()
def plot_own_results(dirs: List[str], num_timesteps: Optional[int], x_axis: str, task_name: str, figsize: Tuple[int, int] = (8, 2)):
    data_frames = []
    for folder in dirs:
        data_frame = load_results(folder)
        if num_timesteps is not None:
            data_frame = data_frame[data_frame.l.cumsum() <= num_timesteps]
        data_frames.append(data_frame)
    xy_list = [ts2xy(data_frame, x_axis) for data_frame in data_frames]
    plot_curves(xy_list, x_axis, task_name, figsize)
def train(maxi=16,p=5,system_size=4,num_workers=8,use_true=False):
    log_dir = "results/model_"+str(maxi)+"_"+str(p)+"_"+str(system_size)+"_"+str(num_workers)+"_results/"
    os.makedirs(log_dir, exist_ok=True)
    if use_true:

            x0_list = [np.array([[20], [20], [10], [np.pi * 2 / 3]]) * np.random.random((4, 1)) - np.array(
                [[10], [10], [5], [np.pi * 1 / 3]]), np.array([[4], [-4]]),
                       np.array([[4], [-4], [4], [-4]])]  # np.array([[0], [-1], [2.5], [0]])
            parameters = dict()
            parameters['num_train_trajectories'] = 10
            parameters['num_train_samples_per_trajectories'] = 125
            parameters['number_of_constraint_points'] = 21
            parameters['num_approx_feasible_set'] = 1000
            parameters['early_stopping'] = 2
            parameters['learning_rate'] = 1e-2
            parameters['max_epoch'] = 1000
            parameters['train_split'] = 0.84
            parameters['batch_size'] = 40
            parameters['x0_list'] = x0_list
            parameters['N_embedded'] = 4
            parameters['max_process'] = 8
            parameters['system_list'] = ['systems.system_' + str(k) for k in range(parameters['N_embedded'])]
            initial_states = x0_list
            M_list = []
    ENV_NAME = 'gym_examples/Schedule-v0'
    env = gym.make(ENV_NAME,use_priority=False,use_both=True,use_true=use_true,max_sample=maxi,max_load=6,max_training=maxi,size_p=num_workers,sample_par=num_workers,training_par=num_workers,p=p,system_size=system_size)

    env = Monitor(env, log_dir)
    callback = SaveOnBestTrainingRewardCallback(check_freq=1_000, log_dir=log_dir)
    model = PPO("MlpPolicy", env,verbose=1,n_steps=10_000,batch_size=100)#,device=torch.device("cpu"))
    timesteps=10_000_000
    model.learn(total_timesteps=timesteps,callback=callback)
    #model.save("results/model/best_model.zip")
    plot_own_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "Scheduling for "+str(system_size)+" systems and "+str(maxi)+" sampling values")
    #plt.figure()
    plt.show(block=False)
    plt.savefig(fname="results/images/reward_"+str(maxi)+"_"+str(p)+"_"+str(system_size)+"_"+str(num_workers)+"_results.svg")
    tikzplotlib.save("results/images/reward_"+str(maxi)+"_"+str(p)+"_"+str(system_size)+"_"+str(num_workers)+"_results.tikz")
    plt.close()
    env.close()
def main():
    maximal=[1250]
    p_list=[0.1]#range(1,10)
    system_size=[16]
    num_workers=[25]
    for m in range(len(num_workers)):
        #for k in range(len(p_list)):
            #for i in range(len(maximal)):
            #    for j in range(len(system_size)):
                    train(maximal[m],p_list[m],system_size[m],num_workers[m])

if __name__ == '__main__':
    main()

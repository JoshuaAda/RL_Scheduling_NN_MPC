import sys
import os
from callback import SaveOnBestTrainingRewardCallback
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common import results_plotter
import tikzplotlib
from stable_baselines3.common.results_plotter import plot_results,load_results,ts2xy,window_func
from stable_baselines3.common.monitor import Monitor
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from typing import List, Optional, Tuple
EPISODES_WINDOW = 1000
sys.path.append('./gym-examples')
import gym_examples
import argparse

##### This is a modified version of the plot_results function from stable_baselines3.common.results_plotter
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

##### This is a modified version of the plot_results function from stable_baselines3.common.results_plotter
def plot_own_results(dirs: List[str], num_timesteps: Optional[int], x_axis: str, task_name: str, figsize: Tuple[int, int] = (8, 2)):
    data_frames = []
    for folder in dirs:
        data_frame = load_results(folder)
        if num_timesteps is not None:
            data_frame = data_frame[data_frame.l.cumsum() <= num_timesteps]
        data_frames.append(data_frame)
    xy_list = [ts2xy(data_frame, x_axis) for data_frame in data_frames]
    plot_curves(xy_list, x_axis, task_name, figsize)


##### This is the main training function for the PPO algorithm as well as the Safe RL
def train(args):
    ###### The following parameters can be set by the user:
    maxi=args.t_s_length
    p=args.failure_prob
    system_size=args.system_size
    num_workers=args.num_workers
    use_true=args.use_true
    max_load=args.max_load
    run_name=args.run_name
    image_dir=args.image_dir
    log_dir = "results/model_"+str(maxi)+"_"+str(p)+"_"+str(system_size)+"_"+str(num_workers)+"_results"+run_name+"/"
    num_iter=args.num_iter
    use_priority=args.use_priority
    use_both=args.use_both
    verbose=args.verbose
    n_steps=args.n_steps
    batch_size=args.batch_size
    check_freq=args.check_freq

    os.makedirs(log_dir, exist_ok=True)
    ###### Experimental: If you set use_true to True, the model will be trained with real simulated systems.
    ######  You will have to define your parameters for the scenario though, see evaluation.py for an example.
    #if use_true:
    #    parameters = dict() ...

    ##### Generating an environment and train the model. The callback is used to save the best model.
    ENV_NAME = 'gym_examples/Schedule-v0'
    env = gym.make(ENV_NAME,use_priority=use_priority,use_both=use_both,use_true=use_true,max_sample=maxi,max_load=max_load,max_training=maxi,size_p=num_workers,sample_par=num_workers,training_par=num_workers,p=p,system_size=system_size)
    env = Monitor(env, log_dir)
    callback = SaveOnBestTrainingRewardCallback(check_freq=check_freq, log_dir=log_dir)
    model = PPO("MlpPolicy", env,verbose=verbose,n_steps=n_steps,batch_size=batch_size)#,device=torch.device("cpu"))
    timesteps=num_iter
    model.learn(total_timesteps=timesteps,callback=callback)
    plot_own_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "Scheduling for "+str(system_size)+" systems and "+str(maxi)+" sampling values")
    plt.show(block=False)
    plt.savefig(fname=image_dir+"/reward_"+str(maxi)+"_"+str(p)+"_"+str(system_size)+"_"+str(num_workers)+"_results"+run_name+".svg")
    tikzplotlib.save(image_dir+"/reward_"+str(maxi)+"_"+str(p)+"_"+str(system_size)+"_"+str(num_workers)+"_results"+run_name+".tikz")
    plt.close()
    env.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_both', type=bool, default=True)
    parser.add_argument('--use_priority', type=bool, default=False)
    parser.add_argument('--use_true', type=bool, default=False)
    parser.add_argument('--t_s_length', type=int, default=1250)
    parser.add_argument('--max_load', type=int, default=6)
    parser.add_argument('--failure_prob', type=float, default=0.1)
    parser.add_argument('--system_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=256)
    parser.add_argument('--num_iter', type=int, default=10000000)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--n_steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--check_freq', type=int, default=1000)
    parser.add_argument('--run_name', type=str, default="SafeRL")
    parser.add_argument('--image_dir', type=str, default="results/images")
    args = parser.parse_args()
    train(args)

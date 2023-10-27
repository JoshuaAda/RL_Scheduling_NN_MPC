import sys
import os

import torch

from callback import SaveOnBestTrainingRewardCallback
import gym
import argparse
import numpy as np
from mpl_toolkits import mplot3d

import matplotlib.pyplot as plt
from stable_baselines3 import PPO
#from stable_baselines3.common import results_plotter
#from stable_baselines3.common.results_plotter import plot_results,load_results,ts2xy,window_func
#from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
from typing import Callable, List, Optional, Tuple
EPISODES_WINDOW = 100
sys.path.append('./gym-examples')
import gym_examples


def evaluate(p=5,maxi=16,system_size=4,num_workers=256,use_priority=False,use_both=True,use_true=False,num_iter=5000,run_first=False,use_mpc=False):
    if use_true:
        x0_list =[np.array([[4],[-4],[4],[-4]]) for k in range(16)]
        parameters = dict()
        parameters['num_train_trajectories'] = 255
        parameters['num_train_samples_per_trajectories'] = 5
        parameters['number_of_constraint_points'] = 21
        parameters['num_approx_feasible_set'] = 1000
        parameters['early_stopping'] = 3000
        parameters['learning_rate'] = 1e-2
        parameters['max_epoch'] = 1250
        parameters['train_split'] = 0.84
        parameters['batch_size'] = 50
        parameters['x0_list'] = x0_list
        parameters['N_embedded'] = system_size
        parameters['max_process'] = num_workers
        parameters['system_list'] = ['systems.system_' + str(k) for k in range(parameters['N_embedded'])]

    else:
        parameters=None
    #os.makedirs(log_dir, exist_ok=True)
    #use_priority=False
    #use_both=True
    ENV_NAME = 'gym_examples/Schedule-v0'
    env = gym.make(ENV_NAME,parameters=parameters,use_true=use_true,use_priority=use_priority,use_both=use_both,run_first=run_first,use_mpc=use_mpc,max_sample=maxi,max_load=6,max_training=maxi,size_p=num_workers,sample_par=num_workers,training_par=num_workers,p=p,system_size=system_size)
    if use_priority:
        model = PPO("MlpPolicy", env, verbose=1)#,device=torch.device("cpu"))
    else:
        dir = "results/model_"+str(maxi)+"_"+str(p)+"_"+str(system_size)+"_"+str(num_workers)+"_results11/best_model.zip"
        model = PPO.load(dir,env=env,device=torch.device("cpu"))#PPO("MlpPolicy", env, verbose=1,device=torch.device("cpu"))#PPO("MlpPolicy", env, verbose=1)

    vec_env = model.get_env()
    obs = vec_env.reset()
    overall_reward=0
    num_don=0
    reward_scenario = 0
    diff_reward=0
    num_don_diff=0
    policy_used=0
    for i in range(num_iter):
        action,_state=model.predict(obs, deterministic=True)#np.ones((4,)).tolist() #model.predict(obs, deterministic=False)
        #action=np.asarray([int(act) for act in action])
        obs, reward, done, info = vec_env.step(action)
        print(info)#['states'])
        reward_scenario += reward
        overall_reward += reward
        if use_both:
            policy_used+=info[0]['policy']
        #if done:
        #    print('done: reward'+str(reward_scenario))
        #    num_don+=1
        #    if reward[0]>-500:
        #        diff_reward+=reward_scenario
        #        num_don_diff+=1

        #    reward_scenario=0
        vec_env.render()
        # VecEnv resets automatically
        # if done:
        #   obs = env.reset()
        print(overall_reward)
    #print(diff_reward/num_don_diff)
    #print(num_don_diff/num_don)
    print(policy_used/num_iter)

    env.close()
    return overall_reward,policy_used/num_iter
def main():
    maximal = [1250]
    p_list = [0.1]  # range(1,10)
    system_size = [16]
    num_workers = [256]
    rewards_policy = np.zeros((len(maximal),1))# len(p_list)))
    pol_policy= np.zeros((len(maximal),1))# len(p_list)))
    rewards_reinforce = np.zeros((len(maximal),1))#, len(p_list)))
    pol_reinforce= np.zeros((len(maximal),1))# len(p_list)))
    for m in range(len(num_workers)):
        #for s in range(len(system_size)):
        #    for k in range(len(p_list)):
        #        for i in range(len(maximal)):
                    #mean_rew,pol =evaluate(use_both=False,use_priority=True,maxi=maximal[m],p=p_list[m],system_size=system_size[m],num_workers=num_workers[m],use_true=False)
                    #rewards_policy[m]=mean_rew
                    #pol_policy[m]=pol
                    mean_rew,pol = evaluate(use_both=False, use_priority=True, maxi=maximal[m], p=p_list[m],system_size=system_size[m],num_workers=num_workers[m],use_true=True,num_iter=5000,run_first=False,use_mpc=False)
                    pol_reinforce[m]=pol
                    rewards_reinforce[m] = mean_rew
    #print(rewards_policy)
    print(rewards_reinforce)
    print(pol_reinforce)
   # np.savez("results/result_reward.npz",policy=rewards_policy,reinforce=rewards_reinforce)
   # np.savez("results/result_pol.npz", policy=pol_policy, reinforce=pol_reinforce)
    #fig = plt.figure()
    #ax = plt.axes(projection='3d')
    #ax.plot_surface(np.array(p_list).reshape((len(p_list),1)), np.array(maximal).reshape((len(maximal),1)).T , rewards_policy.T, cmap='viridis', edgecolor='none')
    #ax.plot_surface(np.array(p_list).reshape((len(p_list), 1)), np.array(maximal).reshape((len(maximal), 1)).T,
    #                rewards_reinforce.T, cmap='viridis', edgecolor='none')
    #ax.set_title('Surface plot')
    #plt.show()
if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--name', type=str, required=True)
    #args = parser.parse_args()
    main()

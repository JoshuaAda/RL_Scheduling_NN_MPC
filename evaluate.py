import sys
import torch
import gym
import argparse
import numpy as np
from stable_baselines3 import PPO
EPISODES_WINDOW = 100
sys.path.append('./gym-examples')
import gym_examples

#### Evaluation file for the mockup scheduling problem as well as the simulated controlled systems
def evaluate(args):
    p = args.p
    maxi = args.t_s_length
    system_size = args.system_size
    num_workers = args.num_workers
    use_priority = args.use_priority
    use_both = args.use_both
    use_true = args.use_true
    run_name=args.run_name
    num_iter = args.num_iter
    run_first = args.run_first
    use_mpc = args.use_mpc
    max_load = args.max_load
    if use_true:
        x0_list =[np.array([[4],[-4],[4],[-4]]) for k in range(system_size)]
        parameters = dict()
        parameters['num_train_trajectories'] = num_workers
        parameters['num_train_samples_per_trajectories'] = 10
        parameters['number_of_constraint_points'] = 21
        parameters['num_approx_feasible_set'] = 1000
        parameters['early_stopping'] = 3000
        parameters['learning_rate'] = 1e-2
        parameters['max_epoch'] = maxi
        parameters['train_split'] = 0.84
        parameters['batch_size'] = 50
        parameters['x0_list'] = x0_list
        parameters['N_embedded'] = system_size
        parameters['max_process'] = num_workers
        parameters['system_list'] = ['systems.system_' + str(k) for k in range(parameters['N_embedded'])]

    else:
        parameters=None

    ENV_NAME = 'gym_examples/Schedule-v0'
    env = gym.make(ENV_NAME,parameters=parameters,use_true=use_true,use_priority=use_priority,use_both=use_both,run_first=run_first,use_mpc=use_mpc,max_sample=maxi,max_load=max_load,max_training=maxi,size_p=num_workers,sample_par=num_workers,training_par=num_workers,p=p,system_size=system_size)
    if use_priority:
        model = PPO("MlpPolicy", env, verbose=1)#,device=torch.device("cpu"))
    else:
        dir = "results/model_"+str(maxi)+"_"+str(p)+"_"+str(system_size)+"_"+str(num_workers)+"_results"+run_name+"/best_model.zip"
        model = PPO.load(dir,env=env,device=torch.device("cpu"))#PPO("MlpPolicy", env, verbose=1,device=torch.device("cpu"))#PPO("MlpPolicy", env, verbose=1)

    vec_env = model.get_env()
    obs = vec_env.reset()
    overall_reward=0
    num_don=0
    reward_scenario = 0
    policy_used=0
    for i in range(num_iter):
        action,_state=model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        print(info)
        reward_scenario += reward
        overall_reward += reward
        if use_both:
            policy_used+=info[0]['policy']
        if done:
            print('done: reward'+str(reward_scenario))
            num_don+=1
            reward_scenario=0
        # VecEnv resets automatically
    # I will leave it up to the user what to do with the results, so far they are only printed here
    print(overall_reward)
    print(overall_reward/num_don)
    print(policy_used/num_iter)
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_both', type=bool, default=True)
    parser.add_argument('--use_priority', type=bool, default=False)
    parser.add_argument('--use_mpc', type=bool, default=False)
    parser.add_argument('--use_true', type=bool, default=False)
    parser.add_argument('--run_name', type=str, default="SafeRL")
    parser.add_argument('--t_s_length', type=int, default=1250)
    parser.add_argument('--max_load', type=int, default=6)
    parser.add_argument('--failure_prob', type=float, default=0.1)
    parser.add_argument('--system_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=256)
    parser.add_argument('--num_iter', type=int, default=1000000)
    parser.add_argument('--run_first', type=bool, default=True)
    args = parser.parse_args()
    evaluate(args)

import random
import matplotlib.pyplot as plt
import gym
from gym import spaces
import pygame
import numpy as np
from cloud import cloud
import pickle
from multi_embedded import multi_embedded,embedded
from cloudMockup import cloudMockup,cloudMockupPriority,cloudMockupBest
import copy
from CloudProcess import CloudProcessMockup

class SchedulingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,use_true=False,use_both=False,use_priority=False, run_first=False,use_mpc=False,parameters=None, render_mode=None,size_p=8,M=3,max_sample=16,max_load=2,max_training=16,sample_par=8,training_par=8,p=5,system_size=4):
        self.size = system_size#4  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.num_sample_states=np.sum(np.asarray([np.ceil(max_sample / T) for T in range(1, sample_par+1)]))
        self.num_train_states = np.sum(np.asarray([np.ceil(max_training / T) for T in range(1, training_par+1)]))
        self.sum_states=2+3+M+max_load+self.num_sample_states+np.sum(np.asarray([np.ceil(max_training/T) for T in range(1,training_par+1)]))
        self.size_p=size_p
        self.M=M
        self.p=p
        self.max_sample=max_sample
        self.max_load=max_load
        self.max_training=max_training
        self.training_par=training_par
        self.sample_par=sample_par
        scenario=dict()
        scenario['system_size']=system_size
        scenario['size_p'] = size_p#+system_size
        scenario['M'] = M
        scenario['max_sampling'] = max_sample
        scenario['max_load'] = max_load
        scenario['max_training'] = max_training
        scenario['training_par'] = training_par
        scenario['sample_par'] = sample_par
        self._use_both=use_both
        self._use_true=use_true
        self._use_priority=use_priority
        if self._use_both and not self._use_true:
            self._cloud_prio = cloudMockupPriority(parameters,use_priority,scenario=scenario)#cloudMockupBest(parameters,use_priority,scenario=scenario) #
            self._cloud = cloudMockup(parameters,scenario=scenario)
        elif not self._use_true:
            if use_priority:
                self._cloud=cloudMockupPriority(parameters,scenario=scenario)#cloudMockupBest(parameters,use_priority,scenario=scenario)
            else:
                self._cloud=cloudMockup(parameters,scenario=scenario)
        if self._use_true:
            self.parameters=parameters
            self._cloud=cloud(parameters,scenario=scenario)
            self._cloud_model=cloudMockup(scenario=scenario)
            M_list=[]
            for k in range(parameters['N_embedded']):
                M_list.append(1)
            initial_states = [np.array([[4],[-4],[4],[-4]]) for k in range(self.size)]

            self._multi_embedded = multi_embedded(self._cloud.number_embedded, initial_states, self._cloud.G_list.copy(),
                                            self._cloud.mpc_list.copy(), M_list.copy(), self._cloud.linear_list.copy())

        self.observation_space = spaces.MultiDiscrete(np.repeat(self.sum_states,self.size).tolist())#np.append(,[100])#spaces.Box(0, self.sum_states, shape=(self.size,), dtype=int)#spaces.Dict(


        self.action_space = spaces.MultiDiscrete(np.repeat(self.size_p,self.size))#spaces.Box(0, max(training_par,sample_par), shape=(self.size,), dtype=int)
        self._num=0


        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode


        self.window = None
        self.clock = None
        self._current_state = [0 for k in range(self.size)]#+[self.p]
        self.failure_list=[]
        self._started=False
        self._time=0
        self._run_first=run_first
        self._use_mpc=use_mpc
        if self._run_first:
            self._changing_lists=[]
            self.x_tilde_lists=[]
        else:
            with open('changing6.pkl', 'rb') as f:
                self._changing_lists=pickle.load(f)
            with open('x_tilde6.pkl', 'rb') as f:
                self.x_tilde_lists=pickle.load(f)


    def _get_obs(self):
            return self._current_state

    @staticmethod
    def _waiting_functions(num):
        if num==1:
            return 'start_sampling'
        elif num==2:
            return 'load_data'
        else:
            return 'train_on_data'
    def _process_list_to_states(self,process_list,waiting_list,p):
        system_numbers_wait=[wait.get_system_number() for wait in waiting_list]
        system_numbers_process = [process.get_system_number() for process in process_list]
        states=np.zeros((self.size,1))#1
        new_list=process_list+waiting_list
        for k in range(self.size):
            if k not in system_numbers_process and k not in system_numbers_wait:
                states[k]=0
            #if k not in system_numbers_process and k in system_numbers_wait:
            #    index=system_numbers_wait.index(k)
            #    states[k]=self.M+1-waiting_list[index].deadline#[waiting_list[s].system_number for s in range(len(waiting_list))][[waiting_list[s].system_number for s in range(len(waiting_list))].index(k)].deadline+1
        sampling_list=[]
        for k,process in enumerate(waiting_list):
            if process.type()=='calculate_online_step':
                states[process.get_system_number()]=self.M+2-process.get_wait_duration()
                sampling_list.append(process.get_system_number())
        for k,process in enumerate(waiting_list):
            if process.type()=="start_sampling" and process.get_system_number() not in sampling_list:
                sampling_list.append(process.get_system_number())
                states[process.get_system_number()] = self.M + 3

        for k,process in enumerate(new_list):
            if process.get_running_time()==0:
                delay=1
            else:
                delay=process.get_running_time()
            if process.type() == 'start_sampling' and process.get_system_number() not in sampling_list:
                delay_list = [0]+np.cumsum(np.asarray([np.ceil(self.max_sample / T) for T in range(1,self.sample_par+1)])).tolist()
                #delay=process.get_running_time()
                par=process.get_num_workers()
                states[process.get_system_number()] = self.M+delay_list[par]-delay+3#self.num_sample_states + self.M + 1
            elif process.type() == 'load_data':
                states[process.get_system_number()] = self.M+self.num_sample_states+3+self.max_load-delay#self.num_sample_states + self.M + 2
            elif process.type() == 'train_on_data':
                delay_list = [0]+np.cumsum(np.asarray([np.ceil(self.max_training / T) for T in range(1,self.training_par+1)])).tolist()
                #delay = process.get_running_time()
                par = process.get_num_workers()
                states[process.get_system_number()] =  self.M+self.num_sample_states+3+self.max_load+delay_list[par] - delay
        #states[-1]=p
        return np.array(states).reshape((len(states,)))

    def _process_list_to_info(self,process_list,wait_list,p):
        process_list_without_c=[p for p in process_list if p.type()!="calculate_online_step"]
        systems_indices_process=[p.get_system_number() for p in process_list_without_c]
        systems_indices_wait_without_c=[p.get_system_number() for p in wait_list if p.type()!="calculate_online_step"]
        systems_indices_wait_c=[p.get_system_number() for p in wait_list if p.type()=="calculate_online_step"]
        systems_indices_wait=systems_indices_wait_c+systems_indices_wait_without_c
        system_types_wait=[p.type() for p in wait_list]
        system_types_process=[p.type() for p in process_list_without_c]
        system_c=[process.get_system_number() for process in process_list if process.get_system_number() not in systems_indices_process]
        systems_sampling_wait=[p.get_system_number() for p in wait_list if p.type()=="start_sampling" and p.get_num_workers() is None]
        system_csampling=[p for p in system_c if p in systems_sampling_wait]
        states=[0 for k in range(self.size)]
        for k in range(self.size):
            if k in system_csampling:
                states[k]="C"
                continue
            if k not in systems_indices_process:
                if k not in systems_indices_wait:
                    if self._current_state[k]==1:
                        states[k]="U"
                    else:
                        states[k]="H"
                else:
                    if system_types_wait[systems_indices_wait.index(k)]=="calculate_online_step":
                        states[k]="W"+str(wait_list[systems_indices_wait.index(k)].get_deadline())
                    elif system_types_wait[systems_indices_wait.index(k)]=="start_sampling":
                        states[k]="S"+str(int(wait_list[systems_indices_wait.index(k)].get_deadline()))+"/"+str(wait_list[systems_indices_wait.index(k)].get_num_workers())#"C"+str(1)
                    elif system_types_wait[systems_indices_wait.index(k)]=="load_data":
                        states[k]="L"+str(int(wait_list[systems_indices_wait.index(k)].get_deadline()))#"C"+str(2)
                    elif system_types_wait[systems_indices_wait.index(k)]=="train_on_data":
                        states[k]="T"+str(int(wait_list[systems_indices_wait.index(k)].get_deadline()))+"/"+str(wait_list[systems_indices_wait.index(k)].get_num_workers())#"C"+str(3)

            if k in systems_indices_process:
                if system_types_process[systems_indices_process.index(k)]=="start_sampling":
                    states[k]="S"+str(int(process_list_without_c[systems_indices_process.index(k)].get_deadline()))+"/"+str(process_list_without_c[systems_indices_process.index(k)].get_num_workers())
                elif system_types_process[systems_indices_process.index(k)]=="load_data":
                    states[k]="L"+str(int(process_list_without_c[systems_indices_process.index(k)].get_deadline()))
                elif system_types_process[systems_indices_process.index(k)]=="train_on_data":
                    states[k]="T"+str(int(process_list_without_c[systems_indices_process.index(k)].get_deadline()))+"/"+str(process_list_without_c[systems_indices_process.index(k)].get_num_workers())
        #states.append(p)
        return states
    def _reward_and_terminal(self,process_list,waiting_list,failure_list,started,finished_early):

        par=0

        if process_list!=[]:
            minimum_workers=min([process.get_num_workers() for process in process_list])
        else:
            minimum_workers=0
        nec_process=len([process for process in process_list if process.type()=="calculate_online_step"])
        for process in process_list:
            par+=process.get_num_workers()


        reward=-finished_early
        if not process_list and not waiting_list:
            if failure_list:
                return(reward-1,False,False)
            else:
                if started:
                    return (reward,True,False)
                else:
                    return (reward,False,False)
        elif par>self.size_p or minimum_workers>self.size_p-nec_process:
            return (reward-100,False,True)
        else:
            return (reward-1,False,False)
    def _get_info(self,process_list,waiting_list,p,policy=None):

            print_dict= {}
            print_dict['states']=self._process_list_to_info(process_list,waiting_list,p)
            if self._use_both:
                print_dict['policy']=policy
            return print_dict

    def reset(self, seed=None, options=None):

            self._reward_rl=0
            self._current_state=[0 for k in range(self.size)]#+[self.p]
            self._cloud.p_list=[]
            self._cloud.waiting_list=[]
            if self._use_both and not self._use_true:
                self._cloud_prio.p_list = []
                self._cloud_prio.waiting_list = []
            if self._use_true:
                #self._cloud_prio.p_list = []
                #self._cloud_prio.waiting_list = []
                self._cloud_model.p_list = []
                self._cloud_model.waiting_list = []
                M_list = []
                for k in range(self.parameters['N_embedded']):
                    M_list.append(1)
                initial_states = [np.array([[4],[-4],[4],[-4]]) for k in range(self.size)]
            self.failure_list=[]
            self._started=False
            self._time=0
            observation = self._get_obs()
            info = self._get_info([],[],10)

            if self.render_mode == "human":
                self._render_frame()

            return observation

    def step(self, action):

            state_list=[0 for k in range(self.size)]
            input_list=[0 for k in range(self.size)]
            failure_pred_list=[self.M for k in range(len(self.failure_list))]

            p=self.p

            weights = [100 - p, p]

            if self._use_true:
                if self._use_mpc:
                    if not self._run_first:
                        changing_list=self._changing_lists[0]
                        x_tilde=self.x_tilde_lists[0]
                        self._changing_lists.pop(0)
                        self.x_tilde_lists.pop(0)
                    self.failure_list=changing_list
                    templates = self._cloud.change_template(changing_list, x_tilde_new=x_tilde)
                    observation = self._get_obs()
                    noise = [0.0 * np.random.randn(self._cloud.mpc_list[k].model.n_x, 1) for k in
                             range(self.parameters['N_embedded'])]

                    u0 = self._cloud.calculate_online_steps(all_mpc=True,x0_list=self._multi_embedded.get_all_states(),u0_pred_list=self._multi_embedded.get_all_inputs())

                    self._multi_embedded.get_next_step_online(u0, noise, self._cloud.online_list.copy())
                    if templates is not None:
                        self._multi_embedded.update_system_parameters(templates, changing_list)
                    self._multi_embedded.visualize_system()
                    plt.pause(0.1)
                    info = self._get_info([], [], p, 0)
                    reward=-1
                    terminated=False
                else:
                    if self._use_both:
                        policy = 1
                        self._cloud_model.wait_list = [p.copy_to_cloudmockup() for p in
                                         self._cloud.waiting_list]  # copy.deepcopy(self._cloud.waiting_list)
                        self._cloud_model.p_list = [p.copy_to_cloudmockup() for p in self._cloud.p_list]
                        old_failure_list = self.failure_list.copy()
                        new_process_list, new_wait_list, finished_early = self._cloud_model.schedule_processes(action,
                                                                                                     self.failure_list,
                                                                                                     state_list,
                                                                                                     failure_pred_list,
                                                                                                     input_list)

                        self._cloud_model.start_processes()
                        reward, terminated,bounds = self._reward_and_terminal(new_process_list, new_wait_list, self.failure_list,
                                                                   self._started,
                                                                   finished_early)  # np.array_equal(self._agent_location, self._target_location)
                    if self._use_both:
                        self._reward_rl += reward
                        if bounds:  # or self._reward_rl<-100
                            policy = 0

                            new_process_list, new_wait_list, finished_early, finished_list = self._cloud.schedule_processes_priority(
                                action,
                                self.failure_list.copy(),
                                self._multi_embedded.get_all_states(),
                                failure_pred_list,
                                self._multi_embedded.get_all_inputs())
                        else:
                            new_process_list, new_wait_list, finished_early, finished_list = self._cloud.schedule_processes(
                                action,
                                self.failure_list.copy(),
                                self._multi_embedded.get_all_states(),
                                failure_pred_list,
                                self._multi_embedded.get_all_inputs())
                    elif self._use_priority:
                        new_process_list, new_wait_list, finished_early, finished_list = self._cloud.schedule_processes_priority(
                            action, self.failure_list.copy(), self._multi_embedded.get_all_states(), failure_pred_list,
                            self._multi_embedded.get_all_inputs())
                    else:
                        new_process_list, new_wait_list, finished_early, finished_list = self._cloud.schedule_processes(
                            action,
                            self.failure_list.copy(),
                            self._multi_embedded.get_all_states(),
                            failure_pred_list,
                            self._multi_embedded.get_all_inputs())
                    self._cloud.start_processes()
                    plt.pause(0.1)
                    self._current_state = self._process_list_to_states(new_process_list, new_wait_list, p)
                    reward, terminated,bounds = self._reward_and_terminal(new_process_list, new_wait_list,
                                                                   self.failure_list,
                                                                   self._started, finished_early)
                    healthy_states = [k for k in range(self.size) if self._current_state[k] == 0]
                    failure, self.failure_list, failure_pred_list = self._multi_embedded.failure()

                    self._current_state = [1 if k in self.failure_list and k not in self._cloud.online_list else val for k, val in
                                           enumerate(self._current_state)]
                    healthy_states = [k for k in range(self.size) if self._current_state[k] == 0]
                    failure_states = random.choices([0, 1], weights=weights, k=len(healthy_states))
                    changing_list=[healthy_states[k] for k, x in enumerate(failure_states) if x == 1]
                    if not self._run_first:
                        changing_list=self._changing_lists[0]
                        x_tilde=self.x_tilde_lists[0]
                        self._changing_lists.pop(0)
                        self.x_tilde_lists.pop(0)
                    num=len(changing_list)
                    self._num+=num
                    self.failure_list = self.failure_list+changing_list#[k] for k in range(len(self.failure_list))]#[healthy_states[k] for k, x in enumerate(failure_states) if x == 1]

                    if self._run_first:
                        x_tilde = np.random.uniform(low=-1.8, high=1.8, size=(4, 1))
                        self._changing_lists.append(changing_list)

                        self.x_tilde_lists.append(x_tilde)
                        with open('changing6.pkl', 'wb') as f:
                            pickle.dump(self._changing_lists, f)
                        with open('x_tilde6.pkl', 'wb') as f:
                            pickle.dump(self.x_tilde_lists, f)
                    templates=self._cloud.change_template(changing_list,x_tilde_new=x_tilde)
                    self._current_state = [1 if k in self.failure_list else val for k, val in
                                           enumerate(self._current_state)]
                    if 1 in self._current_state and not self._started:
                        self._started = True

                    observation = self._get_obs()
                    noise = [0.0 * np.random.randn(self._cloud.mpc_list[k].model.n_x, 1) for k in range(self.parameters['N_embedded'])]
                    u0 = self._cloud.calculate_online_steps()
                    self._multi_embedded.update_network(finished_list)
                    self._multi_embedded.get_next_step_online(u0, noise, self._cloud.online_list.copy())
                    if templates is not None:
                        self._multi_embedded.update_system_parameters(templates,changing_list)
                    self._multi_embedded.visualize_system()
                    if self._use_both:
                        info = self._get_info(new_process_list, new_wait_list, p, policy)
                    else:
                        info = self._get_info(new_process_list, new_wait_list, p)
            else:
                if self._use_both:
                    policy=1
                    old_wait_list=[p.copy() for p in self._cloud.waiting_list]#copy.deepcopy(self._cloud.waiting_list)
                    old_process_list=[p.copy() for p in self._cloud.p_list]
                    old_failure_list=self.failure_list.copy()
                new_process_list,new_wait_list,finished_early=self._cloud.schedule_processes(action,self.failure_list,state_list,failure_pred_list,input_list)

                self._cloud.start_processes()
                self._current_state=self._process_list_to_states(new_process_list,new_wait_list,p)
                healthy_states = [k for k in range(self.size) if self._current_state[k] == 0]

                failure_states = random.choices([0, 1], weights=weights, k=len(healthy_states))
                self.failure_list = [healthy_states[k] for k, x in enumerate(failure_states) if x == 1]

                if not self._run_first:
                    self.failure_list = self._changing_lists[0]
                    self._changing_lists.pop(0)
                if self._run_first:
                    x_tilde = np.random.uniform(low=-1.8, high=1.8, size=(4, 1))
                    self._changing_lists.append(self.failure_list.copy())
                    self.x_tilde_lists.append(x_tilde)

                    with open('changing6.pkl', 'wb') as f:
                        pickle.dump(self._changing_lists, f)
                    with open('x_tilde6.pkl', 'wb') as f:
                        pickle.dump(self.x_tilde_lists, f)
                self._current_state=[1 if k in self.failure_list else val for k,val in enumerate(self._current_state)]
                if 1 in self._current_state[0:-1] and not self._started:
                    self._started=True
                # An episode is done iff the agent has reached the target
                reward,terminated,bounds = self._reward_and_terminal(new_process_list,new_wait_list,self.failure_list,self._started,finished_early)#np.array_equal(self._agent_location, self._target_location)
                if self._use_both:
                    self._reward_rl+=reward
                    if bounds:
                        policy=0
                        self._cloud_prio.p_list=[p.copy() for p in old_process_list]
                        self._cloud_prio.waiting_list = [p.copy() for p in old_wait_list]
                        new_process_list, new_wait_list, finished_early =self._cloud_prio.schedule_processes(action, old_failure_list, state_list, failure_pred_list, input_list)
                        self._cloud_prio.start_processes()
                        self._current_state = self._process_list_to_states(new_process_list, new_wait_list, p)
                        reward, terminated,bounds = self._reward_and_terminal(new_process_list, new_wait_list, self.failure_list,
                                                                       self._started, finished_early)
                        self._cloud.p_list=new_process_list.copy()
                        self._cloud.waiting_list=new_wait_list.copy()
                observation = self._get_obs()
                if self._use_both:
                    info = self._get_info(new_process_list,new_wait_list,p,policy)
                else:
                    info = self._get_info(new_process_list, new_wait_list, p)
                if self._started:
                    self._time+=1

            return observation, reward, terminated, info


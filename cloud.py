import importlib
import matplotlib.pyplot as plt
import numpy as np
import do_mpc
import pickle
import time
import os
import torch
torch.set_default_dtype(torch.float32)
from torch.utils.data import DataLoader, random_split
from dataset_class import CustomMPCDataset
import torch.nn as nn
import torch.optim as optim
from casadi import *

from model_class import NeuralNetwork
from multiprocessing import Process,Queue
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
from CloudProcess import CloudProcess

class cloud:
    """
    Class that represents the cloud actions
    """

    def __init__(self,parameters=None,scenario=None):#
        if parameters is not None:
            if scenario:
                self.max_sampling = scenario['max_sampling']
                self.max_training = scenario['max_training']
                self.max_load = scenario['max_load']
                self.max_process = scenario['size_p']
            else:
                self.max_sampling = 16
                self.max_training = 16
                self.max_process = 8
                self.max_load = 2
            # Initiate the training and general mpc as well as the calculation of the feasible set
            #model = template_model()
            #mpc = template_mpc(model)
            #simulator= template_simulator(model)
            #estimator=do_mpc.estimator.StateFeedback(model)


            self.max_process = parameters['max_process']  # multiprocessing.cpu_count()
            self.num_train_trajectories = parameters['num_train_trajectories']
            self.num_train_samples_per_trajectories = parameters['num_train_samples_per_trajectories']
            self.num_approx_feasible_set = parameters['num_approx_feasible_set']
            self.number_of_constraint_points=parameters['number_of_constraint_points']
            self.early_stopping = parameters['early_stopping']
            self.learning_rate = parameters['learning_rate']
            self.max_epoch = parameters['max_epoch']
            self.train_split = parameters['train_split']
            self.batch_size = parameters['batch_size']
            self.number_embedded = parameters['N_embedded']
            self.current_time = 0
            self.p_list = []
            self.waiting_list=[]
            #self.fig, self.ax = plt.subplots()
            #allowed_tasks=["calculate_online_step","start_sampling","train_on_data","load_data"]
            #self.patches=[]
            #for t in allowed_tasks:
            #    self.patches.append(Patch(color=self.color_code(t),label=t))
            #box = self.ax.get_position()
            #self.ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            #self.ax.legend(handles=self.patches, bbox_to_anchor=(1, 0.5))
            self.mpc_list=[]
            self.simulator_list=[]
            self.estimator_list=[]
            self.current_net_list=[]
            self.linear_list=[]
            #self.ax.grid()
            for k in range(self.number_embedded):
                path=parameters['system_list'][k]
                sys.path.append('path')
                model_path = path+ '.template_model'
                mpc_path = path+ '.template_mpc'
                simulator_path = path+'.template_simulator'
                template_model=importlib.import_module(model_path)
                template_mpc = importlib.import_module(mpc_path)
                template_simulator = importlib.import_module(simulator_path)
                if k>0:
                    importlib.reload(template_model)
                    importlib.reload(template_mpc)
                    importlib.reload(template_simulator)
                model=template_model.template_model()
                mpc=template_mpc.template_mpc(model)
                linear=len(mpc.tvp_fun(0)['_tvp', -1].full())>2
                simulator=template_simulator.template_simulator(model)
                states=mpc.model.n_x
                inputs=mpc.model.n_u

                self.mpc_list.append(mpc)#[template_mpc(model) for k in range(self.number_embedded)]
                self.simulator_list.append(simulator)# [template_simulator(model) for k in range(self.number_embedded)]
                self.estimator_list.append(do_mpc.estimator.StateFeedback(model))# [estimator for k in range(self.number_embedded)]
                PATH = './nets_start/' + str(k) + '/model_net.pt'
                net = NeuralNetwork(states+inputs, inputs)
                if not os.path.exists('./nets_start/' + str(states)):
                    os.mkdir('./nets_start/' + str(states))
                    # net = NeuralNetwork(nx + nu, nu)
                else:
                    net.load_state_dict(torch.load(PATH))
                self.current_net_list.append(NeuralNetwork(states+inputs,inputs)) #for k in range(self.number_embedded))
                self.linear_list.append(linear)

            self.sp_list=[]#do_mpc.sampling.SamplingPlanner()
            self.G_list=[np.array([[0.01858,0.00819],[0.00819,0.01858]]) for k in range(self.number_embedded)]#[self.calculate_feasible_set(k) for k in range(self.number_embedded)]#[np.array([[0.01793,-0.00235,-0.00416,-0.00236],
#[-0.00235,0.01445,0.00216,0.00268],
#[-0.00416,0.00216,0.01999,0.00049],
#[-0.00236,0.00268,0.00049,0.01449]]
#)for k in range(self.number_embedded)]#[self.calculate_feasible_set(k) for k in range(self.number_embedded)]#[np.array([[0.01867646, -0.00366424, -0.00396527, -0.0023837 ], [-0.00366424,  0.01109685, -0.00064795,  0.00272204], [-0.00396527, -0.00064795,  0.01427892, -0.00242744], [-0.0023837 ,  0.00272204, -0.00242744 , 0.00926212]]) for k in range(self.number_embedded)]
            self.online_list=[]
            self.parameters=parameters
            #for k in range(self.number_embedded):
            #    self.calculate_feasible_set(k)



    def copy(self):
        Cloud=cloud()

        #model = template_model()
        #mpc = template_mpc(model)
        #simulator = template_simulator(model)
        #estimator = do_mpc.estimator.StateFeedback(model)

        Cloud.max_process = self.max_process  # multiprocessing.cpu_count()
        Cloud.num_train_trajectories=self.num_train_trajectories
        Cloud.num_train_samples_per_trajectories=self.num_train_samples_per_trajectories
        Cloud.num_approx_feasible_set=self.num_approx_feasible_set
        Cloud.number_of_constraint_points=self.number_of_constraint_points
        Cloud.early_stopping=self.early_stopping
        Cloud.learning_rate=self.learning_rate
        Cloud.max_epoch=self.max_epoch
        Cloud.train_split=self.train_split
        Cloud.batch_size=self.batch_size
        Cloud.number_embedded=self.number_embedded
        Cloud.current_time = self.current_time
        Cloud.p_list = self.p_list.copy()
        Cloud.waiting_list = self.waiting_list.copy()
        #Cloud.fig, Cloud.ax = plt.subplots()
        #allowed_tasks = ["calculate_online_step", "start_sampling", "train_on_data", "load_data"]
        #Cloud.patches = []
        #for t in allowed_tasks:
        #    Cloud.patches.append(Patch(color=self.color_code(t), label=t))
        #box = Cloud.ax.get_position()
        #Cloud.ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        #Cloud.ax.legend(handles=self.patches, bbox_to_anchor=(1, 0.5))

        Cloud.mpc_list = self.mpc_list.copy()#[template_mpc(model) for k in range(self.number_embedded)]
        Cloud.simulator_list = self.simulator_list.copy()#[template_simulator(model) for k in range(self.number_embedded)]
        Cloud.estimator_list = self.estimator_list.copy()#[estimator for k in range(self.number_embedded)]
        Cloud.current_net_list = self.current_net_list.copy()#[NeuralNetwork() for k in range(self.number_embedded)]
        self.sp_list = []  # do_mpc.sampling.SamplingPlanner()
        Cloud.G_list = self.G_list.copy()
        Cloud.online_list = self.online_list.copy()
        return Cloud

    def calculate_feasible_set(self,sys):
        # generate an approximate feasible set

        mpc=self.mpc_list[sys]
        estimator=self.estimator_list[sys]
        simulator=self.simulator_list[sys]
        G = np.zeros(mpc.model.n_x)
        def check_feasible_set():
            # Generates random x0 in the states spaces and check for feasibility
            x0 = np.random.uniform(mpc.bounds['lower', '_x', 'x'].full(),mpc.bounds['upper', '_x', 'x'].full())
            mpc.reset_history()
            mpc.t0 = self.current_time
            simulator.reset_history()
            estimator.reset_history()

            # set initial values and guess

            mpc.x0 = x0
            simulator.x0 = x0
            estimator.x0 = x0
            mpc.set_initial_guess()

            # run the closed loop for system_1 step
            for k in range(1):
                u0 = mpc.make_step(x0)
                y_next = simulator.make_step(u0)
                if mpc.data['success'][-1] == 0:
                    return (x0,False)

            return (x0,True)
        valid_set=[]
        invalid_set=[]
        np.random.seed(1)
        for k in range(self.num_approx_feasible_set):
            value,valid=check_feasible_set()
            if valid:
                valid_set.append(DM(value))
            else:
                invalid_set.append(DM(value))
        # define G_head to find a symmetric G=G_head@G_head.T
        p=SX.sym("p",int((mpc.model.n_x)*((mpc.model.n_x)+1)/2),1)
        G_head=SX.sym('matrix', (mpc.model.n_x), (mpc.model.n_x))
        constraint=np.zeros((int((mpc.model.n_x)*((mpc.model.n_x)+1)/2),1))
        constraint.fill(-inf)
        s=0
        for m in range(G_head.size()[0]):
            for j in range(G_head.size()[1]):
                if j>m:
                    G_head[m, j] = 0
                else:
                    if m==j:
                        constraint[s]=0
                    G_head[m, j] = p[s]
                    s += 1
        G=G_head@G_head.T
        f=trace(G)

        def generate_corner_points():
            # Generate a number of corner points of the state space
            vectors=[np.array([[mpc.bounds['lower', '_x', 'x'].full()[k]],[mpc.bounds['upper', '_x', 'x'].full()[k]]]) for k in range(len(mpc.bounds['lower', '_x', 'x'].full()))]
            points=np.array(np.meshgrid(*vectors)).reshape(len(mpc.bounds['lower', '_x', 'x'].full()),-1)
            points=points.T
            list = []
            state_bounds = np.array(
                [[mpc.bounds['lower', '_x', 'x'].full()], [mpc.bounds['upper', '_x', 'x'].full()]]).squeeze()
            for k in range(np.shape(state_bounds)[1]):
                p=np.linspace(state_bounds[0,k],state_bounds[1,k],self.number_of_constraint_points)
                for m in range(len(p)):
                    point_left=state_bounds[0].copy()
                    point_left[k]=p[m]
                    point_right = state_bounds[1].copy()
                    point_right[k] = p[m]
                    list.append(DM(point_left))
                    list.append(DM(point_right))
            return list

        # add corner points of state constraints to the infeasible set
        corner_points=generate_corner_points()
        invalid_set+=corner_points
        list = [x.T @ G @ x for x in invalid_set]
        g = SX.sym("g", len(list), 1)
        for k, element in enumerate(list):
            g[k] = element

        # setup the solver
        prob = {'x': p, 'f': f, 'g': g}
        solver = nlpsol('solver', 'ipopt', prob)
        if self.linear_list[sys]:
            d = solver(x0=[k for k in range(p.size()[0])], lbx=constraint, lbg=1.5)
        else:
            d=solver(x0=[k for k in range(p.size()[0])],lbx=constraint,lbg=1.5)
        p=d['x'].full()
        G_head=np.zeros(((mpc.model.n_x),(mpc.model.n_x)))
        s = 0
        for m in range(len(G_head)):
            for j in range(len(G_head)):
                if j > m:
                    G_head[m, j] = 0
                else:
                    G_head[m, j] = p[s]
                    s += 1
        G = G_head @ G_head.T
        return G#.G_list.append(G)
        # Possibility to plot G especially if n_x=system_0
        #if sys==0:
        #    mtx=np.zeros((2,2))
        #    mtx[0,0]=G[0,0]
        #    mtx[0,1]=G[0,1]
        #    mtx[1,0]=G[1,0]
        #    mtx[1,1]=G[1,1]
        #    eigenvalues, eigenvectors = np.linalg.eig(mtx)
        #    theta = np.linspace(0, 2 * np.pi, self.num_approx_feasible_set)
        #    ellipsis = (1/np.sqrt(eigenvalues[None, :]) * eigenvectors) @ [np.sin(theta), np.cos(theta)]
        #    plot_xy=[invalid[0:2] for invalid in invalid_set]
        #    plt.scatter(np.array(plot_xy)[:,0],np.array(plot_xy)[:,1])
        #    plt.plot(ellipsis[0, :], ellipsis[1, :])
        #    plt.show()


    def change_template(self, system_number_list,A_new=None,F_new=None,B_new=None,Q_new=None,x_tilde_new=None,x_set_new=None,y_set_new=None):
            for system_number in system_number_list:
                print('Change system '+str(system_number))
                if self.linear_list[system_number]:
                    # Change of the template just as the user (operating the cloud) wants it
                    tvp_template = self.mpc_list[system_number].tvp_fun(0)

                    def tvp_fun(t_curr):
                        x_tilde=tvp_template['_tvp',-1, 'x_tilde']
                        A=tvp_template['_tvp', -1, 'A']
                        F = tvp_template['_tvp', -1, 'F']
                        B=tvp_template['_tvp', -1, 'B']
                        Q=tvp_template['_tvp', -1, 'Q']
                        # tvp_template['_tvp', k, 'R'] = R
                        K=tvp_template['_tvp', -1, 'K']
                        P=tvp_template['_tvp', -1, 'P']

                        for s in range(self.mpc_list[system_number].n_horizon+1):
                            for k in range(self.mpc_list[system_number].n_horizon+1 ):
                                tvp_template['_tvp', k, 'x_tilde'] =x_tilde if x_tilde_new is None else DM(x_tilde_new) #np.array([[system_1], [system_0], [system_0], [system_0]])##system_0.02*(s+t_curr)#system_0.05*t_curr
                                tvp_template['_tvp', k, 'A'] = A if A_new is None else A_new
                                tvp_template['_tvp', k, 'F'] = F if F_new is None else F_new
                                tvp_template['_tvp', k, 'B'] = B if B_new is None else B_new
                                tvp_template['_tvp', k, 'Q'] = Q if Q_new is None else Q_new
                                # tvp_template['_tvp', k, 'R'] = R
                                tvp_template['_tvp', k, 'K'] = K.T
                                tvp_template['_tvp', k, 'P'] = P
                        #self.template = tvp_template
                        return tvp_template
                    self.mpc_list[system_number].set_tvp_fun(tvp_fun)

                else:
                    tvp_template = self.mpc_list[system_number].tvp_fun(0)
                    def tvp_fun(t_curr):
                        x_set=tvp_template['_tvp',-1, 'x_set']
                        y_set = tvp_template['_tvp', -1, 'y_set']
                        for s in range(self.mpc_list[system_number].n_horizon + 1):
                            for k in range(self.mpc_list[system_number].n_horizon + 1):
                                tvp_template['_tvp', k, 'x_set'] = x_set if x_set_new is None else x_set_new
                                tvp_template['_tvp', k, 'y_set'] = y_set if y_set_new is None else y_set_new
                        return tvp_template

                    self.mpc_list[system_number].set_tvp_fun(tvp_fun)

            return [self.mpc_list[system_number].tvp_fun(0) for system_number in system_number_list]
    def schedule_processes(self,actions,failure_list,state_list,failure_pred_list,input_list):
        finished_early = 0
        #actions=[2 for action in actions]
        #types=[p.type() for p in self.p_list]
        #deadlines=[p.get_deadline() for p in self.p_list]
        #if "load_data" in types and 1 in deadlines:
        #    actions[14]=2#[43 for action in actions]
        waiting_list = self.waiting_list.copy()
        for p in self.p_list:
            #try:
            p.reduce_duration()
            #except:
            #    print(p.type())
            #    print(p.get_num_workers())
            #    print("p_list")
            #    print("rl")
            #    sys.exit(1)
        for p in self.waiting_list:
            # if p.type()=="calculate_online_step":
            #try:
                p.reduce_duration()
            #except:
            #    print(p.type())
            #    print(p.get_num_workers())
            #    print("wait_list")
            #    print("rl")
            #    sys.exit(1)
        finished_list = []
        flag_indices = [p.get_system_number() for p in waiting_list if p.type() == "calculate_online_step"]

        next_projected_processes_list = self.p_list.copy()  # +self.waiting_list.copy()

        # for p in self.waiting_list:
        #    p.update_wait_duration()
        #    if p.check_if_necessary():
        #        self.waiting_list.remove(p)
        #        necessary_processes_list.append(p)

        for p in self.waiting_list:
            # if p.type()=="start_sampling" and p.get_deadline()==0:
            #    print('Hi')
            # print(p.get_deadline())
            if p.type() == "calculate_online_step":

                if p.is_alive():
                    if actions[p.get_system_number()] > 0:
                        waiting_list.remove(p)
                        next_projected_processes_list.append(
                            CloudProcess(function=self.calculate_online_step, system_number=p.get_system_number(),
                                               num_workers=1, deadline=10000, x0=state_list[p.get_system_number()],
                                               u0=input_list[p.get_system_number()], q=Queue()))
                        # for q in waiting_list:
                        #    if q.get_system_number()==p.get_system_number() and q.type()=="start_sampling":
                        # q.kill()
                        # deadline=q.get_deadline()
                        # num_workers=q.get_num_workers()
                        # done=(np.floor(self.max_sampling/num_workers)-deadline)*num_workers
                        #        waiting_list.remove(q)
                        #        next_projected_processes_list.append(CloudProcessMockup(function=self.start_sampling,system_number=p.get_system_number(),num_workers=actions[p.get_system_number()],deadline=np.floor((self.max_sampling)/actions[p.get_system_number()])))
                    # else:
                    #    pass
                else:
                    # if actions[p.get_system_number()]>0:
                    waiting_list.remove(p)
                    next_projected_processes_list.append(
                        CloudProcess(function=self.calculate_online_step, system_number=p.get_system_number(),
                                           num_workers=1, deadline=10000, x0=state_list[p.get_system_number()],
                                           u0=input_list[p.get_system_number()], q=Queue()))
                    # for q in waiting_list:
                    #    if q.get_system_number() == p.get_system_number() and q.type() == "start_sampling":
                    # q.kill()
                    # deadline = q.get_deadline()
                    # num_workers = q.get_num_workers()
                    # done = (np.floor(self.max_sampling / num_workers) - deadline) * num_workers
                    #        waiting_list.remove(q)
                    #        next_projected_processes_list.append(
                    #            CloudProcessMockup(function=self.start_sampling, system_number=p.get_system_number(),
                    #                               num_workers=actions[p.get_system_number()],
                    #                               deadline=np.floor((self.max_sampling)/ actions[p.get_system_number()])))
                    # else:
                    #    waiting_list.remove(p)
                    #    next_projected_processes_list.append(
                    #        CloudProcessMockup(function=self.calculate_online_step, system_number=p.get_system_number(),
                    #                           num_workers=1, deadline=10000, x0=state_list[p.get_system_number()],
                    #                           u0=input_list[p.get_system_number()], q=Queue()))

                    # for q in waiting_list:
                    #    if q.get_system_number() == p.get_system_number() and q.type() == "start_sampling":
                    # q.kill()

                    #        waiting_list.remove(q)
                    #        next_projected_processes_list.append(
                    #            CloudProcessMockup(function=self.start_sampling,
                    #                               system_number=p.get_system_number(),
                    #                               num_workers=1,
                    #                               deadline=self.max_sampling))


            else:
                if actions[p.get_system_number()] > 0:
                    if p.type() == "train_on_data":
                        deadline = p.get_deadline()
                        num_workers = p.get_num_workers()
                        # done = (np.floor(self.max_sampling / num_workers) - deadline) * num_workers
                        waiting_list.remove(p)
                        next_projected_processes_list.append(
                            CloudProcess(function=self.train_on_data, system_number=p.get_system_number(),
                                               num_workers=p.get_num_workers(),  # actions[p.get_system_number()],
                                               deadline=p.get_deadline(),startup_function=self.train_on_data_startup))  # np.floor((self.max_training -done)/ actions[p.get_system_number()])))
                    elif p.type() == "load_data":
                        if p.is_alive():
                            waiting_list.remove(p)
                            next_projected_processes_list.append(
                                CloudProcess(function=self.load_data, system_number=p.get_system_number(),
                                                   deadline=p.get_deadline(),
                                                   num_workers=1))
                        else:
                            waiting_list.remove(p)
                            next_projected_processes_list.append(
                                CloudProcess(function=self.train_on_data, system_number=p.get_system_number(),
                                                   deadline=np.ceil(
                                                       self.max_training / actions[p.get_system_number()]),
                                                   num_workers=actions[p.get_system_number()],startup_function=self.train_on_data_startup))
                    elif p.type() == "start_sampling" and p.get_system_number() not in flag_indices:
                        if p.get_num_workers() is not None:
                            if not p.is_alive():
                                waiting_list.remove(p)
                                next_projected_processes_list.append(
                                    CloudProcess(function=self.load_data, system_number=p.get_system_number(),
                                                       deadline=self.max_load, num_workers=1))
                            else:
                                deadline = p.get_deadline()
                                num_workers = p.get_num_workers()
                                # done = (np.floor(self.max_sampling / num_workers) - deadline) * num_workers
                                waiting_list.remove(p)
                                next_projected_processes_list.append(CloudProcess(function=self.start_sampling,
                                                                                        system_number=p.get_system_number(),
                                                                                        num_workers=p.get_num_workers(),
                                                                                        # actions[p.get_system_number()],
                                                                                        deadline=p.get_deadline(),startup_function=self.start_sampling_startup))  # np.floor((self.max_sampling-done)/ actions[p.get_system_number()])))
                        else:
                            waiting_list.remove(p)
                            next_projected_processes_list.append(
                                CloudProcess(function=self.start_sampling, system_number=p.get_system_number(),
                                                   num_workers=actions[p.get_system_number()],
                                                   deadline=np.ceil(
                                                       self.max_sampling / actions[p.get_system_number()]),startup_function=self.start_sampling_startup))

                else:

                    if p.type() == "train_on_data":
                        deadline = p.get_deadline()
                        num_workers = p.get_num_workers()
                        finished_early += deadline
                        finished_list.append(p.get_system_number())
                        p.kill()
                        for q in next_projected_processes_list:
                            if q.get_system_number() == p.get_system_number() and q.type() == "calculate_online_step":
                                q.kill()
                                next_projected_processes_list.remove(q)
                        next_projected_processes_list.remove(p)

                        ##done = (np.floor(self.max_sampling / num_workers) - deadline) * num_workers
                        # waiting_list.remove(p)
                        # waiting_list.append(
                        #    CloudProcessMockup(function=self.train_on_data, system_number=p.get_system_number(),
                        #                       num_workers=num_workers,
                        #                       deadline=deadline))
                    elif p.type() == "load_data":
                        waiting_list.remove(p)
                        waiting_list.append(
                            CloudProcess(function=self.load_data, system_number=p.get_system_number(),
                                               deadline=p.get_deadline(),
                                               num_workers=1))
                    elif p.type() == "start_sampling" and p.get_system_number() not in flag_indices:
                        deadline = p.get_deadline()
                        num_workers = p.get_num_workers()
                        # done = (np.floor(self.max_sampling / num_workers) - deadline) * num_workers
                        waiting_list.remove(p)
                        waiting_list.append(
                            CloudProcess(function=self.start_sampling, system_number=p.get_system_number(),
                                               num_workers=num_workers,
                                               deadline=deadline,startup_function=self.start_sampling_startup))

        for p in self.p_list:
            if p.type() == "train_on_data" and not p.is_alive() or p.type() == "train_on_data" and actions[
                p.get_system_number()] == 0:
                if p.type() == "train_on_data" and actions[p.get_system_number()] == 0 and p.is_alive():
                    finished_early += p.get_deadline()
                finished_list.append(p.get_system_number())
                p.kill()
                for q in next_projected_processes_list:
                    if q.get_system_number() == p.get_system_number() and q.type() == "calculate_online_step":
                        q.kill()
                        next_projected_processes_list.remove(q)
                next_projected_processes_list.remove(p)
                continue

            if p.type() == "load_data" and not p.is_alive():
                if actions[p.get_system_number()] > 0:
                    next_projected_processes_list.append(
                        CloudProcess(function=self.train_on_data, system_number=p.get_system_number(),
                                           num_workers=actions[p.get_system_number()],
                                           deadline=np.ceil(self.max_training / actions[p.get_system_number()]),startup_function=self.train_on_data_startup))
                else:
                    waiting_list.append(CloudProcess(function=self.load_data, system_number=p.get_system_number(),
                                                           deadline=p.get_deadline(), num_workers=p.get_num_workers()))
                p.kill()
                next_projected_processes_list.remove(p)
                continue

            if p.type() == "start_sampling" and not p.is_alive():
                if actions[p.get_system_number()] > 0:
                    next_projected_processes_list.append(
                        CloudProcess(function=self.load_data, system_number=p.get_system_number(),
                                           deadline=self.max_load, num_workers=1))
                else:
                    waiting_list.append(
                        CloudProcess(function=self.start_sampling, system_number=p.get_system_number(),
                                           deadline=p.get_deadline(), num_workers=p.get_num_workers(),startup_function=self.start_sampling_startup))
                p.kill()
                next_projected_processes_list.remove(p)

                continue
            if p.is_alive():
                # p.update_running_time()
                if p.type() != "calculate_online_step":
                    if actions[p.get_system_number()] == 0:
                        next_projected_processes_list.remove(p)
                        waiting_list.append(p)
                        # p.set_deadline(2)
                    # else:
                    #    done=(np.floor(
                    #    self.max_sampling / p.get_num_workers()) - p.get_deadline()) * p.get_num_workers()
                    #    p.set_num_workers(actions[p.get_system_number()])
                    #    p.set_deadline(np.floor((self.max_training-done)/actions[p.get_system_number()]))
                # next_projected_processes_list.remove(p)

                #    running_processes_list.append(p)

                    continue
            p.kill()
            # if p in next_projected_processes_list:
            #    necessary_processes_list.append(p)
            #    next_projected_processes_list.remove(p)
            index = p.get_system_number()
            x0 = state_list[index]
            u0 = input_list[index]
            p.set_u0(u0)
            p.set_x0(x0)
        new_processes_list = []
        # if failure:
        new_processes_list = new_processes_list + [
            CloudProcess(function=self.calculate_online_step, system_number=num, x0=state_list[num],
                               u0=input_list[num], q=Queue(), deadline=10000, num_workers=1) for k, num in
            enumerate(failure_list) if actions[num] >= 1]
        waiting_list = waiting_list + [
            CloudProcess(function=self.calculate_online_step, system_number=num, x0=state_list[num],
                               u0=input_list[num], q=Queue(), deadline=failure_pred_list[k], num_workers=1) for k, num in
            enumerate(failure_list) if actions[num] == 0]
        new_processes_list = new_processes_list + [
            CloudProcess(function=self.start_sampling, system_number=num, num_workers=actions[num],
                               deadline=np.ceil(self.max_sampling / actions[num]),startup_function=self.start_sampling_startup) for num in failure_list if
            actions[num] >= 1]
        waiting_list = waiting_list + [
            CloudProcess(function=self.start_sampling, system_number=num, deadline=1000,startup_function=self.start_sampling_startup) for num in failure_list
            if actions[num] == 0]

        # self.online_list=self.online_list+[num for num in failure_list]
        # if change:
        #    new_processes_list = new_processes_list + [
        #        CloudProcessMockup(function=self.calculate_online_step, system_number=num, x0=state_list[num],u0=input_list[num], q=Queue(),
        #                     deadline=change_number_list[num]) for num in
        #        change_number_list]
        #    new_processes_list = new_processes_list + [CloudProcessMockup(function=self.start_sampling, system_number=num) for
        #                                               num in change_number_list]
        #    self.online_list=self.online_list+[num for num in change_number_list]
        # for p in new_processes_list:
        #    if p.check_if_necessary():
        #        necessary_processes_list.append(p)
        # for p in necessary_processes_list:
        #    if p in new_processes_list:
        #        new_processes_list.remove(p)
        # for system in finished_list:
        #    self.online_list.remove(system)
        next_process_list = next_projected_processes_list + new_processes_list
        for p in waiting_list:
            if p.type() != "calculate_online_step":
                p.set_deadline(p.get_deadline() + 1)
        # finished_processes=self.priority_scheduling(new_processes_list,necessary_processes_list,next_projected_processes_list,running_processes_list,actions)
        # finished_list=finished_processes+finished_list
        self.waiting_list = waiting_list.copy()
        self.p_list = next_process_list.copy()

        return next_process_list, waiting_list, finished_early,finished_list
    def schedule_processes_priority(self, actions,failure_list,state_list,failure_pred_list,input_list):

        for p in self.p_list:
            # try:
            p.reduce_duration()
            # except:
            #    print(p.type())
            #    print(p.get_num_workers())
            #    print("p_list")
            #    print("policy")
            #    sys.exit(1)
        finished_list = []
        for p in self.waiting_list:
            p.reduce_duration()
        waiting_list = self.waiting_list.copy()
        # if [1 for p in self.p_list if p.type()=="calculate_online_step" and p.get_deadline()<0]:
        #    print('Hi')
        next_projected_processes_list = []  # self.p_list.copy()
        running_processes_list = []
        necessary_processes_list = []
        old_processes_list = []
        new_processes_list = []

        for p in waiting_list:

            if p.type() == "load_data" and not p.is_alive():
                old_processes_list.append(p)
                next_projected_processes_list.append(
                    CloudProcess(function=self.train_on_data, system_number=p.get_system_number(),startup_function=self.train_on_data_startup))
                p.kill()
                self.waiting_list.remove(p)
                continue

            if p.type() == "start_sampling":
                if not p.is_alive() and p.get_num_workers() is not None:
                    old_processes_list.append(p)
                    next_projected_processes_list.append(
                        CloudProcess(function=self.load_data, system_number=p.get_system_number(),
                                           deadline=self.max_load))
                    p.kill()
                    self.waiting_list.remove(p)
                elif p.get_num_workers() is None:
                    self.waiting_list.remove(p)
                    new_processes_list.append(
                        CloudProcess(function=self.start_sampling, system_number=p.get_system_number(),
                                           deadline=1000,startup_function=self.start_sampling_startup))

                continue

            if p.type() == "calculate_online_step":

                if p.check_if_necessary():  # and p.type()=="calculate_online_step":
                    self.waiting_list.remove(p)
                    necessary_processes_list.append(p)
                    p.set_deadline(1000)
                    p.set_num_workers(1)
                else:
                    self.waiting_list.remove(p)
                    new_processes_list.append(p)
                # for q in waiting_list:
                #    if q.type()=="start_sampling" and q.get_system_number()==p.get_system_number():
                #        new_processes_list.append(q)
                #        self.waiting_list.remove(q)

        p_list = self.p_list.copy()
        for p in p_list:

            if p.type() == "train_on_data" and not p.is_alive():

                finished_list.append(p.get_system_number())

                p.kill()
                for q in self.p_list:
                    if q.get_system_number() == p.get_system_number() and q.type() == "calculate_online_step":
                        q.kill()
                        self.p_list.remove(q)
                for q in necessary_processes_list:
                    if q.get_system_number() == p.get_system_number() and q.type() == "calculate_online_step":
                        q.kill()
                        necessary_processes_list.remove(q)
                for q in running_processes_list:
                    if q.get_system_number() == p.get_system_number() and q.type() == "calculate_online_step":
                        q.kill()
                        running_processes_list.remove(q)

                self.p_list.remove(p)
                continue

            if p.type() == "load_data" and not p.is_alive():
                old_processes_list.append(p)
                next_projected_processes_list.append(
                    CloudProcess(function=self.train_on_data, system_number=p.get_system_number(),startup_function=self.train_on_data_startup))
                p.kill()
                self.p_list.remove(p)
                continue

            if p.type() == "start_sampling" and not p.is_alive():
                old_processes_list.append(p)
                next_projected_processes_list.append(
                    CloudProcess(function=self.load_data, system_number=p.get_system_number(),
                                       deadline=self.max_load))
                p.kill()
                self.p_list.remove(p)

                continue
            if p.is_alive() and not p.type() == "calculate_online_step":
                # p.update_running_time()
                running_processes_list.append(p)
                self.p_list.remove(p)

                continue
            p.kill()
            if p in self.p_list:  # next_projected_processes_list:
                necessary_processes_list.append(p)
                # p.set_deadline(1000)
                # p.set_num_workers(1)
                self.p_list.remove(p)
            index = p.get_system_number()
            x0 = state_list[index]
            u0 = input_list[index]
            p.set_u0(u0)
            p.set_x0(x0)

        # new_processes_list = []
        [p.set_deadline(1000) for p in necessary_processes_list]
        new_processes_list = new_processes_list + [
            CloudProcess(function=self.calculate_online_step, system_number=num, x0=state_list[num],
                               u0=input_list[num], q=Queue(), deadline=failure_pred_list[k]) for k, num in
            enumerate(failure_list)]

        new_processes_list = new_processes_list + [
            CloudProcess(function=self.start_sampling, system_number=num, deadline=1000,startup_function=self.start_sampling_startup) for num in failure_list]
        # self.online_list = self.online_list + [num for num in failure_list]
        # if new_processes_list!=[]:
        #    print('Hi')
        # for p in new_processes_list:
        #    if p.type()=="calculate_online_step":
        #        necessary_processes_list.append(p)
        #        p.set_deadline(1000)
        #        p.set_num_workers(1)
        for p in necessary_processes_list:
            if p in new_processes_list:
                new_processes_list.remove(p)
        # for system in finished_list:
        #    self.online_list.remove(system)
        self.p_list = []
        finished_early = self.priority_scheduling(new_processes_list, necessary_processes_list,
                                                  next_projected_processes_list, running_processes_list,
                                                  old_processes_list.copy())
        # if len(self.p_list)==1 and self.p_list[0].type()=="calculate_online_step":
        #    print('Hi')
        for p in self.waiting_list:
            if p.type() != "calculate_online_step":
                p.set_deadline(p.get_deadline() + 1)
        next_process_list = self.p_list.copy()
        next_waiting_list = self.waiting_list.copy()
        # finished_list = finished_processes + finished_list
        return next_process_list, next_waiting_list, finished_early,finished_list

    def priority_scheduling(self, new_p_list, necessary_p_list, next_p_list, running_p_list, old_processes):
        enough_space = True

        finished = []
        finished_early = 0
        running_p_list = running_p_list + self.waiting_list.copy()  # self.waiting_list.copy()
        self.waiting_list = []
        # self.p_list = running_p_list.copy()
        if np.sum(np.asarray([p.get_num_workers() for p in running_p_list])) + len(necessary_p_list) > self.max_process:
            enough_space = False
        while not enough_space:
            if np.sum(np.asarray([p.get_num_workers() for p in running_p_list])) <= self.max_process - len(
                    necessary_p_list):
                enough_space = False
                # sample_list = [p for p in running_p_list if p.type() != "train_on_data"]
                # if sample_list == []:
                #    "Mark: sample list can also contain load_data"
                train_list = [p for p in running_p_list if p.type() == "train_on_data"]
                if train_list != []:
                    index = train_list.index(max([p.get_running_time() for p in train_list]))
                    p = train_list[index]

                    finished_early += p.get_deadline()
                    finished.append(p.get_system_number())
                    found = False
                    for q in next_p_list:
                        if q.get_system_number() == p.get_system_number() and q.type() == "calculate_online_step":
                            found = True
                            q.kill()
                            next_p_list.remove(q)
                    for q in necessary_p_list:
                        if q.get_system_number() == p.get_system_number() and q.type() == "calculate_online_step":
                            found = True
                            q.kill()
                            necessary_p_list.remove(q)
                    p.kill()
                else:
                    index = [p.get_num_workers() for p in running_p_list].index(
                        max([p.get_num_workers() for p in running_p_list]))
                    p = running_p_list[index]
                    # p.set_deadline(2)
                    self.waiting_list.append(p)
                # else:
                #    index = [p.get_num_workers() for p in sample_list].index(
                #        max([p.get_num_workers() for p in sample_list]))
                #    p = sample_list[index]
                #    # p.set_deadline(2)
                #    self.waiting_list.append(p)
                #    #index = [p.get_deadline() for p in train_list].index(
                #    #    max([p.get_deadline() for p in train_list]))
                #    #p = train_list[index]
                #    #finished_early+=1
                #    #finished.append(p.get_system_number())
                #    #found=False
                #    #for q in next_p_list:
                #    #    if q.get_system_number() == p.get_system_number() and q.type() == "calculate_online_step":
                #    #        found = True
                #    #        q.kill()
                #    #        next_p_list.remove(q)
                #    #for q in necessary_p_list:
                #    #    if q.get_system_number() == p.get_system_number() and q.type() == "calculate_online_step":
                #    #        found = True
                #    #        q.kill()
                #    #        necessary_p_list.remove(q)

                ##p.kill()
                running_p_list.remove(p)
            else:
                enough_space = True
        self.p_list = running_p_list.copy() + necessary_p_list.copy()
        N = np.sum(np.asarray([p.get_num_workers() for p in self.p_list]))
        wait_processes_list = new_p_list.copy() + next_p_list.copy()  # +wait_without.copy()#self.waiting_list.copy() +
        assignment_next_p = [0 for k in range(len(new_p_list))] + [k + 1 for k in range(len(next_p_list))]
        num_wait_processes = len(wait_processes_list)

        if num_wait_processes <= self.max_process - N:
            simple_list = [p for p in wait_processes_list if
                           p.type() != "train_on_data" and p.type() != "start_sampling"]
            [p.set_deadline(1000) if p.type() == "calculate_online_step" else p.set_deadline(self.max_load) for p in
             simple_list]
            [p.set_num_workers(num_workers=1) for p in simple_list]

            self.p_list = self.p_list + simple_list.copy()
            N = np.sum(np.asarray([p.get_num_workers() for p in self.p_list]))
            M = len([p for p in wait_processes_list if p.type() == "train_on_data" or p.type() == "start_sampling"])
            if M > 0:
                num_workers = min(int((self.max_process - N) / M), self.max_process - len(necessary_p_list))
                # done = [(np.floor(
                #    self.max_sampling / p.get_num_workers()) - p.get_deadline()) * p.get_num_workers() if p.type() == "start_sampling" or p.type() == "train_on_data" else 0
                #        for p
                #        in self.waiting_list]
                # done += [0 for k in range(len(new_p_list + next_p_list))]
                [p.set_num_workers(num_workers=num_workers) for p in wait_processes_list if
                 p.type() == "train_on_data" or p.type() == "start_sampling"]
                [p.set_deadline(np.ceil(self.max_sampling / num_workers)) for k, p in enumerate(wait_processes_list) if
                 p.type() == "train_on_data" or p.type() == "start_sampling"]
                self.p_list = self.p_list + [p for p in wait_processes_list if
                                             p.type() == "train_on_data" or p.type() == "start_sampling"]
            # self.waiting_list = []
        else:

            # done = [(np.floor(
            #    self.max_sampling / p.get_num_workers()) - p.get_deadline()) * p.get_num_workers() if p.type() == "start_sampling" or p.type() == "train_on_data" else 0
            #        for p
            #        in self.waiting_list]
            # done += [0 for k in range(len(new_p_list + next_p_list))]
            N = np.sum(np.asarray([p.get_num_workers() for p in self.p_list]))
            number_of_slots = int(self.max_process - N)
            control_tasks = [p for p in wait_processes_list if p.type() == "calculate_online_step"]
            ass_control = [assignment_next_p[k] for k, p in enumerate(wait_processes_list) if
                           p.type() == "calculate_online_step"]
            samples_tasks = [p for p in wait_processes_list if p.type() == "start_sampling"]
            ass_sampling = [assignment_next_p[k] for k, p in enumerate(wait_processes_list) if
                            p.type() == "start_sampling"]
            load_tasks = [p for p in wait_processes_list if p.type() == "load_data"]
            ass_load = [assignment_next_p[k] for k, p in enumerate(wait_processes_list) if p.type() == "load_data"]
            training_task = [p for p in wait_processes_list if p.type() == "train_on_data"]
            ass_training = [assignment_next_p[k] for k, p in enumerate(wait_processes_list) if
                            p.type() == "train_on_data"]
            # done_sorted=done_control+done_training+done_load+done_sampling
            sorted_waiting_list = control_tasks + training_task + load_tasks + samples_tasks
            assignment_next_p_sorted = ass_control + ass_training + ass_load + ass_sampling
            new_p_list = sorted_waiting_list[0:number_of_slots]
            [p.set_num_workers(num_workers=1) for p in new_p_list]
            [p.set_deadline(self.max_sampling) for p in new_p_list if p.type() == "start_sampling"]
            [p.set_deadline(self.max_training) for p in new_p_list if p.type() == "train_on_data"]
            [p.set_deadline(1000) for p in new_p_list if p.type() == "calculate_online_tasks"]
            [p.set_deadline(self.max_load) for p in new_p_list if p.type() == "load_data"]
            self.p_list = self.p_list + new_p_list
            new_waiting_list = sorted_waiting_list[number_of_slots:len(sorted_waiting_list)].copy()
            for index, k in enumerate(assignment_next_p_sorted[number_of_slots:len(sorted_waiting_list)]):
                if k > 0:
                    new_waiting_list[index] = old_processes[k - 1].copy()
                    # new_waiting_list[index].set_deadline(1)
            # [p.set_deadline(2) for p in new_waiting_list if p.type() is not "calculate_online_step"]
            self.waiting_list = self.waiting_list + new_waiting_list.copy()
            # sorted(control_tasks,key=CloudProcess.get_wait_duration())

        return finished_early
    def schedule_processes_priority_old(self,failure,failure_index_list,failure_pred_list,change,change_number_list,state_list,input_list):

        finished_list=[]
        next_projected_processes_list=self.p_list.copy()
        running_processes_list=[]
        necessary_processes_list=[]
        for p in self.waiting_list:
            p.update_wait_duration()
            if p.check_if_necessary():
                self.waiting_list.remove(p)
                necessary_processes_list.append(p)

        for p in self.p_list:
            if p.type() == "train_on_data" and not p.is_alive():
                finished_list.append(p.get_system_number())
                p.kill()
                for q in next_projected_processes_list:
                    if q.get_system_number()==p.get_system_number() and q.type()=="calculate_online_step":
                        q.kill()
                        next_projected_processes_list.remove(q)
                for q in necessary_processes_list:
                    if q.get_system_number()==p.get_system_number() and q.type()=="calculate_online_step":
                        q.kill()
                        necessary_processes_list.remove(q)
                next_projected_processes_list.remove(p)
                continue

            if p.type()=="load_data" and not p.is_alive():
                next_projected_processes_list.append(CloudProcess(function=self.train_on_data,system_number=p.get_system_number(),startup_function=self.train_on_data_startup))
                p.kill()
                next_projected_processes_list.remove(p)
                continue

            if p.type()=="start_sampling" and not p.is_alive():
                next_projected_processes_list.append(CloudProcess(function=self.load_data,system_number=p.get_system_number()))
                p.kill()
                next_projected_processes_list.remove(p)

                continue
            if p.is_alive():
                p.update_running_time()
                next_projected_processes_list.remove(p)
                running_processes_list.append(p)
                continue
            p.kill()
            if p in next_projected_processes_list:
                necessary_processes_list.append(p)
                next_projected_processes_list.remove(p)
            index=p.get_system_number()
            x0=state_list[index]
            u0=input_list[index]
            p.set_u0(u0)
            p.set_x0(x0)
        new_processes_list=[]
        if failure:
            new_processes_list = new_processes_list + [CloudProcess(function=self.calculate_online_step,system_number=num, x0=state_list[num],u0=input_list[num],q=Queue(),deadline=failure_pred_list[k]) for k,num in
                                         enumerate(failure_index_list)]
            new_processes_list = new_processes_list + [CloudProcess(function=self.start_sampling,system_number=num,startup_function=self.start_sampling_startup) for num in failure_index_list]
            self.online_list=self.online_list+[num for num in failure_index_list]
        if change:
            new_processes_list = new_processes_list + [
                CloudProcess(function=self.calculate_online_step, system_number=num, x0=state_list[num],u0=input_list[num], q=Queue(),
                             deadline=1) for num in
                change_number_list if num not in self.online_list]
            new_processes_list = new_processes_list + [CloudProcess(function=self.start_sampling, system_number=num,startup_function=self.start_sampling_startup) for
                                                       num in change_number_list if num not in self.online_list]
            self.online_list=self.online_list+[num for num in change_number_list if num not in self.online_list]
        for p in new_processes_list:
            if p.check_if_necessary():
                necessary_processes_list.append(p)
        for p in necessary_processes_list:
            if p in new_processes_list:
                new_processes_list.remove(p)

        finished_processes=self.priority_scheduling(new_processes_list,necessary_processes_list,next_projected_processes_list,running_processes_list)
        finished_list=finished_processes+finished_list
        for system in finished_list:
            self.online_list.remove(system)
        return finished_list




        #restart_processes = [num for num in self.online_list if num not in failure_index_list or num not in change_number_list]
        #p_list = p_list + [
        #    CloudProcess(function=self.calculate_online_step, system_number=num, x0=state_list[num], q=Queue())
        #    for num in restart_processes]


    def priority_scheduling_old(self,new_p_list,necessary_p_list,next_p_list,running_p_list):
        enough_space=True
        finished=[]
        self.p_list=running_p_list
        if np.sum(np.asarray([p.get_num_workers() for p in running_p_list]))+len(necessary_p_list)>self.max_process:
            enough_space=False
        while not enough_space:
            if np.sum(np.asarray([p.get_num_workers() for p in running_p_list]))>self.max_process-len(necessary_p_list):
                enough_space=False
                train_list=[p for p in running_p_list if p.type()=="train_on_data"]
                if train_list==[]:
                    "Mark: sample list can also contain load_data"
                    sample_list=[p for p in running_p_list if p.type()!="train_on_data"]
                    #index = sample_list.index(max([p.get_running_time() for p in sample_list]))
                    index = [p.get_running_time() for p in sample_list].index(
                        max([p.get_running_time() for p in sample_list]))
                    p = sample_list[index]
                    self.waiting_list.append(p)
                else:
                    index = [p.get_running_time() for p in train_list].index(max([p.get_running_time() for p in train_list]))
                    p=train_list[index]
                    finished.append(p.get_system_number())
                    for q in next_p_list:
                        if q.get_system_number() == p.get_system_number() and q.type() == p.type():
                            q.kill()
                            next_p_list.remove(q)
                    for q in necessary_p_list:
                        if q.get_system_number() == p.get_system_number() and q.type() == p.type():
                            q.kill()
                            next_p_list.remove(q)
                p.kill()
                running_p_list.remove(p)
            else:
                enough_space=True
        self.p_list=running_p_list+necessary_p_list
        N=np.sum(np.asarray([p.get_num_workers() for p in self.p_list]))

        wait_processes_list=self.waiting_list+new_p_list+next_p_list
        num_wait_processes=len(wait_processes_list)
        if num_wait_processes<=self.max_process-N:
            self.p_list=self.p_list+[p for p in wait_processes_list if p.type()!="train_on_data" and p.type()!="start_sampling"]
            N = np.sum(np.asarray([p.get_num_workers() for p in self.p_list]))
            M=len([p for p in wait_processes_list if p.type()=="train_on_data" or p.type()=="start_sampling"])
            if M>0:
                num_workers=int((self.max_process-N)/M)
                [p.set_num_workers(num_workers=num_workers) for p in wait_processes_list if p.type() == "train_on_data" or p.type() == "start_sampling"]
                self.p_list=self.p_list+[p for p in wait_processes_list if p.type()=="train_on_data" or p.type()=="start_sampling"]
            self.waiting_list=[]
        else:
            N = np.sum(np.asarray([p.get_num_workers() for p in self.p_list]))
            number_of_slots=self.max_process-N
            control_tasks=[p for p in wait_processes_list if p.type()=="calculate_online_step"]
            samples_tasks=[p for p in wait_processes_list if p.type()=="start_sampling"]
            load_tasks=[p for p in wait_processes_list if p.type()=="load_data"]
            training_task=[p for p in wait_processes_list if p.type()=="train_on_data"]
            sorted_waiting_list=control_tasks+training_task+load_tasks+samples_tasks
            self.p_list =self.p_list+sorted_waiting_list[0:number_of_slots]
            [p.set_num_workers(num_workers=1) for p in wait_processes_list if
             p.type() == "train_on_data" or p.type() == "start_sampling"]
            self.waiting_list=self.waiting_list+sorted_waiting_list[number_of_slots:len(sorted_waiting_list)]
            #sorted(control_tasks,key=CloudProcess.get_wait_duration())

        return finished

    def visualize_processes(self):
        current_process=0
        for p in self.p_list:
            size_rect=p.get_num_workers()
            process_type=p.type()
            sys_number=p.get_system_number()
            self.draw_rect(size_rect,process_type,sys_number,current_process)
            current_process+=size_rect
        self.ax.set_xlim(0, self.max_process)
        self.ax.set_ylim(max(0, self.current_time - 8), self.current_time + 1)
        #self.fig.show()


    def draw_rect(self, size_rect,process_type,sys_number,current_process):
        left_point=(current_process,self.current_time)
        width=size_rect
        height=1
        mid_point=(left_point[0]+width/2,left_point[1]+height/2)
        self.ax.add_patch(Rectangle(left_point,width,height,edgecolor='k',facecolor=self.color_code(process_type)))
        for k in range(size_rect):
            point = (left_point[0] + k+0.5, left_point[1] + height / 2)
            self.ax.text(point[0],point[1],sys_number)

    @staticmethod
    def color_code(process_type):
        if process_type=="calculate_online_step":
            return 'r'
        elif process_type=="start_sampling":
            return 'b'
        elif process_type=="train_on_data":
            return 'g'
        elif process_type=="load_data":
            return 'm'
        else:
            return 'w'

    def start_processes(self):
        for p in self.p_list:
            #if not p.is_alive():
                p.start()

    def calculate_online_steps(self,all_mpc=False,x0_list=[],u0_pred_list=[]):

        #q = multiprocessing.Queue()
        self.online_list=[]
        if all_mpc:
            u0_list=[]
            self.online_list=[k for k in range(len(self.mpc_list))]
            for k in self.online_list:
                u0=self.calculate_online_step(k,x0_list[k],u0_pred_list[k])
                u0_list.append(u0)
        else:
            for p in self.p_list:
                if p.type()=="calculate_online_step":
                    self.online_list.append(p.get_system_number())
            u0_list=[np.zeros((self.mpc_list[k].model.n_u,1)) for k in self.online_list]#np.zeros((len(self.online_list),1))
            for p in self.p_list:
                if p.type()=="calculate_online_step":
                    u0=p.get_value()
                    u0_list[self.online_list.index(p.get_system_number())]=u0
        return u0_list


    def calculate_online_step(self,sys_number,x0,u0):
        # calculate optimal control input
        #plt.pause(0.1)
        '''
        path = self.parameters['system_list'][sys_number]
        sys.path.append('path')
        model_path = path + '.template_model'
        mpc_path = path + '.template_mpc'
        simulator_path = path + '.template_simulator'
        template_model = importlib.import_module(model_path)
        template_mpc = importlib.import_module(mpc_path)
        template_simulator = importlib.import_module(simulator_path)
        #if sys_number > 0:
        importlib.reload(template_model)
        importlib.reload(template_mpc)
        importlib.reload(template_simulator)
        model = template_model.template_model()
        mpc = template_mpc.template_mpc(model)
        '''
        mpc=self.mpc_list[sys_number]
        '''
        tvp_template = self.mpc_list[sys_number].tvp_fun(0)

        def tvp_fun(t_curr):
            x_tilde = tvp_template['_tvp', -1, 'x_tilde']
            A = tvp_template['_tvp', -1, 'A']
            F = tvp_template['_tvp', -1, 'F']
            B = tvp_template['_tvp', -1, 'B']
            Q = tvp_template['_tvp', -1, 'Q']
            # tvp_template['_tvp', k, 'R'] = R
            K = tvp_template['_tvp', -1, 'K']
            P = tvp_template['_tvp', -1, 'P']

            for s in range(self.mpc_list[sys_number].n_horizon + 1):
                for k in range(self.mpc_list[sys_number].n_horizon + 1):
                    tvp_template[
                        '_tvp', k, 'x_tilde'] = x_tilde   # np.array([[system_1], [system_0], [system_0], [system_0]])##system_0.02*(s+t_curr)#system_0.05*t_curr
                    tvp_template['_tvp', k, 'A'] = A
                    tvp_template['_tvp', k, 'F'] = F
                    tvp_template['_tvp', k, 'B'] = B
                    tvp_template['_tvp', k, 'Q'] = Q
                    # tvp_template['_tvp', k, 'R'] = R
                    tvp_template['_tvp', k, 'K'] = K.T
                    tvp_template['_tvp', k, 'P'] = P
            # self.template = tvp_template
            return tvp_template
        mpc.set_tvp_fun(tvp_fun)
        '''

        mpc.reset_history()
        mpc.t0 = self.current_time
        mpc.x0 = x0
        mpc.u0=u0.squeeze()
        mpc.set_initial_guess()
        u0 = mpc.make_step(x0)
        if not mpc.data.success[-1]:
            print('Error')

        #q.put(u0)
        return u0
    def setpoint_change(self,k):
        return (k==60,[0])
    #def sample_and_train_on_new_data(self,number_of_processes):
    #    #self.calculate_feasible_set()
    #    self.sample_new_data(number_of_processes)
    #    self.train_on_data(number_of_processes)
    def start_sampling_startup(self,system_number,num_workers):
        # Sample new data
        sp=do_mpc.sampling.SamplingPlanner()
        sp.sampling_plan = []
        mpc = self.mpc_list[system_number]
        simulator = self.simulator_list[system_number]
        estimator = self.estimator_list[system_number]
        G=self.G_list[system_number]
        # Generate sampling function for initial states
        def gen_initial_states():
            np.random.seed(99)
            number=0
            x_array=np.zeros((mpc.model.n_x,self.num_train_trajectories))
            while number<self.num_train_trajectories:
                x0 = np.random.uniform(mpc.bounds['lower','_x','x'],mpc.bounds['upper','_x','x'])
                if x0.T@G@x0<1:
                    x_array[:,number]=x0.squeeze()
                    number+=1
            return x_array

        # Add variables

        sp.set_sampling_var('X0', gen_initial_states)
        sp.set_param(overwrite=True)
        sp.data_dir = './samples/'+str(system_number)+"/"
        training_points=gen_initial_states()
       # 25 Sample trajectories for the training and validation data set
        for k in range(self.num_train_trajectories):
            plan=sp.add_sampling_case(X0=training_points[:,k])
        """ Execute sampling plan """
        # Define the sampling function, in this case closed-loop mpc runs


        def run_closed_loop(X0):
            #plt.pause(0.5)
            mpc.reset_history()
            mpc.t0=self.current_time
            simulator.reset_history()
            estimator.reset_history()

            # set initial values and guess
            x0=X0
            mpc.x0 = x0
            mpc.u0=np.zeros((mpc.model.n_u,1))
            simulator.x0 = x0
            estimator.x0 = x0
            mpc.set_initial_guess()

            # run the closed loop for x steps
            for k in range(self.num_train_samples_per_trajectories):
                u0 = mpc.make_step(x0)
                y_next = simulator.make_step(u0)
                if mpc.data['success'][-1]==0:
                     return mpc.data
                x0 = estimator.make_step(y_next)

            return mpc.data

        # Feed plan to sampler
        sampler = do_mpc.sampling.Sampler(plan)
        sampler.data_dir = './samples/'+str(system_number)+"/"

        # set sampling function
        sampler.set_sample_function(run_closed_loop)
        sampler.set_param(overwrite=True)
        sampler.set_param(print_progress=False)
        # Generate the samples
        #q.put(sampler)
        #worker_lists=self.indices(num_workers=num_workers)
        maximal=self.num_train_trajectories
        max=self.num_train_samples_per_trajectories
        return mpc,estimator,simulator,training_points,maximal,max
        #p_list = [Process(target=self.sample_indices,args=(sampler, worker_lists[num])) for num in range(num_workers)]
        #for p in p_list:
        #    p.start()
        directory = 'samples/' + str(system_number) + '/'
        finished=False
        #while not finished:
        #    finished = (len(os.listdir(directory)) == self.num_train_trajectories)
        #plt.pause(10.0)


    def indices(self,num_workers):
        worker_lists=[[] for k in range(num_workers)]
        for i in range(self.num_train_trajectories):
            worker_index = np.mod(i, num_workers)
            worker_lists[worker_index].append(i)
        return worker_lists
    def start_sampling(self,mpc,simulator,estimator,x0,u0):

            mpc.reset_history()
            mpc.t0 = self.current_time
            simulator.reset_history()
            estimator.reset_history()
            #u0=np.array([0])
            # set initial values and guess

            mpc.x0 = x0
            mpc.u0 = u0.squeeze()#np.zeros((mpc.model.n_u, 1))
            simulator.x0 = x0
            estimator.x0 = x0
            mpc.set_initial_guess()
            x0_list=[]
            u0_list=[]
            for k in range(1):
                u0 = mpc.make_step(x0)
                if mpc.data['success'][-1] == 0:
                    return x0_list,u0_list
                u0_list.append(u0)
                y_next = simulator.make_step(u0)
                x0 = estimator.make_step(y_next)
                x0_list.append(x0)

            return x0_list,u0_list

    #plt.pause(2.0)
    def load_data(self,system_number,input_data,output_data,deadline,num_workers=1):
        # Train on the current training set

        #input_data = []
        #output_data = []
        directory = 'samples/'+str(system_number)+'/'
        count=0
        finished=(os.listdir(directory) == [])
        for filename in os.listdir(directory):
            if count>1250:
                break
            f = os.path.join(directory, filename)
            if os.path.getsize(f) > 0:
                with open(f, 'rb') as pickle_file:
                    data = pickle.load(pickle_file)
                    for arr in data['x']:
                        arr.reshape((2,1))
                    if np.size(np.array(data['x']),0)>1:
                        old_u=np.array(data['u'][0:-1]).reshape((len(data['u'][0:-1]),1))#np.concatenate((np.zeros((1,np.shape(data._u[0:-1])[1])),data._u[0:-1]))
                        input_data.append(np.concatenate((np.array(data['x'][0:-1]).reshape((np.size(np.array(data['x'][0:-1]),0),np.size(np.array(data['x'][0:-1]),1))),old_u),axis=1))
                        output_data.append(data['u'][1:])
                        count+=np.size(np.array(data['x']),0)
                    os.remove(f)
            #else:
            #    with open('samples/error.txt', 'w') as f:
            #        print('Training space is infeasible!')
            #        f.write('Training space is infeasible!')
                #self.left_state_space+=system_1
                #self.right_state_space-=system_1
            #    return
        finished = (os.listdir(directory) == [])
        if deadline==1 and finished is not True:
            for filename in os.listdir(directory):
                #if os.path.getsize(d) > 0:
                os.remove(filename)
                finished=True
        if finished:
            #if any([np.size(input_data[k], 0)<self.num_train_samples_per_trajectories for k in range(len(input_data))]):
            #    with open('samples/error.txt', 'w') as f:
            #        print('Training space is infeasible!')
            #        f.write('Training space is infeasible!')
            #    return
            states=[]

            for input in input_data:
                for arr in input:
                    states.append(arr.tolist())
            states=np.array(states)
            #states = np.asarray(input_data).reshape(np.size(np.asarray(input_data), 0) * np.size(np.asarray(input_data), 1),
            #                                        np.size(np.asarray(input_data), 2))
            input = []
            for output in output_data:
                for arr in output:
                    input.append(arr.tolist())
            input=np.array(input).reshape((np.size(np.array(input)),1))
            #input = np.asarray(output_data).reshape(
            #    np.size(np.asarray(output_data), 0) * np.size(np.asarray(output_data), 1),
            #    np.size(np.asarray(output_data), 2))
            results=dict()
            results["states"]=states
            results["input"]=input
            if not os.path.exists('./data/' + str(system_number)):
                os.mkdir('./data/' + str(system_number))
            with open('data/'+str(system_number)+'/data.pickle', 'wb') as handle:
                pickle.dump(results, handle)
                #pickle.dump(states,handle)
        return input_data,output_data

    def train_on_data_startup(self,system_number,num_workers):
        device ='cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #PATH = './nets_start/' + str(system_number) + '/model_net.pth'
        #states=self.mpc_list[system_number].model.n_x
        #inputs=self.mpc_list[system_number].model.n_u
        #net = NeuralNetwork(states+inputs, inputs)
        #if not os.path.exists('./nets_start/' + str(states)):
        #    os.mkdir('./nets_start/' + str(states))
        #    # net = NeuralNetwork(nx + nu, nu)
        #else:
        #    net.load_state_dict(torch.load(PATH))
        #self.current_net_list[system_number]=net
        self.current_net_list[system_number].to(device)
        #try:
        with open('data/'+str(system_number)+'/data.pickle', 'rb') as handle:
                results = pickle.load(handle)
        os.remove('data/'+str(system_number)+'/data.pickle')
        #except:
        #    plt.pause(5)
        #    with open('data/'+str(system_number)+'/data.pickle', 'rb') as handle:
        #        results = pickle.load(handle)
        #    os.remove('data/'+str(system_number)+'/data.pickle')
        states=results["states"]
        input=results["input"]
        #### set batch size and transform for the image preprocessing. The masked images are normalized with its mean and stddev over all images
        batch_size = self.batch_size



        ### Generate own dataset based upon the dataset_class.py
        dataset = CustomMPCDataset(states, input)
        length=len(states)
        trainset, testset = random_split(dataset,[int(np.ceil(length*self.train_split)),int(np.floor(length*(1-self.train_split)))],)# [int(np.ceil(self.num_train_samples_per_trajectories*self.num_train_trajectories*self.train_split)), int(np.floor(self.num_train_samples_per_trajectories*self.num_train_trajectories*(1-self.train_split)))],generator=torch.Generator().manual_seed(99))

        trainloader = DataLoader(trainset, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers, drop_last=True)

        testloader = DataLoader(testset, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers)


        ### Create an instance of the network model and use MSE as loss

        net = self.current_net_list[system_number]
        PATH = './nets_start/' + str(system_number) + '/model_net.pt'

        #n_x=self.mpc_list[system_number].model.n_x
        #n_u=self.mpc_list[system_number].model.n_u
        #net= NeuralNetwork(n_x+ n_u, n_u)
        #if not os.path.exists('./nets_start/' + str(states)):
        #    os.mkdir('./nets_start/' + str(states))
        #    # net = NeuralNetwork(nx + nu, nu)
        #else:
        #    net.load_state_dict(torch.load(PATH))
        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=self.learning_rate)
        return criterion,optimizer,net,trainloader,testloader,self.max_epoch,self.learning_rate
        ########################################################################
        # Train the network
        # ^^^^^^^^^^^^^^^^^^^^
    def train_on_data(self,criterion,optimizer,net,trainloader,testloader,device,system_number,number_of_increased_loss,old_test_loss,num_iter,num_workers,k):
        save_loss = []
        save_test_loss = []
        epoch = 0
        #old_test_loss = 100
        early_stopped = True
        #number_of_increased_loss = 1
        #while epoch < self.max_epoch and early_stopped:  # loop over the dataset multiple times
        end=False
        loss_per_epoch = 0.0
        test_loss_per_epoch = 0.0
        num_workers_training=int(np.ceil(num_workers*0.84))
        num_workers_test = int(np.floor(num_workers * 0.16))
        #gradients = [p.grad for p in net.state_dict()]

        if num_workers_test>=1:
            for i, data in enumerate(testloader, 0):
                # get the inputs; data is a list of [states,inputs]
                if i>num_workers_test:
                    break
                states, input = data
                num_iter += 1
                states = states.to(device)
                input = input.to(device)
                outputs = net(states)
                # input = torch.reshape(input, (test_batch_size,))
                test_loss = criterion(outputs, input)
                test_loss_per_epoch += test_loss.item()

            print(f'[{epoch + 1}] test loss: {test_loss_per_epoch:.3f}')
            save_test_loss.append(test_loss_per_epoch)

            if test_loss_per_epoch < old_test_loss:
                old_test_loss = test_loss_per_epoch
                number_of_increased_loss = 0
            else:
                number_of_increased_loss += 1

            if number_of_increased_loss > self.early_stopping:  # and epoch > 200:
                early_stopped = False
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [states, inputs]
                if i > num_workers_training:
                    break
                states, input = data
                num_iter += 1
                states = states.to(device)
                input = input.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize

                outputs = net(states)

                loss = criterion(outputs, input)
                loss.backward()

                loss_per_epoch += loss.item()

            optimizer.step()
            # print statistics
            # net=net.state_dict()
            # gradients=[]
            # gradients=[p.grad for p in net.parameters()]

            print(f'[{epoch + 1}] loss: {loss_per_epoch:.3f}')
            save_loss.append(loss_per_epoch)

        else:
            if False:#np.mod(k,4)==0:
                for i, data in enumerate(trainloader, 0):
                    # get the inputs; data is a list of [states,inputs]
                    if i > num_workers_training:
                        break
                    states, input = data
                    num_iter += 1
                    states = states.to(device)
                    input = input.to(device)
                    outputs = net(states)
                    # input = torch.reshape(input, (test_batch_size,))
                    test_loss = criterion(outputs, input)
                    test_loss_per_epoch += test_loss.item()

                print(f'[{epoch + 1}] test loss: {test_loss_per_epoch:.3f}')
                save_test_loss.append(test_loss_per_epoch)

                if test_loss_per_epoch < old_test_loss:
                    old_test_loss = test_loss_per_epoch
                    number_of_increased_loss = 0
                else:
                    number_of_increased_loss += 1

                if number_of_increased_loss > self.early_stopping:  # and epoch > 200:
                    early_stopped = False
            else:
                for i, data in enumerate(trainloader, 0):
                    # get the inputs; data is a list of [states, inputs]
                    if i > num_workers_training:
                        break
                    states, input = data
                    num_iter += 1
                    states = states.to(device)
                    input = input.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize

                    outputs = net(states)

                    loss = criterion(outputs, input)
                    loss.backward()

                    loss_per_epoch += loss.item()

                optimizer.step()
                # print statistics
                # net=net.state_dict()
                # gradients=[]
                # gradients=[p.grad for p in net.parameters()]

                print(f'[{epoch + 1}] loss: {loss_per_epoch:.3f}')
                save_loss.append(loss_per_epoch)
        #PATH = './nets/' + str(system_number) + '/model_net.pth'
        #torch.save(net.state_dict(), PATH)


        #epoch = epoch + 1
        #print(num_iter)
        #print(self.max_epoch)
        if not early_stopped or num_iter>=100*self.max_epoch:#not early_stopped or
            print('Finished Training')
            end=True
        self.current_net_list[system_number]=net
        PATH = './nets/'+str(system_number)+'/model_net.pt'
        torch.save(net.state_dict(), PATH)

        return number_of_increased_loss,old_test_loss,num_iter,end#,gradients
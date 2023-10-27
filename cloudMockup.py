import matplotlib.pyplot as plt
from multiprocessing import Queue
from matplotlib.patches import Patch
from CloudProcess import CloudProcessMockup
from cloud import cloud
import importlib
import do_mpc
from model_class import NeuralNetwork
import sys
import numpy as np

class cloudMockup(cloud):
    def __init__(self, parameters=None,scenario=None):  #
        if parameters is not None:


            self.max_process = parameters['max_process']  # multiprocessing.cpu_count()
            self.num_train_trajectories = parameters['num_train_trajectories']
            self.num_train_samples_per_trajectories = parameters['num_train_samples_per_trajectories']
            self.num_approx_feasible_set = parameters['num_approx_feasible_set']
            self.number_of_constraint_points = parameters['number_of_constraint_points']
            self.early_stopping = parameters['early_stopping']
            self.learning_rate = parameters['learning_rate']
            self.max_epoch = parameters['max_epoch']
            self.train_split = parameters['train_split']
            self.batch_size = parameters['batch_size']
            self.number_embedded = parameters['N_embedded']
            self.current_time = 0
            self.p_list = []
            self.waiting_list = []
            self.mpc_list = []
            self.simulator_list = []
            self.estimator_list = []
            self.current_net_list = []
            self.linear_list = []

            for k in range(self.number_embedded):
                path = parameters['system_list'][k]
                sys.path.append('path')
                model_path = path + '.template_model'
                mpc_path = path + '.template_mpc'
                simulator_path = path + '.template_simulator'
                template_model = importlib.import_module(model_path)
                template_mpc = importlib.import_module(mpc_path)
                template_simulator = importlib.import_module(simulator_path)
                if k > 0:
                    importlib.reload(template_model)
                    importlib.reload(template_mpc)
                    importlib.reload(template_simulator)
                model = template_model.template_model()
                mpc = template_mpc.template_mpc(model)
                linear = len(mpc.tvp_fun(0)['_tvp', -1].full()) > 2
                simulator = template_simulator.template_simulator(model)
                states = mpc.model.n_x
                inputs = mpc.model.n_u
                self.mpc_list.append(mpc)  # [template_mpc(model) for k in range(self.number_embedded)]
                self.simulator_list.append(
                    simulator)  # [template_simulator(model) for k in range(self.number_embedded)]
                self.estimator_list.append(
                    do_mpc.estimator.StateFeedback(model))  # [estimator for k in range(self.number_embedded)]
                self.current_net_list.append(
                    NeuralNetwork(states + inputs, inputs))  # for k in range(self.number_embedded))
                self.linear_list.append(linear)

            self.sp_list = []
            self.G_list = []
            self.online_list = []

        else:

            self.p_list = []
            self.waiting_list = []
            if scenario:
                self.system_size=scenario['system_size']
                self.max_sampling = scenario['max_sampling']
                self.max_training = scenario['max_training']
                self.max_load = scenario['max_load']
                self.max_process = scenario['size_p']
            else:
                self.system_size=4
                self.max_sampling = 16
                self.max_training = 16
                self.max_process = 8
                self.max_load = 2


    def copy(self):
        Cloud = cloudMockup()

        Cloud.max_process = self.max_process
        Cloud.num_train_trajectories = self.num_train_trajectories
        Cloud.num_train_samples_per_trajectories = self.num_train_samples_per_trajectories
        Cloud.num_approx_feasible_set = self.num_approx_feasible_set
        Cloud.number_of_constraint_points = self.number_of_constraint_points
        Cloud.early_stopping = self.early_stopping
        Cloud.learning_rate = self.learning_rate
        Cloud.max_epoch = self.max_epoch
        Cloud.train_split = self.train_split
        Cloud.batch_size = self.batch_size
        Cloud.number_embedded = self.number_embedded
        Cloud.current_time = self.current_time
        Cloud.p_list = self.p_list.copy()
        Cloud.waiting_list = self.waiting_list.copy()
        Cloud.fig, Cloud.ax = plt.subplots()
        allowed_tasks = ["calculate_online_step", "start_sampling", "train_on_data", "load_data"]
        Cloud.patches = []
        for t in allowed_tasks:
            Cloud.patches.append(Patch(color=self.color_code(t), label=t))
        box = Cloud.ax.get_position()
        Cloud.ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        Cloud.ax.legend(handles=self.patches, bbox_to_anchor=(1, 0.5))
        Cloud.mpc_list = self.mpc_list.copy()
        Cloud.simulator_list = self.simulator_list.copy()
        Cloud.estimator_list = self.estimator_list.copy()
        Cloud.current_net_list = self.current_net_list.copy()  # [NeuralNetwork() for k in range(self.number_embedded)]
        self.sp_list = []  # do_mpc.sampling.SamplingPlanner()
        Cloud.G_list = self.G_list.copy()
        Cloud.online_list = self.online_list.copy()
        return Cloud
    def schedule_processes(self,actions,failure_list,state_list,failure_pred_list,input_list):

        finished_early=0
        waiting_list=self.waiting_list.copy()
        for p in self.p_list:
            p.start()
            p.reduce_duration()

        for p in self.waiting_list:
            p.start()
            p.reduce_duration(real=True)

        finished_list=[]
        flag_indices=[p.get_system_number() for p in waiting_list if p.type()=="calculate_online_step"]

        next_projected_processes_list=self.p_list.copy()

        for p in self.waiting_list:
            if p.type()=="calculate_online_step":
                p.start()
                if p.is_alive():
                    if actions[p.get_system_number()]>0:
                        waiting_list.remove(p)
                        next_projected_processes_list.append(CloudProcessMockup(function=self.calculate_online_step,system_number=p.get_system_number(),num_workers=1,deadline=10000,x0=state_list[p.get_system_number()],u0=input_list[p.get_system_number()],q=Queue()))

                else:

                    waiting_list.remove(p)
                    next_projected_processes_list.append(
                            CloudProcessMockup(function=self.calculate_online_step, system_number=p.get_system_number(),
                                               num_workers=1, deadline=10000, x0=state_list[p.get_system_number()],
                                               u0=input_list[p.get_system_number()], q=Queue()))



            else:
                if actions[p.get_system_number()]>0:
                    if p.type()=="train_on_data":
                        waiting_list.remove(p)
                        next_projected_processes_list.append(
                            CloudProcessMockup(function=self.train_on_data, system_number=p.get_system_number(),
                                               num_workers=p.get_num_workers(),#actions[p.get_system_number()],
                                               deadline=p.get_deadline()))#np.floor((self.max_training -done)/ actions[p.get_system_number()])))
                    elif p.type()=="load_data":
                        if p.is_alive():
                            waiting_list.remove(p)
                            next_projected_processes_list.append(
                                CloudProcessMockup(function=self.load_data, system_number=p.get_system_number(), deadline=p.get_deadline(),
                                                   num_workers=1))
                        else:
                            waiting_list.remove(p)
                            next_projected_processes_list.append(
                                CloudProcessMockup(function=self.train_on_data, system_number=p.get_system_number(),
                                                   deadline=np.floor(self.max_training / actions[p.get_system_number()]),
                                                   num_workers=actions[p.get_system_number()]))
                    elif p.type() == "start_sampling" and p.get_system_number() not in flag_indices:
                        if p.get_num_workers() is not None:
                            if not p.is_alive():
                                waiting_list.remove(p)
                                next_projected_processes_list.append(
                                    CloudProcessMockup(function=self.load_data, system_number=p.get_system_number(),
                                                       deadline=self.max_load, num_workers=1))
                            else:
                                waiting_list.remove(p)
                                next_projected_processes_list.append(CloudProcessMockup(function=self.start_sampling, system_number=p.get_system_number(),
                                                                                        num_workers=p.get_num_workers(),#actions[p.get_system_number()],
                                                   deadline=p.get_deadline()))#np.floor((self.max_sampling-done)/ actions[p.get_system_number()])))
                        else:
                            waiting_list.remove(p)
                            next_projected_processes_list.append(
                                CloudProcessMockup(function=self.start_sampling, system_number=p.get_system_number(),
                                                   num_workers= actions[p.get_system_number()],
                                                   deadline=np.floor(self.max_sampling/ actions[p.get_system_number()])))

                else:

                    if p.type()=="train_on_data":
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

                    elif p.type()=="load_data":
                        waiting_list.remove(p)
                        waiting_list.append(
                            CloudProcessMockup(function=self.load_data, system_number=p.get_system_number(), deadline=p.get_deadline(),
                                               num_workers=1))
                    elif p.type() == "start_sampling" and p.get_system_number() not in flag_indices:
                        deadline = p.get_deadline()
                        num_workers = p.get_num_workers()
                        waiting_list.remove(p)
                        waiting_list.append(CloudProcessMockup(function=self.start_sampling, system_number=p.get_system_number(), num_workers=num_workers,
                                           deadline=deadline))


        for p in self.p_list:
            if p.type() == "train_on_data" and not p.is_alive()or p.type() == "train_on_data" and actions[p.get_system_number()]==0:
                if p.type() == "train_on_data" and actions[p.get_system_number()]==0 and p.is_alive():
                    finished_early+=p.get_deadline()
                finished_list.append(p.get_system_number())
                p.kill()
                for q in next_projected_processes_list:
                    if q.get_system_number()==p.get_system_number() and q.type()=="calculate_online_step":
                        q.kill()
                        next_projected_processes_list.remove(q)
                next_projected_processes_list.remove(p)
                continue

            if p.type()=="load_data" and not p.is_alive():
                if actions[p.get_system_number()]>0:
                    next_projected_processes_list.append(CloudProcessMockup(function=self.train_on_data,system_number=p.get_system_number(),num_workers=actions[p.get_system_number()],deadline=np.floor(self.max_training/actions[p.get_system_number()])))
                else:
                    waiting_list.append(CloudProcessMockup(function=self.load_data,system_number=p.get_system_number(),deadline=p.get_deadline(),num_workers=p.get_num_workers()))
                p.kill()
                next_projected_processes_list.remove(p)
                continue

            if p.type()=="start_sampling" and not p.is_alive():
                if actions[p.get_system_number()] > 0:
                    next_projected_processes_list.append(CloudProcessMockup(function=self.load_data,system_number=p.get_system_number(),deadline=self.max_load,num_workers=1))
                else:
                    waiting_list.append(
                        CloudProcessMockup(function=self.start_sampling, system_number=p.get_system_number(),deadline=p.get_deadline(),num_workers=p.get_num_workers()))
                p.kill()
                next_projected_processes_list.remove(p)

                continue
            if p.is_alive():

                if p.type()!="calculate_online_step":
                    if actions[p.get_system_number()] == 0:
                        next_projected_processes_list.remove(p)
                        waiting_list.append(p)
                continue
            p.kill()
            index=p.get_system_number()
            x0=state_list[index]
            u0=input_list[index]
            p.set_u0(u0)
            p.set_x0(x0)
        new_processes_list=[]
        new_processes_list = new_processes_list + [CloudProcessMockup(function=self.calculate_online_step,system_number=num, x0=state_list[k],u0=input_list[k],q=Queue(),deadline=10000,num_workers=1) for k,num in
                                         enumerate(failure_list) if actions[num]>=1]
        waiting_list=waiting_list+[CloudProcessMockup(function=self.calculate_online_step,system_number=num, x0=state_list[k],u0=input_list[k],q=Queue(),deadline=failure_pred_list[k],num_workers=1) for k,num in
                                         enumerate(failure_list) if actions[num]==0]
        new_processes_list = new_processes_list + [CloudProcessMockup(function=self.start_sampling,system_number=num,num_workers=actions[num],deadline=np.floor(self.max_sampling/actions[num])) for num in failure_list if actions[num]>=1]
        waiting_list=waiting_list+[CloudProcessMockup(function=self.start_sampling,system_number=num,deadline=1000) for num in failure_list if actions[num]==0]
        next_process_list=next_projected_processes_list+new_processes_list
        for p in waiting_list:
            if p.type()!= "calculate_online_step":
                p.set_deadline(p.get_deadline()+1)
        self.waiting_list=waiting_list.copy()
        self.p_list=next_process_list.copy()

        return next_process_list,waiting_list,finished_early
    def start_processes(self,duration_list=None):
        if duration_list is None:
            duration_list=[1 for p in self.p_list]
        assert(len(duration_list)==len(self.p_list))
        for k,p in enumerate(self.p_list):
            p.start(duration_list[k])
class cloudMockupPriority(cloudMockup):
    def __init__(self, parameters=None, use_both=False,scenario=None):  #
        super().__init__(parameters)
        if parameters is not None:
            self.max_process = parameters['max_process']  # multiprocessing.cpu_count()
            self.num_train_trajectories = parameters['num_train_trajectories']
            self.num_train_samples_per_trajectories = parameters['num_train_samples_per_trajectories']
            self.num_approx_feasible_set = parameters['num_approx_feasible_set']
            self.number_of_constraint_points = parameters['number_of_constraint_points']
            self.early_stopping = parameters['early_stopping']
            self.learning_rate = parameters['learning_rate']
            self.max_epoch = parameters['max_epoch']
            self.train_split = parameters['train_split']
            self.batch_size = parameters['batch_size']
            self.number_embedded = parameters['N_embedded']
            self.current_time = 0
            self.p_list = []
            self.waiting_list = []
            self.mpc_list = []
            self.simulator_list = []
            self.estimator_list = []
            self.current_net_list = []
            self.linear_list = []
            for k in range(self.number_embedded):
                path = parameters['system_list'][k]
                sys.path.append('path')
                model_path = path + '.template_model'
                mpc_path = path + '.template_mpc'
                simulator_path = path + '.template_simulator'
                template_model = importlib.import_module(model_path)
                template_mpc = importlib.import_module(mpc_path)
                template_simulator = importlib.import_module(simulator_path)
                if k > 0:
                    importlib.reload(template_model)
                    importlib.reload(template_mpc)
                    importlib.reload(template_simulator)
                model = template_model.template_model()
                mpc = template_mpc.template_mpc(model)
                linear = len(mpc.tvp_fun(0)['_tvp', -1].full()) > 2
                simulator = template_simulator.template_simulator(model)
                states = mpc.model.n_x
                inputs = mpc.model.n_u
                self.mpc_list.append(mpc)  # [template_mpc(model) for k in range(self.number_embedded)]
                self.simulator_list.append(
                    simulator)  # [template_simulator(model) for k in range(self.number_embedded)]
                self.estimator_list.append(
                    do_mpc.estimator.StateFeedback(model))  # [estimator for k in range(self.number_embedded)]
                self.current_net_list.append(
                    NeuralNetwork(states + inputs, inputs))  # for k in range(self.number_embedded))
                self.linear_list.append(linear)

            self.sp_list = []
            self.G_list = []
            self.online_list = []
        else:
            self.use_both = use_both
            self.p_list = []
            self.waiting_list = []
            if scenario:
                self.max_sampling = scenario['max_sampling']
                self.max_training = scenario['max_training']
                self.max_load=scenario['max_load']
                self.max_process = scenario['size_p']
                self.system_size=scenario['system_size']
            else:
                self.max_sampling=16
                self.max_training=16
                self.max_process=8
                self.max_load=2
    def schedule_processes(self, actions,failure_list,state_list,failure_pred_list,input_list):

        for p in self.p_list:
            p.reduce_duration()
        finished_list = []
        for p in self.waiting_list:
            p.start()
            p.reduce_duration(real=True)
        waiting_list=self.waiting_list.copy()
        next_projected_processes_list =[]
        running_processes_list = []
        necessary_processes_list = []
        old_processes_list=[]
        new_processes_list=[]

        for p in waiting_list:

            if p.type() == "load_data" and not p.is_alive():
                old_processes_list.append(p)
                next_projected_processes_list.append(
                    CloudProcessMockup(function=self.train_on_data, system_number=p.get_system_number()))
                p.kill()
                self.waiting_list.remove(p)
                continue

            if p.type() == "start_sampling":
                if not p.is_alive():
                    old_processes_list.append(p)
                    next_projected_processes_list.append(
                        CloudProcessMockup(function=self.load_data, system_number=p.get_system_number(),deadline=self.max_load))
                    p.kill()
                    self.waiting_list.remove(p)
                elif p.get_num_workers() is None:
                    self.waiting_list.remove(p)
                    new_processes_list.append(CloudProcessMockup(function=self.start_sampling,system_number=p.get_system_number(),deadline=1000))

                continue

            if p.type()=="calculate_online_step":

                if p.check_if_necessary():
                    self.waiting_list.remove(p)
                    necessary_processes_list.append(p)
                    p.set_deadline(1000)
                    p.set_num_workers(1)
                else:
                    self.waiting_list.remove(p)
                    new_processes_list.append(p)

        p_list=self.p_list.copy()
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
                    CloudProcessMockup(function=self.train_on_data, system_number=p.get_system_number()))
                p.kill()
                self.p_list.remove(p)
                continue

            if p.type() == "start_sampling" and not p.is_alive():
                old_processes_list.append(p)
                next_projected_processes_list.append(
                    CloudProcessMockup(function=self.load_data, system_number=p.get_system_number(),deadline=self.max_load))
                p.kill()
                self.p_list.remove(p)

                continue
            if p.is_alive() and not p.type()=="calculate_online_step":
                running_processes_list.append(p)
                self.p_list.remove(p)

                continue
            p.kill()
            if p in self.p_list:
                necessary_processes_list.append(p)
                self.p_list.remove(p)
            index = p.get_system_number()
            x0 = state_list[index]
            u0 = input_list[index]
            p.set_u0(u0)
            p.set_x0(x0)

        [p.set_deadline(1000) for p in necessary_processes_list]
        new_processes_list = new_processes_list + [
            CloudProcessMockup(function=self.calculate_online_step, system_number=num, x0=state_list[num],
                               u0=input_list[num], q=Queue(), deadline=failure_pred_list[k]) for k, num in
            enumerate(failure_list)]

        new_processes_list = new_processes_list + [
            CloudProcessMockup(function=self.start_sampling, system_number=num,deadline=1000) for num in failure_list]

        for p in necessary_processes_list:
            if p in new_processes_list:
                new_processes_list.remove(p)
        #for system in finished_list:
        #    self.online_list.remove(system)
        self.p_list=[]
        finished_early = self.priority_scheduling(new_processes_list, necessary_processes_list,
                                                      next_projected_processes_list, running_processes_list,old_processes_list.copy())

        for p in self.waiting_list:
            if p.type()!= "calculate_online_step":
                p.set_deadline(p.get_deadline()+1)
        next_process_list=self.p_list.copy()
        next_waiting_list=self.waiting_list.copy()
        return next_process_list,next_waiting_list,finished_early

    def priority_scheduling(self, new_p_list, necessary_p_list, next_p_list, running_p_list,old_processes):
        enough_space = True

        finished = []
        finished_early=0
        running_p_list=running_p_list+self.waiting_list.copy()#self.waiting_list.copy()
        self.waiting_list=[]
        #self.p_list = running_p_list.copy()
        if np.sum(np.asarray([p.get_num_workers() for p in running_p_list])) + len(necessary_p_list) > self.max_process:
            enough_space = False
        while not enough_space:
            if np.sum(np.asarray([p.get_num_workers() for p in running_p_list])) <= self.max_process - len(
                    necessary_p_list):
                enough_space = False
                #sample_list = [p for p in running_p_list if p.type() != "train_on_data"]
                #if sample_list == []:
                #    "Mark: sample list can also contain load_data"
                train_list = [p for p in running_p_list if p.type() == "train_on_data"]
                if train_list!=[]:
                    index = train_list.index(max([p.get_running_time() for p in train_list]))
                    p=train_list[index]

                    finished_early+=p.get_deadline()
                    finished.append(p.get_system_number())
                    found=False
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
                    self.waiting_list.append(p)

                running_p_list.remove(p)
            else:
                enough_space = True
        self.p_list = running_p_list.copy() + necessary_p_list.copy()
        N = np.sum(np.asarray([p.get_num_workers() for p in self.p_list]))
        wait_processes_list =  new_p_list.copy() + next_p_list.copy()#+wait_without.copy()#self.waiting_list.copy() +
        assignment_next_p=[0 for k in range(len(new_p_list))]+[k+1 for k in range(len(next_p_list))]
        num_wait_processes = len(wait_processes_list)

        if num_wait_processes <= self.max_process - N:
            simple_list=[p for p in wait_processes_list if
             p.type() != "train_on_data" and p.type() != "start_sampling"]
            [p.set_deadline(1000) if p.type()=="calculate_online_step" else p.set_deadline(self.max_load) for p in simple_list]
            [p.set_num_workers(num_workers=1) for p in simple_list]

            self.p_list = self.p_list +simple_list.copy()
            N = np.sum(np.asarray([p.get_num_workers() for p in self.p_list]))
            M = len([p for p in wait_processes_list if p.type() == "train_on_data" or p.type() == "start_sampling"])
            if M > 0:
                num_workers =min(int((self.max_process - N) / M),self.max_process-len(necessary_p_list))
                #done = [(np.floor(
                #    self.max_sampling / p.get_num_workers()) - p.get_deadline()) * p.get_num_workers() if p.type() == "start_sampling" or p.type() == "train_on_data" else 0
                #        for p
                #        in self.waiting_list]
                #done += [0 for k in range(len(new_p_list + next_p_list))]
                [p.set_num_workers(num_workers=num_workers) for p in wait_processes_list if
                 p.type() == "train_on_data" or p.type() == "start_sampling"]
                [p.set_deadline(np.ceil(self.max_sampling/ num_workers)) for k,p in enumerate(wait_processes_list) if
                 p.type() == "train_on_data" or p.type() == "start_sampling"]
                self.p_list = self.p_list + [p for p in wait_processes_list if
                                             p.type() == "train_on_data" or p.type() == "start_sampling"]

        else:
            N = np.sum(np.asarray([p.get_num_workers() for p in self.p_list]))
            number_of_slots = int(self.max_process - N)
            control_tasks = [p for p in wait_processes_list if p.type() == "calculate_online_step"]
            ass_control = [assignment_next_p[k] for k, p in enumerate(wait_processes_list) if p.type() == "calculate_online_step"]
            samples_tasks = [p for p in wait_processes_list if p.type() == "start_sampling"]
            ass_sampling =[assignment_next_p[k] for k,p in enumerate(wait_processes_list) if p.type() == "start_sampling"]
            load_tasks = [p for p in wait_processes_list if p.type() == "load_data"]
            ass_load= [assignment_next_p[k] for k,p in enumerate(wait_processes_list) if p.type() == "load_data"]
            training_task = [p for p in wait_processes_list if p.type() == "train_on_data"]
            ass_training = [assignment_next_p[k] for k, p in enumerate(wait_processes_list) if p.type() == "train_on_data"]
            sorted_waiting_list = control_tasks + training_task + load_tasks + samples_tasks
            assignment_next_p_sorted=ass_control+ass_training+ass_load+ass_sampling
            new_p_list=sorted_waiting_list[0:number_of_slots]
            [p.set_num_workers(num_workers=1) for p in new_p_list]
            [p.set_deadline(self.max_sampling) for p in new_p_list if p.type()=="start_sampling"]
            [p.set_deadline(self.max_training) for p in new_p_list if p.type() == "train_on_data"]
            [p.set_deadline(1000) for p in new_p_list if p.type() == "calculate_online_tasks"]
            [p.set_deadline(self.max_load) for p in new_p_list if p.type() == "load_data"]
            self.p_list = self.p_list +new_p_list
            new_waiting_list=sorted_waiting_list[number_of_slots:len(sorted_waiting_list)].copy()
            for index,k in enumerate(assignment_next_p_sorted[number_of_slots:len(sorted_waiting_list)]):
                if k>0:
                    new_waiting_list[index]=old_processes[k-1].copy()
            self.waiting_list = self.waiting_list+new_waiting_list.copy()

        return finished_early

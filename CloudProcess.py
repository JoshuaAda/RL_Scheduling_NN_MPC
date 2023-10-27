from multiprocessing import Process
from copy import deepcopy
import pickle
import numpy as np
import torch
import sys
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
class CloudProcess:
    def __init__(self,function,system_number,num_workers=None,q=None,x0=None,u0=None,deadline=None,startup_function=None,num=0,r_num=0):

        self._startup_function=startup_function
        self._function = function
        self._system_number=system_number
        self._num_workers=num_workers#system_1 if num_workers is None else num_workers#args if args is not None else system_0
        self._q=q
        self._p=None
        self._x0=x0
        self._u0=u0
        self._wait_duration=0
        self._duration=0
        self._deadline=deadline
        self._num = num
        self._r_num = r_num
        if self.type()=="start_sampling" and startup_function is not None:
            self._mpc,self._estimator,self._simulator,self._training_points,self._maximal,self._max = self._startup_function(self._system_number, self._num_workers)
            self._x0_list=[[self._training_points[:,k].reshape((4,1))] for k in range(np.size(self._training_points,1))]
            self._u0_list=[[0*np.ones((self._mpc.model.n_u,1))] for k in range(np.size(self._training_points,1))]
            self._started=False
            #for m in range(self.get_num_workers()):
            #    self._x0_list.append([self._training_points[:,k] for k in range(np.size(self._training_points,1)) if np.mod(k,self.get_num_workers())==m])

        if self.type()=="load_data":
            self._input_data=[]
            self._output_data=[]
        if self.type()=="train_on_data" and startup_function is not None:
            #self._criterion, self._optimizer, self._net, self._trainset, self._testset,self._maximal,self._learning_rate = self._startup_function(
            #    self._system_number, 1)
            self._started_training=True
            self._criterion, self._optimizer, self._net, self._trainloader, self._testloader, self._maximal, self._learning_rate = self._startup_function(
               self._system_number, 8)
            #self._started=False
            self._num = num
            self._num_iter=0
            self._end=False
            self._number_of_increased_loss=0
            self._old_test_loss=10000
    def copy(self):
        return CloudProcess(self._function,self._system_number,self._num_workers,self._q,self._x0,self._u0,self._deadline,self._startup_function,self._num,self._r_num)
    def copy_to_cloudmockup(self):
        mockup=CloudProcessMockup(str(self._function.__name__),self._system_number,self._num_workers,self._q,self._x0,self._u0,self._deadline)
        mockup.start()
        return mockup
    def type(self):
        return str(self._function.__name__)
    def start(self):
        if self._deadline>0:
            if self._q is not None:
                "control task"
                if self._x0 is not None:
                    self._q=self._function(self._system_number,self._x0,self._u0)
                    #self._p=Process(target=self._function,args=(self._system_number,self._x0,self._u0,self._q))
                #else:
                #    self._p = Process(target=self._function, args=(self._system_number,self._q))
            else:
                "sampling"
                if self.type()=="start_sampling":
                    #if self._num_workers!=0:
                        if not self._started:
                            deadline = self.get_deadline()
                            self._data = [[] for k in range(np.size(self._training_points,1))]
                            num_workers = self.get_num_workers()
                            runs = int(np.ceil(np.size(self._training_points,1) / num_workers))
                            self._last_max = int(deadline - (runs - 1) * self._max)
                            self._num=[k for k in range(num_workers)]
                            self._r_num=[0 for k in range(num_workers)]
                            self._started=True

                        #self._p = Process(target=self._function, args=(self._system_number,self._num_workers))
                        for k in range(self._num_workers):
                            if self._num[k]<self._maximal:
                                x0_list,u0_list=self._function(self._mpc,self._estimator,self._simulator,self._x0_list[self._num[k]][-1],self._u0_list[self._num[k]][-1])
                                #u=[u0[0] for u0 in u0_list]
                                [self._x0_list[self._num[k]].append(x0) for x0 in x0_list]
                                [self._u0_list[self._num[k]].append(u0) for u0 in u0_list]

                                if self._r_num[k]==self._max-1:
                                    directory = 'samples/' + str(self._system_number) + '/' + 'sample_' + str(
                                        self._num[k]) + '.pkl'
                                    data=dict()
                                    data['x']=self._x0_list[self._num[k]]
                                    data['u']=self._u0_list[self._num[k]]

                                    self._r_num[k] = 0
                                    self._num[k] += self.get_num_workers()
                                    pickle.dump(data, open(directory, "wb"))

                                self._r_num[k]+=1



                elif self.type()=="load_data":
                    "load data"
                    self._input_data,self._output_data=self._function(self._system_number,self._input_data,self._output_data,self._deadline)
                elif self.type()=="train_on_data":
                    if self._num_workers is not None:
                        for k in range(100):
                            if not self._end:
                                self._number_of_increased_loss,self._old_test_loss,self._num_iter,self._end=self._function(self._criterion,self._optimizer,self._net,self._trainloader,self._testloader,"cpu",self._system_number,self._number_of_increased_loss,self._old_test_loss,self._num_iter,self._num_workers,k)
                                print(k)
                                if self._end:
                                    self._deadline=1
                                    break
                            self._num+=1

            self._p=True
    def kill(self):
        if self.type() != "calculate_online_step":
            self._deadline=0
    def is_alive(self):
        if self.started():
            return self._deadline>0
        else:
            return False
    def should_alive(self):
        return self._deadline>0
    def started(self):
        return self._p is not None
    def get_value(self):
        assert hasattr(self, '_q')
        return self._q#.get()
    def set_deadline(self,deadline):
        self._deadline=deadline

    def get_deadline(self):
        return self._deadline

    def reduce_duration(self):
        if self._deadline>0:
            self._deadline-=1
    def get_system_number(self):
        return self._system_number
    def get_num_workers(self):
        return self._num_workers
    def set_num_workers(self,num_workers):
        self._num_workers=num_workers
    def set_x0(self,x0):
        self._x0=x0
    def set_u0(self,u0):
        self._u0=u0
    def update_wait_duration(self):
        self._wait_duration+=1
    def check_if_necessary(self):
        return self._deadline==0

    def update_running_time(self):
        self._deadline-=1
        #self._duration+=1
    def get_wait_duration(self):
        return self._deadline#self._wait_duration

    def get_running_time(self):
        return self._deadline
class CloudProcessMockup(CloudProcess):
    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memodict))
        return result
    def copy(self):
        new_p=CloudProcessMockup(self._function,self._system_number,num_workers=self._num_workers,q=self._q,x0=self._x0,u0=self._u0,deadline=self._deadline)
        new_p.start()
        return new_p
    def type(self):
        if isinstance(self._function,str):
            return self._function
        else:
            return str(self._function.__name__)
    def start(self,duration=1):
        if self._q is not None:
            "control task"
            if self._x0 is not None:
                self._p=ProcessMockup(target=self._function,args=(self._system_number,self._x0,self._u0,self._q))
            else:
                self._p = ProcessMockup(target=self._function, args=(self._system_number,self._q))
        else:
            "train and sampling"
            if self._num_workers is not None:
                self._p = ProcessMockup(target=self._function, args=(self._system_number,self._num_workers))
            else:
                "load data"
                self._p = ProcessMockup(target=self._function,args=(self._system_number,))
        self._p.start(self._deadline)
    def set_deadline(self,deadline):
        self._deadline=deadline

    def get_running_time(self):
        return self._deadline
    def get_deadline(self):
        return self._deadline
    def is_alive(self):
        return self._deadline>0
    def reduce_duration(self,real=False):
        dead=False
        if dead and not real:
            self._deadline=2
        self._deadline-=1
        self._p.reduce_duration(dead)
class ProcessMockup:
    def __init__(self,target,args):
        self._target=target
        self._args=args
        self._duration=3
    def start(self,duration=1):
        self._duration=duration

    def is_alive(self):
        if self._duration>0:
            return True
        else:
            return False
    def reduce_duration(self,dead):
        if dead:
            self._duration=2
        if self._duration>0:
            self._duration -= 1
    def kill(self):
        self._duration=0


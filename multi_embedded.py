import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from embedded import embedded
import sys
import importlib
import tikzplotlib
class multi_embedded:
    def __init__(self,number:int,current_states:list,g_list:list,mpc_list:list,m_list:list,linear_list:list)->None:
        self.fig=plt.figure()
        self.max_nx=max([mpc.model.n_x for mpc in mpc_list])
        self.max_nu=max([mpc.model.n_u for mpc in mpc_list])
        self.embedded_list=[embedded(current_states[k],g_list[k],mpc_list[k].bounds,mpc_list[k].model.n_x,mpc_list[k].model.n_u,mpc_list[k].tvp_fun(0),m_list[k],k,linear_list[k],mpc_list[k].model._rhs_fun) for k in range(number)]
        self._xtilde = [[] for k in range(len(self.embedded_list))]

    def failure(self)->(bool,list,list):
         failure_list=[x.failure() for k,x in enumerate(self.embedded_list)]
         failure_index_list=[]
         failure_pred_list=[]
         failure=False
         for k in range(len(failure_list)):
            if failure_list[k]>0:
                failure_index_list.append(k)
                failure_pred_list.append(failure_list[k])
                failure=True
         return (failure,failure_index_list,failure_pred_list)


    def update_system_parameters(self,template_list:list,num_list:list)->None:
        for k,num in enumerate(num_list):
            self.embedded_list[num].update_system_parameters(template_list[k])
            #self._mpc_list[num].set_tvp_fun(template_list[k])
    def get_state_of(self,num:int)->np.array:
        return self.embedded_list[num].current_state
    def get_input_of(self,num:int)->np.array:
        return self.embedded_list[num].current_input
    def get_states_of(self,num_list:list)->list:
        return [self.get_state_of(num) for num in num_list]
    def get_all_states(self)->list:
        return [self.get_state_of(num) for num in range(len(self.embedded_list))]
    def get_all_inputs(self)->list:
        return [self.get_input_of(num) for num in range(len(self.embedded_list))]

    def calculate_next_step_offline(self,noise:np.array)->list:
        for k in range(len(self.embedded_list)):
            self.embedded_list[k].calculate_next_step_offline(noise[k])
    def get_next_step_online(self,u0_list:list,noise_list:list,online_list:list)->None:
        for k in range(len(self.embedded_list)):
            if k in online_list:
                index=online_list.index(k)
                self.embedded_list[k].get_next_step_online(u0_list[index],noise_list[k])
            else:
                self.embedded_list[k].calculate_next_step_offline(noise_list[k])
    def update_network(self,update_list:list)->None:
        for num in update_list:
            self.embedded_list[num].update_network(num)
    def visualize_system(self)->None:
        plt.ion()
        n=len(self.embedded_list)
        ax = plt.subplot(2, 1, 1)
        ax.clear()
        track=0
        track_rel=0
        for k,embedded_sys in enumerate(self.embedded_list):
            time = np.linspace(0, len(embedded_sys.store_states) * 0.5, len(embedded_sys.store_states))
            x1=np.asarray(embedded_sys.store_states)[:, 0]
            xtilde=self.embedded_list[k].x_tilde[0]
            self._xtilde[k].append(xtilde)
            xtilde=np.asarray(self._xtilde[k])
            track+=np.sum(np.abs(x1-xtilde))
            track_rel+=np.sum(np.abs(x1-xtilde))/(np.size(x1,0))
            plt.plot(time, x1, label="System " + str(k+1),alpha=0.5)
            plt.xlabel('Time [s]')
            plt.ylabel('x' + str(1))
            plt.legend()

        ax = plt.subplot(2, 1, 2)
        ax.clear()
        for k, embedded_sys in enumerate(self.embedded_list):
            time = np.linspace(0, len(embedded_sys.store_states) * 0.5, len(embedded_sys.store_states))
            plt.plot(time, np.asarray(embedded_sys.store_inputs).squeeze()[1:],alpha=0.5, label="System" + str(k + 1))
            plt.xlabel('Time [s]')
            plt.ylabel('u')

        np.savez("tracking22.npz",tracking_error=track,tracking_rel=track_rel)
        self.fig.show()
        tikzplotlib.save("image_all.tikz")
        plt.savefig("image_all.svg")
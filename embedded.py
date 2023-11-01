import torch
import matplotlib.pyplot as plt
import numpy as np
import do_mpc
import os
import importlib
from model_class import NeuralNetwork
##### Simulation class for one embedded system
class embedded:
    """
    Class of the system on the premised with an explicit embedded controller approximation
    """
    def __init__(self,x0:np.array,G,mpc_bounds,nx,nu,template,number_of_pred:int,system_number:int,linear:bool,rhs_fun):
        # Initialize parameters of the system
        model_path = 'systems.system_'+str(system_number)+ '.template_model'
        simulator_path = 'systems.system_'+str(system_number)+ '.template_simulator'
        template_model=importlib.import_module(model_path)
        template_simulator = importlib.import_module(simulator_path)
        model=template_model.template_model()
        simulator = template_simulator.template_simulator(model)
        estimator = do_mpc.estimator.StateFeedback(model)
        """
        Get neural network model
        """
        PATH='./nets_start/'+str(system_number)+'/model_net.pth'
        self.nx=nx
        self.nu=nu
        self.linear=linear
        self.system_number=system_number
        self.rhs_fun=rhs_fun

        net = NeuralNetwork(nx+nu,nu)
        if not os.path.exists('./nets_start/' + str(system_number)):
            os.mkdir('./nets_start/'+str(system_number))

        else:
            net.load_state_dict(torch.load(PATH))
        torch.save(net.state_dict(), PATH)
        self.current_net=net
        self.estimator=estimator
        self.simulator=simulator
        self.current_state=x0
        self.current_input=np.zeros((nu,1))
        self.estimator.x0=x0
        self.simulator.x0=x0
        self.current_time=0
        self.store_states=[]
        self.store_inputs=[5*np.ones((nu,1))]
        self.current_state_space=G
        #self.fig=plt.figure()
        self.bounds=dict()
        self.bounds['x_left']=mpc_bounds['lower','_x','x']
        self.bounds['x_right'] = mpc_bounds['upper', '_x', 'x']
        self.bounds['u_left'] = mpc_bounds['lower', '_u', 'u']
        self.bounds['u_right'] = mpc_bounds['upper', '_u', 'u']
        self.controlled_online=False
        self.update_system_parameters(template)
        self.number_of_pred=number_of_pred

    def set_number_of_pred(self, number_of_pred):
        # set number of predictions
        self.number_of_pred=number_of_pred

    def update_system_parameters(self, template):
        # Update system parameters
        if self.linear:
            self.template = template
            self.x_tilde = template['_tvp', -1, 'x_tilde'].full()
            self.A = template['_tvp', -1, 'A'].full()
            self.B = template['_tvp', -1, 'B'].full()
            self.Q = template['_tvp', -1, 'Q'].full()
            self.K = template['_tvp', -1, 'K'].full()
            self.P = template['_tvp', -1, 'P'].full()
        #self.mpc.set_tvp_fun(template)
    def update_network(self,system_number):
        # Update network from training process thread stored in ./model_net.pth
        self.controlled_online=False
        net = NeuralNetwork(self.nx+self.nu,self.nu)
        PATH="./nets/"+str(system_number)+"/model_net.pth"
        net.load_state_dict(torch.load(PATH))
        self.current_net=net
    def calculate_next_step_offline(self,noise):
        # simulate the next step of the system with the embedded controller value as input
        self.store_states.append(self.current_state)
        last_input=self.store_inputs[-1]
        training_state=np.concatenate((self.current_state,np.array(last_input).reshape((self.nu,1))))
        u0 = self.current_net(torch.Tensor(training_state).T).detach().numpy()  # mpc.make_step(x0)
        if u0>self.bounds['u_right']:
            u0=self.bounds['u_right'].full()
        y_next = self.simulator.make_step(u0,w0=noise)
        self.current_state = self.estimator.make_step(y_next)
        self.current_input=u0
        self.store_inputs.append(u0)
        self.current_time+=0.5


    def get_next_step_online(self,u0,noise):
        # simulate the next step of the system with the cloud controller value as input
        self.store_states.append(self.current_state)
        y_next = self.simulator.make_step(u0,w0=noise)
        self.current_state = self.estimator.make_step(y_next)
        self.current_input = u0
        self.store_inputs.append(u0)


    def failure(self):
        # embedded feasibility and constraints multi-step prediction
        if self.controlled_online:
            return 0
        last_input = self.store_inputs[-1]
        training_state = np.concatenate((self.current_state, np.array(last_input).reshape(len(last_input), 1)))
        input = self.current_net(torch.Tensor(training_state).T).detach().numpy()
        if input > self.bounds['u_right']:
            input = self.bounds['u_right'].full()
        elif input < self.bounds['u_left']:
            input = self.bounds['u_left'].full()
        if self.linear:
            new_state = self.A@self.current_state+self.B@input
        else:
            new_state=self.rhs_fun(self.current_state,input,0,0,0,0).full()
        fail=self.failure_prediction(new_state,input)
        if fail:
            print(new_state)
            self.controlled_online=True
            return 1
        for k in range(1,self.number_of_pred):
            if self.linear:
                new_state = self.A @ self.current_state + self.B @ input
            else:
                new_state = self.rhs_fun(self.current_state, input, 0, 0, 0, 0).full()
            training_state=np.concatenate((new_state,input))
            input = self.current_net(torch.Tensor(training_state).T).detach().numpy()
            if input > self.bounds['u_right']:
                input = self.bounds['u_right']
            elif input < self.bounds['u_left']:
                input=self.bounds['u_left']
            fail = self.failure_prediction(new_state, input)
            if fail:
                print(k)
                print(input)
                print(new_state)
                self.controlled_online=True
                return k
        return 0
    def failure_prediction(self,new_state,input):
        # preditction of violations for one state in the multi-step prediction
        input_failure=False
        if self.space_violation(input.T,self.bounds['u_left'],self.bounds['u_right']):
            input_failure=True
            print('input violation expected!')
        state_failure = False


        if self.space_violation(new_state, self.bounds['x_left'], self.bounds['x_right']):
            state_failure=True
            print('State violation expected!')

        #print(new_state.T@self.current_state_space@new_state>1)
        state_train_failure=False
        if new_state.T@self.current_state_space@new_state>1:#self.space_violation(new_state, self.current_state_space_left, self.current_state_space_right):
            state_train_failure=True
            print('Feasible set violation expected!')



        cost_failure=False
        #if self.current_state.T@self.Q@self.current_state>system_4:
        #    cost_failure=True
        #    print('Cost too bad!')


        if input_failure or state_failure or cost_failure or state_train_failure:
            #print('Failure')
            return True
        else:
            return False




    def space_violation(self,value,left,right):
        # constraint violation helper function
        if any([value[k]>right[k] or value[k]<left[k] for k,val in enumerate(value)]):
            print('Failure')
            return True
        else:
            return False
    def visualize_system(self,fig):


        time = np.linspace(0, len(self.store_states) * 0.5, len(self.store_states))
        n=2
        for k in range(self.nx):#len(self.store_states[system_0])):
            ax = fig.add_subplot(k,1,1)
            plt.plot(time,np.asarray(self.store_states)[:, k],label="System"+str(self.system_number))
            plt.xlabel('Time [s]')
            plt.ylabel('x'+str(k))




        plt.subplot(212)
        input_array=np.zeros((len(time),np.shape(self.store_inputs[0])[0]))
        for m in range(np.shape(self.store_inputs[0])[0]):
            for k in range(len(time)):
                input_array[k,m]=self.store_inputs[k][m]
            plt.plot(time,input_array[:,m])
            plt.xlabel('Time [s]')
            plt.ylabel('u'+str(m+1))
            plt.legend(['System system_0', 'System system_1'])


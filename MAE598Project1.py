# overhead

import logging
import math
import random
import numpy as np
import time
import torch as t
import torch.nn as nn
from torch import optim
import torch
from torch.nn import utils
import matplotlib.pyplot as plt

#logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# environment parameters
# values below are normalized to prevent loss from "exploding" this way reactions are minimized to allow for convergence
# with a non zero initial value
FRAME_TIME = .5  # time interval (s)
GRAVITY_ACCEL = 9.8/1000  # gravity constant (km/s^2)
ROTATION_ACCEL = 9/1000  # rotation constant rad/s^2
BOOST_ACCEL = 12/1000 # Acceleration provided by thrust constant (km/s^2)


class Dynamics(nn.Module):

    def __init__(self):
        super(Dynamics, self).__init__()

    @staticmethod
    def forward(state, action):
        """
        action: thrust or no thrust, Rot_Accel
        state[0] = y
        state[1] = y_dot
        state[2] = x
        state[3] = x_dot
        state[4] = theta
        """

        # Here the dynamic equations are split into matricies of tensor form so that values can be updated without overwriting
        # a computational position with a value. This allows the tensor to be continuously differentiated at each step.

        delta_state_gravity = t.tensor([0, 0.5 * GRAVITY_ACCEL * FRAME_TIME, 0., 0., 0.]) # Setting up influence of gravity
        # on y velocity. Only in the position corresponding to velocity in y direction because it has no influence on the other
        # state variables

        #Drag force- F = 0.5 * rho_air * v^2 * Cd * Cross-sectional Area
        # Cross sectional area taken to = 1m^2 , density of air = 1.225kg/m^3
        #Drag coefficient assumed to be 0.3 based on average value
        Cd = 0.000003 # drag value significantly lowered since simulation is very large scale
        rho = 1225000000 #kg/km^3
        A = .000001  #km^2


        sign= t.sign(state) #extract signs of state values to make sure drag force will be in opposite direction of velocity
        delta_state_drag = t.tensor([0, sign[2] * 0.5 * rho * state[2] ** 2 * Cd * A, 0., sign[4] * 0.5 * rho * state[4] ** 2 * Cd * A, 0.])



       #Need to use torch. operations on the tensors

        delta_dot_state = t.zeros((5, 2))
        delta_dot_state[1, 0] = torch.cos(state[4]) * FRAME_TIME
        delta_dot_state[3, 0] = -torch.sin(state[4]) * FRAME_TIME
        delta_dot_state[4, 1] = 1

        delta2_state = t.zeros(2)
        delta2_state[0] = 1 * action[0] * BOOST_ACCEL
        delta2_state[1] = -1 * action[1] * ROTATION_ACCEL
        delta_state = t.matmul(delta_dot_state, delta2_state)  # issue is probably arising from indexin state[4] here.

        # Update velocity
        state = state + delta_state - delta_state_gravity - delta_state_drag# Updating velocity to update all other states below

        # Update state
        # Note: Same as above. Use operators on matrices/tensors as much as possible. Do not use element-wise operators as they are considered inplace.
        step_mat = t.tensor([[1., FRAME_TIME, 0., 0., 0.],
                             [0., 1., 0., 0., 0.],
                             [0., 0., 1., FRAME_TIME, 0.],
                             [0., 0., 0., 1., 0.],
                             [0., 0., 0., 0., 1.]])
        state = t.matmul(step_mat, state) # yeilds updated state matrix 

        return state


# a deterministic controller
# Note:
# 0. You only need to change the network architecture in "__init__"
# 1. nn.Sigmoid outputs values from 0 to 1, nn.Tanh from -1 to 1
# 2. You have all the freedom to make the network wider (by increasing "dim_hidden") or deeper (by adding more lines to nn.Sequential)
# 3. Always start with something simple

class Controller(nn.Module):

    def __init__(self, dim_input, dim_hidden, dim_output):
        """
        dim_input: # of system states
        dim_output: # of actions
        dim_hidden: up to you
        """
        super(Controller, self).__init__()
        self.network = nn.Sequential(  #nn.Sequential houses a list of commands and feeds an input through each command
            nn.Linear(dim_input, dim_hidden),  # and uses the output as the input for the next commend
            nn.Sigmoid(),
            nn.Linear(dim_hidden, dim_output), # nn.Linear(n,m) assigns estimated weights to values in n and returns a
            nn.Tanh(),                       # batch that has size of m so when useing them here in nn.sequential it
            nn.Linear(dim_output, dim_hidden), # is important to make sure m1 = n2 to prevent sizing issues
            nn.Tanh(),
        ) # more layers are added to improve convergence using to better "weight" distribution

    def forward(self, state):
        action = self.network(state)
        return action


# the simulator that rolls out x(1), x(2), ..., x(T)
# Note:
# 0. Need to change "initialize_state" to optimize the controller over a distribution of initial states
# 1. self.action_trajectory and self.state_trajectory stores the action and state trajectories along time

class Simulation(nn.Module):

    def __init__(self, controller, dynamics, T):
        super(Simulation, self).__init__()
        self.state = self.initialize_state()
        self.controller = controller
        self.dynamics = dynamics
        self.T = T
        self.action_trajectory = []
        self.state_trajectory = []

    def forward(self, state):
        self.action_trajectory = []
        self.state_trajectory = []
        for _ in range(T):
            action = self.controller.forward(state)
            state = self.dynamics.forward(state, action)
            self.action_trajectory.append(action)
            self.state_trajectory.append(state)
        return self.error(state)

    @staticmethod
    def initialize_state():
        state = [2, 0., 1, 0., .24]  # TODO: need batch of initial states
        return t.tensor(state, requires_grad=False).float()

    def error(self, state):
        return torch.mean(state ** 2) # Using mean squared error to get loss values faster



class Optimize:
    def __init__(self, simulation):
        self.simulation = simulation
        self.parameters = simulation.controller.parameters()
        self.optimizer = optim.SGD(self.parameters, lr=0.01)
        #lr = .05 is too large, causes "overstepping"
        #best lr value found from experimentation is .01


    def step(self):
        def closure():
            loss = self.simulation(self.simulation.state)
            self.optimizer.zero_grad()
            loss.backward()
            return loss

        self.optimizer.step(closure)
        return closure()

    def train(self, epochs):
        for epoch in range(epochs):
            loss = self.step()
            print('[%d] loss: %.3f' % (epoch + 1, loss))
            self.visualize()

    def visualize(self):
        data = np.array([self.simulation.state_trajectory[i].detach().numpy() for i in range(self.simulation.T)])
        y = data[:, 0]           # this section extracts all of the data to be plotted in a graph 
        y_dot = data[:, 1]
        x = data[:, 2]
        x_dot = data[:, 3]
        theta = data[:, 4]
        plt.plot(y, label="Height", linestyle="-")
        plt.plot(y_dot, label="Vertical Velocity", linestyle="--")
        plt.plot(x, label="Horizontal location", linestyle="-.")
        plt.plot(x_dot,  label="Horizontal velocity", linestyle=":")
        plt.plot(theta, label="Orientation", linestyle=":")
        plt.legend()
        plt.show()


# Now it's time to run the code!

T = 100 # number of time steps increasing T and reducing time frame leads to faster convergence but takes longer to iterate
dim_input = 5  # state space dimensions
dim_hidden = 14  # latent dimensions
dim_output = 2  # action space dimensions
d = Dynamics()  # define dynamics
c = Controller(dim_input, dim_hidden, dim_output)  # define controller
s = Simulation(c, d, T)  # define simulation
o = Optimize(s)  # define optimizer
o.train(30)  # solve the optimization problem


# notes: Need to use small learning rate, .01 had best convergence. Since learning rate is small, set of time (time interval * T )
#needs to be small or the initial loss is too large to converge in 50 iterations. Latent dimensions should be chosen based on complexity of
#the problem. If there are too many latent dimensions, the solution will not converge since there are too many dimensions characterizing
# the behavior of the problem.  

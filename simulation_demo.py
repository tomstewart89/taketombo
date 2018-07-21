"""Simple test script for taketombo
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import time
import copy

from dynamic_model import ModelParam, DynamicModel, ModelState

print_sim_time = False
plot_visualization = True
plot_states = True

# create parameter struct
param = ModelParam()

# initial state
x0 = ModelState()
hover_omega = np.sqrt(0.5 * (param.m1 + param.m2 + param.m3 + 2 * param.mp) * param.g / param.c_t)
# x0.beta_dot = np.array([hover_omega, -hover_omega])

# instantiate model
model = DynamicModel(param, x0)

# simulation time step
dt = 0.05

# prepare simulation
max_sim_time = 2.5
sim_time = 0
sim_time_vec = [sim_time]
state_vec = [copy.copy(model.state)]
start_time = time.time()

if plot_visualization:
    fig = plt.figure(0)
    ax = fig.add_subplot(111, projection='3d')

# simulate
while sim_time < max_sim_time:
    if plot_visualization:
        # get visualization
        vis = model.get_visualization()

        plt.cla()

        # plot bodies
        for v in [vis['upper_body'], vis['middle_body'], vis['lower_body']]:
            collection = Poly3DCollection(v, linewidths=1, edgecolors='k', alpha=0.5)
            collection.set_facecolor('white')
            ax.add_collection3d(collection)

        # plot thrust vectors
        ax.quiver(*vis['thrust1'][0], *vis['thrust1'][1], colors='red')
        ax.quiver(*vis['thrust2'][0], *vis['thrust2'][1], colors='blue')

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        pos = model.state.pos
        range = 0.1
        ax.set_xlim3d(pos[0] - range, pos[0] + range)
        ax.set_ylim3d(pos[1] - range, pos[1] + range)
        ax.set_zlim3d(pos[2] - range, pos[2] + range)
        plt.show(block=False)
        time_passed = time.time() - start_time
        plt.pause(max(dt - time_passed, 0.001))

        start_time = time.time()

    # simulate one time step
    model.simulate_step(dt, np.array(
        [0.0001, -0.0001, 1.11 * hover_omega, -1.1 * hover_omega]))
    sim_time += dt

    # save states as matrix, sim_time and inputs as lists
    state_vec.append(copy.copy(model.state))
    sim_time_vec.append(sim_time)

    if print_sim_time:
        print('sim_time: {0:.3f} s'.format(sim_time))

if plot_states:
    plt.figure()
    plt.plot(sim_time_vec, [state.beta_dot[0] for state in state_vec], label='beta_1_dot')
    plt.plot(sim_time_vec, [state.beta_dot[1] for state in state_vec], label='beta_2_dot')
    plt.xlabel('time [s]')
    plt.ylabel('propeller speeds [rad/s]')
    plt.legend()
    plt.title('propeller speeds')

    plt.figure()
    plt.plot(sim_time_vec, [state.alpha[0] for state in state_vec], label='alpha_x')
    plt.plot(sim_time_vec, [state.alpha[1] for state in state_vec], label='alpha_y')
    plt.xlabel('time [s]')
    plt.ylabel('servo angles [rad]')
    plt.legend()
    plt.title('servo angles')
    plt.show(block=True)

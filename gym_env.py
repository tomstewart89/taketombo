import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import time
import copy
import gym
from gym.core import Env

from dynamic_model import ModelParam, DynamicModel, ModelState

class TaketomboEnv(Env):
    def __init__(self):
        self.x0 = ModelState()
        self.goal_state = ModelState()
        self.goal_state.pos = np.array([0.5,0.,0.1])
        
        self.param = ModelParam()
        self.model = DynamicModel(self.param, self.x0)
        self.vis = self.model.get_visualization()

        self.max_sim_time = 2.5
        self.sim_time = 0.
        self.dt = 0.05
        self.start_time = time.time()

        self.fig = plt.figure(0)
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.CLOSE_ENOUGH = 0.05 

        high = np.array([self.param.max_servo_angle, self.param.max_servo_angle, self.param.max_rotor_speed, -self.param.max_rotor_speed])
        low = np.array([self.param.min_servo_angle, self.param.min_servo_angle, self.param.min_rotor_speed, self.param.min_rotor_speed])
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

        high = np.inf * np.ones(self.observation().shape)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
    
    def step(self, action):
        self.model.simulate_step(self.dt, action)
        self.sim_time += self.dt

        distance_to_goal = self.compute_distance_to_goal(self.model.state, self.goal_state)

        done = self.sim_time > self.max_sim_time or distance_to_goal < self.CLOSE_ENOUGH
        reward = distance_to_goal - self.prev_distance

        return self.observation(), reward, done, {}

    def compute_distance_to_goal(self, achieved_goal, desired_goal):
        return np.linalg.norm(achieved_goal.pos - desired_goal.pos)

    def observation(self):
        return np.hstack([self.model.state.x, self.model.state.pos, self.goal_state.pos])

    def reset(self):
        self.model.state.set_state(self.x0.x)
        self.prev_distance = self.compute_distance_to_goal(self.model.state, self.goal_state)
        return self.model.state

    def render(self, mode='human'):
        vis = self.model.get_visualization()
        plt.cla()

        # plot bodies
        for v in [vis['upper_body'], vis['middle_body'], vis['lower_body']]:
            collection = Poly3DCollection(v, linewidths=1, edgecolors='k', alpha=0.5)
            collection.set_facecolor('white')
            self.ax.add_collection3d(collection)

        # plot thrust vectors
        self.ax.quiver(*vis['thrust1'][0], *vis['thrust1'][1], colors='red')
        self.ax.quiver(*vis['thrust2'][0], *vis['thrust2'][1], colors='blue')

        self.ax.set_xlabel('x [m]')
        self.ax.set_ylabel('y [m]')
        self.ax.set_zlabel('z [m]')
        pos = self.model.state.pos
        range = 0.1
        self.ax.set_xlim3d(pos[0] - range, pos[0] + range)
        self.ax.set_ylim3d(pos[1] - range, pos[1] + range)
        self.ax.set_zlim3d(pos[2] - range, pos[2] + range)
        plt.show(block=False)
        time_passed = time.time() - self.start_time
        plt.pause(max(self.dt - time_passed, 0.001))

        self.start_time = time.time()

    def close(self):
        pass


if __name__ == '__main__':
    env = TaketomboEnv()

    while 1:
        frame = score = 0
        obs = env.reset()

        while 1:
            a = env.action_space.sample()
            obs, r, done, _ = env.step(a)
            score += r
            frame += 1

            # if not env.render("human"): continue
            if not done: continue

            print("score=%0.2f in %i frames" % (score, frame))
            break


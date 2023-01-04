import glob
import os
import sys
from collections import namedtuple

try:
    sys.path.append('D:/Programs/Carla/CARLA_0.9.13/WindowsNoEditor/PythonAPI/carla')
except IndexError:
    pass

import carla
import torch
import csv
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import random
import time
import numpy as np
import cv2
import math
import gym
import pandas as pd
import carla_env

IM_WIDTH = 80
IM_HEIGHT = 60
SHOW_PREVIEW = True

SECOND_PER_EPISODE = 10


torch.cuda.empty_cache()
use_cuda = torch.cuda.is_available()
print(use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")
filePath = 'all.csv'


# class Car_Env():
#     SHOW_CAM = SHOW_PREVIEW
#     STEER_AMT = 1.0
#     im_width = IM_WIDTH
#     im_height = IM_HEIGHT
#     front_camera = None
#
#     def __init__(self):
#         self.client = carla.Client('localhost', 2000)
#         self.client.set_timeout(10.0)
#         self.world = self.client.get_world()
#         self.blueprint_library = self.world.get_blueprint_library()
#         self.model_3 = self.blueprint_library.filter('model3')[0]
#
#     def reset(self):
#         self.collision_hist = []
#         self.radar_hist = []
#         self.actor_list = []
#         self.transform = self.world.get_map().get_spawn_points()[100]  # spwan_points共265个点，选第一个点作为初始化小车的位置
#         self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
#
#         self.actor_list.append(self.vehicle)
#
#         self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
#         self.rgb_cam.set_attribute('image_size_x', f'{self.im_width}')
#         self.rgb_cam.set_attribute('image_size_y', f'{self.im_height}')
#         self.rgb_cam.set_attribute('fov', f'110')
#
#         transform = carla.Transform(carla.Location(x=2.5, z=0.7))
#         self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
#         self.actor_list.append(self.sensor)
#         self.sensor.listen(lambda data: self.process_img(data))
#
#         self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
#
#         time.sleep(4)
#
#         # collision sensor
#         colsensor = self.blueprint_library.find('sensor.other.collision')
#         self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
#         self.actor_list.append(self.colsensor)
#         self.colsensor.listen(lambda event: self.collision_data(event))
#
#         # target_transform 定义驾驶目的地坐标
#         self.target_transform = self.world.get_map().get_spawn_points()[101]
#         self.target_dis = self.target_transform.location.distance(self.vehicle.get_location())
#
#         while self.front_camera is None:
#             time.sleep(0.01)
#
#         self.episode_start = time.time()
#         self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
#
#         return self.front_camera
#
#     def collision_data(self, event):
#         self.collision_hist.append(event)
#
#     def radar_data(self, mesure):
#         self.radar_hist.append(mesure)
#
#     def process_img(self, image):
#         i = np.array(image.raw_data)
#         i2 = i.reshape((self.im_height, self.im_width, 4))
#         i3 = i2[:, :, : 3]
#         if self.SHOW_CAM:
#             cv2.imshow("", i3)
#             cv2.waitKey(1)
#         self.front_camera = i3
#         return i3 / 255.0
#
#     def step(self, action):
#         last_dis = self.target_dis
#         if action == 0:
#             self.vehicle.apply_control(
#                 carla.VehicleControl(throttle=1.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False))
#         elif action == 1:
#             self.vehicle.apply_control(
#                 carla.VehicleControl(throttle=0.5, steer=-1, brake=0.0, hand_brake=False, reverse=False))
#         elif action == 2:
#             self.vehicle.apply_control(
#                 carla.VehicleControl(throttle=0.5, steer=1, brake=0.0, hand_brake=False, reverse=False))
#         elif action == 4:
#             self.vehicle.apply_control(
#                 carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.5, hand_brake=False, reverse=False))
#         else:
#             self.vehicle.apply_control(
#                 carla.VehicleControl(throttle=1.0, steer=0.0, brake=0.0, hand_brake=False, reverse=True))
#
#         self.target_dis = self.target_transform.location.distance(self.vehicle.get_location())
#
#         v = self.vehicle.get_velocity()
#         kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
#         if len(self.collision_hist) != 0:
#             done = True
#             reward = -200
#         elif last_dis < self.target_dis:  # 距离目标越来越远了
#             done = False
#             reward = -1
#         else:
#             done = False
#             reward = 1
#
#         if self.episode_start + SECOND_PER_EPISODE < time.time():
#             done = True
#         time.sleep(1)
#         return self.front_camera, reward, done, None


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from torch import FloatTensor

Tensor = FloatTensor

EPSILON = 0.9  # epsilon used for epsilon greedy approach
GAMMA = 0.9
TARGET_NETWORK_REPLACE_FREQ = 100  # How frequently target netowrk updates
MEMORY_CAPACITY = 200
BATCH_SIZE = 32
LR = 0.01  # learning rate


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(PolicyNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)

        # uniform_将tensor用从均匀分布中抽样得到的值填充。参数初始化
        self.linear3.weight.data.uniform_(-init_w, init_w)
        # 也用用normal_(0, 0.1) 来初始化的，高斯分布中抽样填充，这两种都是比较有效的初始化方式
        self.linear3.bias.data.uniform_(-init_w, init_w)
        # 其意义在于我们尽可能保持 每个神经元的输入和输出的方差一致。
        # 使用 RELU（without BN） 激活函数时，最好选用 He 初始化方法，将参数初始化为服从高斯分布或者均匀分布的较小随机数
        # 使用 BN 时，减少了网络对参数初始值尺度的依赖，此时使用较小的标准差(eg：0.01)进行初始化即可

        # 但是注意DRL中不建议使用BN

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0]


class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3,
                 decay_period=10000):  # decay_period要根据迭代次数合理设置
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class LSTMPrediction(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMPrediction, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


class DDPG(object):
    def __init__(self, action_dim, state_dim, hidden_dim):
        super(DDPG, self).__init__()
        self.action_dim, self.state_dim, self.hidden_dim = action_dim, state_dim, hidden_dim
        self.batch_size = 50
        self.gamma = 0.99
        self.min_value = -np.inf
        self.max_value = np.inf
        self.soft_tau = 2e-2
        self.replay_buffer_size = 10000
        self.value_lr = 0.001
        self.policy_lr = 0.0001

        self.value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

        self.target_value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)

        self.value_criterion = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

        # ICM
        # self.predict_module = nn.Linear(state_dim + action_dim, state_dim).to(device)
        # self.inv_predict_module = nn.Linear(state_dim + state_dim, action_dim).to(device)
        self.predict_module = LSTMPrediction(state_dim + action_dim, hidden_dim, state_dim).to(device)
        self.inv_predict_module = LSTMPrediction(state_dim + state_dim, hidden_dim, action_dim).to(device)

    def ddpg_update(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        # ICM
        pred_next_state = self.predict_module(torch.cat((state, action), 1))
        pred_action = self.inv_predict_module(torch.cat((state, next_state), 1))
        forward_loss = F.mse_loss(pred_next_state, next_state)
        inverse_loss = F.cross_entropy(pred_action, action)
        intrinsic_reward = 0.1 * forward_loss
        reward += intrinsic_reward
        forward_loss.backward()
        inverse_loss.backward()

        policy_loss = self.value_net(state, self.policy_net(state))
        policy_loss = -policy_loss.mean()

        next_action = self.target_policy_net(next_state)
        target_value = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, self.min_value, self.max_value)

        value = self.value_net(state, action)
        value_loss = self.value_criterion(value, expected_value.detach())

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )

        return intrinsic_reward.item()



def main():
    TASK_MODE = 'Lane'
    params = {
        'dt': 0.1,  # time interval between two frames
        'port': 2000,  # connection port
        'task_mode': TASK_MODE,  # mode of the task, [random, roundabout (only for Town03)]
        'max_time_episode': 1000,  # maximum timesteps per episode
        'obs_size': [160, 100],  # screen size of bird-eye render
        'desired_speed': 8,  # desired speed (m/s)
        'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle

        'max_past_step': 1,  # the number of past steps to draw
        'max_waypt': 12,  # maximum number of waypoints
        'obs_range': 32,  # observation range (meter)
        'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
        'd_behind': 12,  # distance behind the ego vehicle (meter)
        'display_route': False,  # whether to render the desired route

        'number_of_vehicles': 10,
        'number_of_walkers': 3,
        'out_lane_thres': 2.0,  # threshold for out of lane

        'code_mode': 'train',
        'discrete': False,  # whether to use discrete control space
        'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
        'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
        'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
        'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range

        'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
        #         'town': 'Town03',  # which town to simulate
        #         'pixor_size': 64,  # size of the pixor labels
    }
    env = gym.make('CarlaEnv-v0', params=params)
    env.reset()
    #     env = NormalizedActions(env)
    ou_noise = OUNoise(env.action_space)

    # print(env.observation_space.shape[0])
    # print(env.action_space.shape[0])

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # print("状态维度" + str(state_dim))
    # print("动作维度" + str(action_dim))
    # print(env.action_space)
    hidden_dim = 256

    ddpg = DDPG(action_dim, state_dim, hidden_dim)

    max_steps = 1 # 30000
    rewards = []
    batch_size = 54
    VAR = 1  # control exploration
    reward_traf = []

    for step in range(max_steps):
        print("================第{}回合======================================".format(step + 1))
        state = env.reset()
        state = torch.flatten(torch.tensor(state))
        ou_noise.reset()
        intrinsic_reward = 0
        episode_reward = 0
        done = False
        st = 0

        while not done:
            action = ddpg.policy_net.get_action(state)
            # print(action)
            action[0] = np.clip(np.random.normal(action[0], VAR), -1, 1)  # 在动作选择上添加随机噪声
            action[1] = np.clip(np.random.normal(action[1], VAR), -0.2, 0.2)  # 在动作选择上添加随机噪声
            # action = ou_noise.get_action(action, st)
            next_state, reward, done, _ = env.step(action)  # 奖励函数的更改需要自行打开安装的库在本地的位置进行修改
            next_state = torch.flatten(torch.tensor(next_state))
            if reward == 0.0:  # 车辆出界，回合结束
                reward = -10
                done = True
            ddpg.replay_buffer.push(state, action, reward, next_state, done)

            if len(ddpg.replay_buffer) > batch_size:
                VAR *= .9995  # decay the action randomness
                intrinsic_reward = ddpg.ddpg_update()

            state = next_state
            episode_reward += reward
            env.render()
            st = st + 1

        rewards.append([episode_reward, intrinsic_reward])
        print("回合奖励为：{}, {}".format(episode_reward, intrinsic_reward))

    env.close()

    df1 = pd.DataFrame(data=rewards,
                       columns=['reward', 'icm_reward'])
    df1.to_csv(filePath, index=False)
    torch.save(ddpg.value_net.state_dict(), 'value_net.pt')
    torch.save(ddpg.policy_net.state_dict(), 'policy_net.pt')
    torch.save(ddpg.target_value_net.state_dict(), 'target_value_net.pt')
    torch.save(ddpg.target_policy_net.state_dict(), 'target_policy_net.pt')


if __name__ == '__main__':
    main()


# if __name__ == '__main__':
#     env = Car_Env()
#     s = env.reset()
#     dqn = DQN()
#     count = 0
#     for i in range(2000):
#         a = dqn.choose_action(s)
#         s_, r, done, info = env.step(a)
#         dqn.push_memory(s, a, r, s_)
#         s = s_
#         if (dqn.position % (MEMORY_CAPACITY - 1)) == 0:
#             dqn.learn()
#             count += 1
#             print('learned times:', count)

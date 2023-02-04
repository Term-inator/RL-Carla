import copy
import glob
import os
import sys
from collections import namedtuple

from PIL import Image

from ICM import ICM, ICMType
from carla_env.carla_env import State

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
from torch.autograd import Variable
import torch.nn.functional as F
from torch import FloatTensor
import random
import time
import numpy as np
import cv2
import math
import gym
import pandas as pd
import carla_env
from encoder import encode_image

IM_WIDTH = 80
IM_HEIGHT = 60
SHOW_PREVIEW = True

SECOND_PER_EPISODE = 10

torch.cuda.empty_cache()
use_cuda = torch.cuda.is_available()
print(use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")
filePath = 'all.csv'

Tensor = FloatTensor

EPSILON = 0.9  # epsilon used for epsilon greedy approach
GAMMA = 0.9
TARGET_NETWORK_REPLACE_FREQ = 100  # How frequently target netowrk updates
MEMORY_CAPACITY = 200
BATCH_SIZE = 32
LR = 0.01  # learning rate


def get_cost(info, state, action, done, last_action=None, last_state=None):
    """ define your risk function here! """
    cost = 0
    if info["crashed"]:
        cost = 100
    elif info["offroad"]:
        cost = 10

    return cost


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


class Risk(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Risk, self).__init__()
        self.fc = nn.Sequential(
                nn.Linear(state_dim+action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        risk_value = self.fc(state_action)
        return risk_value


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


class ReplayBufferRisk:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done, cost):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, cost)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, cost = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done, cost

    def __len__(self):
        return len(self.buffer)


class DDPG(object):
    def __init__(self, config, action_dim, state_dim, hidden_dim):
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
        self.use_risk = config['risk']
        self.risk_lr = 0.001

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

        if self.use_risk:
            self.risk = Risk(state_dim, action_dim, hidden_dim).to(device)
            self.risk_target = Risk(state_dim, action_dim, hidden_dim).to(device)

            self.risk_optimizer = optim.Adam(self.risk.parameters(), lr=self.risk_lr)
            # self.risk_target.load_state_dict(self.risk.state_dict())

            self.replay_buffer = ReplayBufferRisk(self.replay_buffer_size)
        else:
            self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

    def ddpg_update(self):
        if self.use_risk:
            state, action, reward, next_state, done, cost = self.replay_buffer.sample(self.batch_size)
        else:
            state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)
        if self.use_risk:
            cost = torch.FloatTensor(np.array(cost)).unsqueeze(1).to(device)

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

        if self.use_risk:
            target_risk = self.risk_target(next_state, next_action)
            target_risk = cost + ((1 - done) * self.gamma * target_risk).detach()
            current_risk = self.risk(state, action)
            loss_risk = F.mse_loss(current_risk, target_risk)
            self.risk_optimizer.zero_grad()
            loss_risk.backward()
            self.risk_optimizer.step()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )


class Trajectory:
    def __init__(self, state, coordinate, action, cost=None):
        self.state = state
        self.coordinate = coordinate
        self.action = action
        self.cost = cost

    def to_list(self):
        # print(self.state, self.action)
        if self.cost is None:
            return np.concatenate((self.state, self.coordinate, self.action), axis=0)
        return np.concatenate((self.state, self.coordinate, self.action, [self.cost]), axis=0)


def get_info_from_obs(obs, pixor=False):
    if not pixor:
        return obs['camera'], obs['state']


def main():
    TASK_MODE = 'Lane'
    params = {
        'dt': 0.1,  # time interval between two frames
        'port': 2000,  # connection port
        'task_mode': TASK_MODE,  # mode of the task, [random, roundabout (only for Town03)]
        'max_time_episode': 1000,  # 1000,  # maximum timesteps per episode
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
        'out_lane_thres': 0,  # 2.0,  # threshold for out of lane

        'code_mode': 'train',
        'discrete': False,  # whether to use discrete control space
        'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
        'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
        'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
        'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range

        'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
        #         'town': 'Town03',  # which town to simulate
        #         'pixor_size': 64,  # size of the pixor labels

        'icm': True,
        'icm_type': [ICMType.LINEAR, ICMType.LSTM, ICMType.DNN][2],
        'icm_scale': 500,  # 0.001,
        'icm_only': True,

        'store_coordinate': True,
        'risk': False,

        # 'display_size': 500,
        'pixor': False,
    }
    env = gym.make('CarlaEnv-v0', params=params)
    env.reset()
    #     env = NormalizedActions(env)
    ou_noise = OUNoise(env.action_space)

    # print(env.observation_space.shape[0])
    # print(env.action_space.shape[0])

    state_dim = env.observation_space['state'].shape[0]
    action_dim = env.action_space.shape[0]
    # print("状态维度" + str(state_dim))
    # print("动作维度" + str(action_dim))
    # print(env.action_space)
    hidden_dim = 768

    ddpg = DDPG(params, action_dim, state_dim, hidden_dim)

    if params['icm']:
        # ICM
        # icm_module = ICM(params['icm_type'], state_dim, action_dim).to(device)
        icm_module = ICM(params['icm_type'], 512, action_dim).to(device)

    max_steps = 2000
    trajectorys = []
    rewards = []
    batch_size = 54
    VAR = 1  # control exploration
    reward_traf = []

    for step in range(max_steps):
        print("================第{}回合======================================".format(step + 1))
        obs = env.reset()
        camera, state = get_info_from_obs(obs, pixor=params['pixor'])
        camera = Image.fromarray(camera)
        # camera.show()
        camera_feature = encode_image(camera)
        state = torch.flatten(torch.tensor(state))
        # 开始状态
        if params['risk']:
            trajectorys.append(Trajectory(np.zeros(21), [0.0, 0.0], [0.0, 0.0], 0).to_list())
        else:
            trajectorys.append(Trajectory(np.zeros(19), [0.0, 0.0], [0.0, 0.0]).to_list())
        ou_noise.reset()
        episode_intrinsic_reward = 0
        episode_reward = 0
        done = False
        st = 0

        while not done:
            action = ddpg.policy_net.get_action(state)
            # print(action)
            action[0] = np.clip(np.random.normal(action[0], VAR), -1, 1)  # 在动作选择上添加随机噪声
            action[1] = np.clip(np.random.normal(action[1], VAR), -0.2, 0.2)  # 在动作选择上添加随机噪声
            # action = ou_noise.get_action(action, st)
            next_obs, reward, done, info = env.step(action)  # 奖励函数的更改需要自行打开安装的库在本地的位置进行修改
            next_camera, next_state = get_info_from_obs(next_obs, pixor=params['pixor'])
            next_camera = Image.fromarray(next_camera)
            next_camera_feature = encode_image(next_camera)
            next_state = torch.flatten(torch.tensor(next_state))
            intrinsic_reward = 0
            if reward == 0.0:  # 车辆出界，回合结束
                reward = -10
                done = True

            if params['risk']:
                """Define the cost"""
                cost = get_cost(info, next_state, action, done)

            if params['icm']:
                # ICM
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
                action_tensor = torch.FloatTensor(action).unsqueeze(0).to(device)
                # pred_next_state, pred_action = icm_module(state_tensor, next_state_tensor, action_tensor)
                pred_next_state, pred_action = icm_module(camera_feature, next_camera_feature, action_tensor)
                # pred_next_state = predict_module(torch.cat((state_tensor, action_tensor), 1))
                # pred_action = inv_predict_module(torch.cat((state_tensor, next_state_tensor), 1))
                # forward_loss = F.mse_loss(pred_next_state, next_state_tensor).to(device)
                forward_loss = F.mse_loss(pred_next_state, next_camera_feature).to(device)
                inverse_loss = F.cross_entropy(pred_action, action_tensor).to(device)
                intrinsic_reward = params['icm_scale'] * forward_loss.item()
                episode_intrinsic_reward += intrinsic_reward
                forward_loss.backward(retain_graph=True)
                inverse_loss.backward(retain_graph=True)

            if params['icm_only']:
                reward = 0  # 关闭外部奖励
            if params['risk']:
                ddpg.replay_buffer.push(state, action, reward + intrinsic_reward, next_state, done, cost)
            else:
                ddpg.replay_buffer.push(state, action, reward + intrinsic_reward, next_state, done)

            if len(ddpg.replay_buffer) > batch_size:
                VAR *= .9995  # decay the action randomness
                ddpg.ddpg_update()

            if params['risk']:
                trajectorys.append(Trajectory(state.numpy(), info['coordinate'], action, cost).to_list())
            else:
                trajectorys.append(Trajectory(state.numpy(), info['coordinate'], action).to_list())

            state = next_state
            camera = next_camera
            episode_reward += reward
            env.render()
            st = st + 1

        rewards.append([episode_reward, episode_intrinsic_reward])
        if params['risk']:
            trajectorys.append(Trajectory(np.zeros(21), [1, 1], [1, 1], 1).to_list())
        else:
            trajectorys.append(Trajectory(np.zeros(19), [1, 1], [1, 1]).to_list())
        print("回合奖励为：{}, {}".format(episode_reward, episode_intrinsic_reward))

    env.close()
    # print(states)
    df1 = pd.DataFrame(data=rewards,
                       columns=['reward', 'icm_reward'])
    df1.to_csv(filePath, index=False)
    columns = ['velocity_t_x', 'velocity_t_y', 'accel_t_x', 'accel_t_y',
                'delta_yaw_t', 'dyaw_dt_t', 'lateral_dist_t',
                'action_last_accel', 'accel_last_steer',
                'future_angles_0', 'future_angles_1', 'future_angles_3',
                'speed', 'angle', 'offset', 'e_speed', 'distance',
                'e_distance', 'safe_distance', 'light_state', 'sl_distance',
                'x', 'y', 'action_acc', 'action_steer']
    if params['risk']:
        columns.append('cost')
    df2 = pd.DataFrame(data=trajectorys, columns=columns)
    df2.to_csv('trajectory.csv', index=False)
    torch.save(ddpg.value_net.state_dict(), 'value_net.pt')
    torch.save(ddpg.policy_net.state_dict(), 'policy_net.pt')
    torch.save(ddpg.target_value_net.state_dict(), 'target_value_net.pt')
    torch.save(ddpg.target_policy_net.state_dict(), 'target_policy_net.pt')


if __name__ == '__main__':
    main()

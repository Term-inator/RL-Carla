import copy
import glob
import os
import sys
from collections import namedtuple

from PIL import Image
from tensorboardX import SummaryWriter

from ICM import ICM, ICMType
import RND
from RND import RNDModel
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
file_path = 'trajectory'

Tensor = FloatTensor

EPSILON = 0.9  # epsilon used for epsilon greedy approach
GAMMA = 0.9
TARGET_NETWORK_REPLACE_FREQ = 100  # How frequently target netowrk updates
MEMORY_CAPACITY = 200
BATCH_SIZE = 32
LR = 0.01  # learning rate


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_cost(info, state, action, done, last_action=None, last_state=None):
    """ define your risk function here! """
    cost = 0
    if info["crashed"]:
        cost = 100
    elif info["offroad"]:
        cost = 10

    return cost


class Critic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(Critic, self).__init__()

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


class Actor(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(Actor, self).__init__()

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
            nn.Linear(state_dim + action_dim, hidden_dim),
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


class CTD3(object):
    def __init__(self, config, action_dim, state_dim, hidden_dim, writer=None):
        super(CTD3, self).__init__()
        self.action_dim, self.state_dim, self.hidden_dim = action_dim, state_dim, hidden_dim
        self.batch_size = 50
        self.gamma = 0.99
        self.min_value = -np.inf
        self.max_value = np.inf
        self.soft_tau = 2e-2
        self.replay_buffer_size = 10000
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        self.use_risk = config.get('use_risk', False)
        self.risk_lr = 0.001
        self.policy_delay = config.get("policy_delay", 2)

        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic1 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(device)

        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim, hidden_dim).to(device)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(param.data)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.critic_lr)

        if config['code_mode'] == 'test':
            self.load('output_logger', config['test_index'])
            self.actor.eval()
            self.critic1.eval()
            self.critic2.eval()
            self.actor_target.eval()
            self.critic1_target.eval()
            self.critic2_target.eval()

        self.actor_criterion = nn.MSELoss()

        if self.use_risk:
            self.risk = Risk(state_dim, action_dim, hidden_dim).to(device)
            self.risk_target = Risk(state_dim, action_dim, hidden_dim).to(device)

            self.risk_optimizer = optim.Adam(self.risk.parameters(), lr=self.risk_lr)
            # self.risk_target.load_state_dict(self.risk.state_dict())

            self.replay_buffer = ReplayBufferRisk(self.replay_buffer_size)
        else:
            self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

        self.writer = writer

    def update(self):
        # print("update")
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

        next_action = self.actor_target(next_state)
        target_Q1 = self.critic1_target(next_state, next_action)
        target_Q2 = self.critic2_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + ((1 - done) * self.gamma * target_Q).detach()

        current_Q1 = self.critic1(state, action)
        loss_Q1 = F.mse_loss(current_Q1, target_Q)
        self.critic1_optimizer.zero_grad()
        loss_Q1.backward()
        self.critic1_optimizer.step()
        self.writer.add_scalar('Loss/Q1_loss', loss_Q1, global_step=self.num_critic_update_iteration)

        current_Q2 = self.critic2(state, action)
        loss_Q2 = F.mse_loss(current_Q2, target_Q)
        self.critic2_optimizer.zero_grad()
        loss_Q2.backward()
        self.critic2_optimizer.step()
        self.writer.add_scalar('Loss/Q2_loss', loss_Q2, global_step=self.num_critic_update_iteration)

        if self.use_risk:
            target_risk = self.risk_target(next_state, next_action)
            target_risk = cost + ((1 - done) * self.gamma * target_risk).detach()
            current_risk = self.risk(state, action)
            loss_risk = F.mse_loss(current_risk, target_risk)
            self.risk_optimizer.zero_grad()
            loss_risk.backward()
            self.risk_optimizer.step()
            self.writer.add_scalar('Loss/Risk_loss', loss_risk, global_step=self.num_critic_update_iteration)

        if self.num_critic_update_iteration % self.policy_delay == 0:

            actor_loss = - self.critic1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.writer.add_scalar('Loss/Actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(
                    ((1 - self.soft_tau) * target_param.data) + self.soft_tau * param.data
                )
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(
                    ((1 - self.soft_tau) * target_param.data) + self.soft_tau * param.data
                )
            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(
                    ((1 - self.soft_tau) * target_param.data) + self.soft_tau * param.data
                )

            self.num_actor_update_iteration += 1

        self.num_critic_update_iteration += 1
        self.num_training += 1

    def save(self, path, index=''):
        torch.save(self.actor.state_dict(), os.path.join(path, f'actor_{index}.pth'))
        torch.save(self.actor_target.state_dict(), os.path.join(path, f'actor_target_{index}.pth'))
        torch.save(self.critic1.state_dict(), os.path.join(path, f'critic1_{index}.pth'))
        torch.save(self.critic1_target.state_dict(), os.path.join(path, f'critic1_target_{index}.pth'))
        torch.save(self.critic2.state_dict(), os.path.join(path, f'critic2_{index}.pth'))
        torch.save(self.critic2_target.state_dict(), os.path.join(path, f'critic2_target_{index}.pth'))
        print("model has been saved...")

    def load(self, path, index=''):
        self.actor.load_state_dict(torch.load(os.path.join(path, f'actor_{index}.pth')))
        self.actor_target.load_state_dict(torch.load(os.path.join(path, f'actor_target_{index}.pth')))
        self.critic1.load_state_dict(torch.load(os.path.join(path, f'critic1_{index}.pth')))
        self.critic1_target.load_state_dict(torch.load(os.path.join(path, f'critic1_target_{index}.pth')))
        self.critic2.load_state_dict(torch.load(os.path.join(path, f'critic2_{index}.pth')))
        self.critic2_target.load_state_dict(torch.load(os.path.join(path, f'critic2_target_{index}.pth')))
        print("model has been loaded...")


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


def save_trajectory(file_path, trajectories, rewards, trajectory_columns, reward_columns, index=''):
    df1 = pd.DataFrame(data=rewards,
                       columns=reward_columns)

    df2 = pd.DataFrame(data=trajectories, columns=trajectory_columns)
    df = pd.concat([df2, df1], axis=1)
    df.to_csv(f'{file_path}_{index}.csv', index=False)
    print("trajectory has been saved...")


def main():
    TASK_MODE = 'Lane'
    params = {
        'dt': 0.1,  # time interval between two frames
        'port': 2000,  # connection port
        'task_mode': TASK_MODE,  # mode of the task, [random, roundabout (only for Town03)]
        'max_time_episode': 5000,  # 1000,  # maximum timesteps per episode
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
        'out_lane_thres': 0.0,  # 2.0,  # threshold for out of lane

        'code_mode': 'train',
        'test_index': '8000',
        'discrete': False,  # whether to use discrete control space
        'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
        'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
        'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
        'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range

        'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
        #         'town': 'Town03',  # which town to simulate
        #         'pixor_size': 64,  # size of the pixor labels

        'icm': False,
        'icm_type': [ICMType.LINEAR, ICMType.LSTM, ICMType.DNN][2],
        'icm_scale': 500,  # 500,
        'icm_only': False,

        'rnd': False,
        'rnd_scale': 1.0,
        'rnd_only': False,

        'store_coordinate': True,
        'use_risk': False,

        # 'display_size': 500,
        'pixor': False,
        'policy_delay': 2,
    }
    seed_torch()
    writer = SummaryWriter('./tensorboard/' + TASK_MODE + '/')
    df1_columns = ['reward', 'intrinsic_reward']
    df2_columns = ['velocity_t_x', 'velocity_t_y', 'accel_t_x', 'accel_t_y',
                   'delta_yaw_t', 'dyaw_dt_t', 'lateral_dist_t',
                   'action_last_accel', 'accel_last_steer',
                   'future_angles_0', 'future_angles_1', 'future_angles_3',
                   'speed', 'angle', 'offset', 'e_speed', 'distance',
                   'e_distance', 'safe_distance', 'light_state', 'sl_distance',
                   'x', 'y', 'action_acc', 'action_steer']
    if params['use_risk']:
        df2_columns.append('cost')

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

    ddpg = CTD3(params, action_dim, state_dim, hidden_dim, writer=writer)

    if params['icm']:
        # ICM
        # icm_module = ICM(params['icm_type'], state_dim, action_dim).to(device)
        icm_module = ICM(params['icm_type'], 512, action_dim).to(device)
    elif params['rnd']:
        # RND
        rnd_module = RNDModel(512, 512).to(device)

    max_steps = 10010
    trajectorys = []
    rewards = []
    batch_size = 54
    VAR = 1  # control exploration
    reward_traf = []

    iteration = 0

    for step in range(max_steps):
        print("================第{}回合======================================".format(step + 1))
        obs = env.reset()
        camera, state = get_info_from_obs(obs, pixor=params['pixor'])
        camera = Image.fromarray(camera)
        # camera.show()
        camera_feature = encode_image(camera)
        state = torch.flatten(torch.tensor(state))

        # 开始状态
        if params['use_risk']:
            trajectorys.append(Trajectory(np.zeros(21), [0.0, 0.0], [0.0, 0.0], 0).to_list())
        else:
            trajectorys.append(Trajectory(np.zeros(21), [0.0, 0.0], [0.0, 0.0]).to_list())
        rewards.append([0, 0])

        ou_noise.reset()
        episode_intrinsic_reward = 0
        episode_reward = 0
        done = False
        st = 0

        while not done:
            action = ddpg.actor.get_action(state)
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

            if params['use_risk']:
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
                forward_loss.backward(retain_graph=True)
                inverse_loss.backward(retain_graph=True)
            elif params['rnd']:
                # RND
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
                target_next_feature = rnd_module.target(next_camera_feature)
                predict_next_feature = rnd_module.predictor(next_camera_feature)
                forward_loss = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2
                intrinsic_reward = params['rnd_scale'] * forward_loss.item()
                forward_loss.backward(retain_graph=True)

            if (params['icm'] and params['icm_only']) or (params['rnd'] and params['rnd_only']):
                reward = 0  # 关闭外部奖励
            if params['use_risk']:
                ddpg.replay_buffer.push(state, action, reward + intrinsic_reward, next_state, done, cost)
            else:
                ddpg.replay_buffer.push(state, action, reward + intrinsic_reward, next_state, done)

            if params['code_mode'] == 'train':
                if len(ddpg.replay_buffer) > batch_size:
                    VAR *= .9995  # decay the action randomness
                    ddpg.update()

            if params['use_risk']:
                trajectorys.append(Trajectory(state.numpy(), info['coordinate'], action, cost).to_list())
            else:
                trajectorys.append(Trajectory(state.numpy(), info['coordinate'], action).to_list())

            state = next_state
            camera = next_camera
            episode_intrinsic_reward += intrinsic_reward
            episode_reward += reward
            st += 1
            iteration += 1
            writer.add_scalar('Reward/extrinsic_reward', reward, global_step=iteration)
            writer.add_scalar('Reward/intrinsic_reward', intrinsic_reward, global_step=iteration)
            rewards.append([reward, intrinsic_reward])

            env.render()

        if params['use_risk']:
            trajectorys.append(Trajectory(np.zeros(21), [1, 1], [1, 1], 1).to_list())
        else:
            trajectorys.append(Trajectory(np.zeros(21), [1, 1], [1, 1]).to_list())
        rewards.append([1, 1])
        writer.add_scalar('Reward/epoch_reward', (episode_reward + episode_intrinsic_reward) / st, global_step=step)
        print("回合奖励为：{}, {}".format(episode_reward, episode_intrinsic_reward))

        # 保存轨迹
        if step % 500 == 0 and step != 0:
            save_trajectory(file_path, trajectorys, rewards, trajectory_columns=df2_columns, reward_columns=df1_columns,
                            index=str(step))

        # 保存模型
        if step % 1000 == 0 and step != 0:
            if params['code_mode'] == 'train':
                ddpg.save('output_logger', index=str(step))

    env.close()

    save_trajectory(file_path, trajectorys, rewards, trajectory_columns=df2_columns, reward_columns=df1_columns,
                    index='final')
    if params['code_mode'] == 'train':
        ddpg.save('output_logger', index='final')


if __name__ == '__main__':
    main()

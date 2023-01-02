import glob
import os
import sys
from collections import namedtuple

try:
    sys.path.append('D:/Programs/Carla/CARLA_0.9.13/WindowsNoEditor/PythonAPI/carla')
except IndexError:
    pass

import carla
import random
import time
import numpy as np
import cv2
import math

IM_WIDTH = 80
IM_HEIGHT = 60
SHOW_PREVIEW = True

SECOND_PER_EPISODE = 10


class Car_Env():
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter('model3')[0]

    def reset(self):
        self.collision_hist = []
        self.radar_hist = []
        self.actor_list = []
        self.transform = self.world.get_map().get_spawn_points()[100]  # spwan_points共265个点，选第一个点作为初始化小车的位置
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)

        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute('image_size_x', f'{self.im_width}')
        self.rgb_cam.set_attribute('image_size_y', f'{self.im_height}')
        self.rgb_cam.set_attribute('fov', f'110')

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        time.sleep(4)

        # collision sensor
        colsensor = self.blueprint_library.find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        # target_transform 定义驾驶目的地坐标
        self.target_transform = self.world.get_map().get_spawn_points()[101]
        self.target_dis = self.target_transform.location.distance(self.vehicle.get_location())

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera

    def collision_data(self, event):
        self.collision_hist.append(event)

    def radar_data(self, mesure):
        self.radar_hist.append(mesure)

    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, : 3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3
        return i3 / 255.0

    def step(self, action):
        last_dis = self.target_dis
        if action == 0:
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=1.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False))
        elif action == 1:
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=0.5, steer=-1, brake=0.0, hand_brake=False, reverse=False))
        elif action == 2:
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=0.5, steer=1, brake=0.0, hand_brake=False, reverse=False))
        elif action == 4:
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.5, hand_brake=False, reverse=False))
        else:
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=1.0, steer=0.0, brake=0.0, hand_brake=False, reverse=True))

        self.target_dis = self.target_transform.location.distance(self.vehicle.get_location())

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif last_dis < self.target_dis:  # 距离目标越来越远了
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        if self.episode_start + SECOND_PER_EPISODE < time.time():
            done = True
        time.sleep(1)
        return self.front_camera, reward, done, None


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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(896, 5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # 一层卷积
        x = F.relu(self.bn2(self.conv2(x)))  # 两层卷积
        x = F.relu(self.bn3(self.conv3(x)))  # 三层卷积
        return self.head(x.view(x.size(0), -1))  # 全连接层


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        # Define counter, memory size and loss function
        self.learn_step_counter = 0  # count the steps of learning process

        self.memory = []
        self.position = 0  # counter used for experience replay buff
        self.capacity = 200

        # ------- Define the optimizer------#
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)

        # ------Define the loss function-----#
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        # This function is used to make decision based upon epsilon greedy

        x = torch.unsqueeze(torch.FloatTensor(x), 0)  # add 1 dimension to input state x
        x = x.permute(0, 3, 2, 1)  # 把图片维度从[batch, height, width, channel] 转为[batch, channel, height, width]
        # input only one sample
        if np.random.uniform() < EPSILON:  # greedy
            # use epsilon-greedy approach to take action
            actions_value = self.eval_net.forward(x)
            # print(torch.max(actions_value, 1))
            # torch.max() returns a tensor composed of max value along the axis=dim and corresponding index
            # what we need is the index in this function, representing the action of cart.
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]

        else:
            action = np.random.randint(0, 5)

        return action

    def push_memory(self, s, a, r, s_):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(torch.unsqueeze(torch.FloatTensor(s), 0),
                                                torch.unsqueeze(torch.FloatTensor(s_), 0), \
                                                torch.from_numpy(np.array([a])),
                                                torch.from_numpy(np.array([r], dtype='int64')))
        self.position = (self.position + 1) % self.capacity

    def get_sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def learn(self):
        # Define how the whole DQN works including sampling batch of experiences,
        # when and how to update parameters of target network, and how to implement
        # backward propagation.

        # update the target network every fixed steps
        if self.learn_step_counter % TARGET_NETWORK_REPLACE_FREQ == 0:
            # Assign the parameters of eval_net to target_net
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        transitions = self.get_sample(BATCH_SIZE)  # 抽样
        batch = Transition(*zip(*transitions))

        # extract vectors or matrices s,a,r,s_ from batch memory and convert these to torch Variables
        # that are convenient to back propagation
        b_s = Variable(torch.cat(batch.state))
        # convert long int type to tensor
        b_a = Variable(torch.cat(batch.action))
        b_r = Variable(torch.cat(batch.reward))
        b_s_ = Variable(torch.cat(batch.next_state))

        # b_s和b_s_分别对应当前帧和下一帧的图像数据，变量的维度是80*60*3(x*y*rgb_channel)，但进入神经网络需将其维度变为3*80*60
        b_s = b_s.permute(0, 3, 2, 1)
        b_s_ = b_s_.permute(0, 3, 2, 1)

        # calculate the Q value of state-action pair
        q_eval = self.eval_net(b_s).gather(1, b_a.unsqueeze(1))  # (batch_size, 1)

        # calculate the q value of next state
        q_next = self.target_net(b_s_).detach()  # detach from computational graph, don't back propagate
        # select the maximum q value
        # q_next.max(1) returns the max value along the axis=1 and its corresponding index
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # (batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()  # reset the gradient to zero
        loss.backward()
        self.optimizer.step()  # execute back propagation for one step


Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward'))

if __name__ == '__main__':
    env = Car_Env()
    s = env.reset()
    dqn = DQN()
    count = 0
    for i in range(2000):
        a = dqn.choose_action(s)
        s_, r, done, info = env.step(a)
        dqn.push_memory(s, a, r, s_)
        s = s_
        if (dqn.position % (MEMORY_CAPACITY - 1)) == 0:
            dqn.learn()
            count += 1
            print('learned times:', count)

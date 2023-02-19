import copy
import os
import traceback

import numpy as np
import random
import time

# from pyglet.resource import file
from skimage.transform import resize

import gym
import pygame
from gym import spaces
from gym.utils import seeding

from .misc import _vec_decompose, delta_angle_between
from carla_env.misc import *
from carla_env.route_planner import RoutePlanner
from carla_env.coordinates import train_coordinates
from .carla_logger import *


# 导入Carla
import glob
import os
import sys
# try:
#     sys.path.append(glob.glob('./carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass
import carla
import math
import random
import re
import weakref
from carla import ColorConverter as cc

from .render import BirdeyeRender

'''
{'obs_size': (160, 100), 'max_past_step': 1, 'dt': 0.025, 'ego_vehicle_filter': 'vehicle.lincoln*', 'port': 2021, 'task_mode': 'Lane', 'code_mode': 'train', 'max_time_episode': 250, 'desired_speed': 15, 'max_ego_spawn_times': 100, 'number_of_vehicles': 3, 'number_of_walkers': 5, 'max_waypt': 20, 'obs_range': 32, 'lidar_bin': 0.125, 'd_behind': 12, 'out_lane_thres': 2.0, 'display_route': False, 'discrete': False, 'discrete_acc': [-3.0, 0.0, 3.0], 'discrete_steer': [-0.2, 0.0, 0.2], 'continuous_accel_range': [-3.0, 3.0], 'continuous_steer_range': [-0.3, 0.3]},
'''

is_end = False
e_speed = 10
e_distance = 20
safe_distance = 10
k1 = 0.5
k2 = 0.5
k3 = 1


# def find_weather_presets():
#     rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
#     name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
#     presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
#     return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

class State:
    """
    状态类，包含状态信息
    """

    def __init__(self):
        self.speed = 0.0
        self.angle = 0.0
        self.offset = 0.0
        self.e_speed = e_speed
        self.distance = -1
        self.e_distance = e_distance
        self.safe_distance = safe_distance
        self.light_state = None
        self.sl_distance = 50

        self.coordinate = None

    def __repr__(self):
        return f'({self.speed}, {self.angle}, {self.offset}, {self.e_speed}, {self.distance}, ' \
               f'{self.e_distance}, {self.safe_distance}, {self.light_state}, {self.sl_distance})'


class ObstacleDetector:
    def __init__(self, parent_actor, ):
        self.other_actor = None
        self.distance = safe_distance + 10
        self.sensor = None
        self._parent = parent_actor
        world = self._parent.get_world()
        # 获得障碍传感器
        bp = world.get_blueprint_library().find("sensor.other.obstacle")
        # 设置障碍传感器参数
        bp.set_attribute('distance', str(40.0))
        bp.set_attribute('debug_linetrace', str(True))
        bp.set_attribute('only_dynamics', str(True))
        bp.set_attribute('hit_radius', str(0.8))
        # 生成传感器
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: ObstacleDetector._on_detected(weak_self, event))

    # @staticmethod
    def _on_detected(weak_self, event: carla.ObstacleDetectionEvent):
        self = weak_self()
        if not self:
            return
        self.distance = event.distance
        self.other_actor = event.other_actor
        # print(f'other_actor:{self.other_actor.type_id}, distance:{self.distance}')


class CarlaEnv(gym.Env):

    def __init__(self, params):
        """
        初始化参数
        """
        self.logger = setup_carla_logger(
            "output_logger", experiment_name=str(params['port']))
        self.logger.info("Env running in port {}".format(params['port']))

        self.dt = params['dt']  # 时间间隔
        self.port = params['port']
        self.task_mode = params['task_mode']
        self.max_time_episode = params['max_time_episode']
        self.obs_size = params['obs_size']  # rendering screen size[160,100]
        self.state_size = (self.obs_size[0], self.obs_size[1] - 36)  # ?
        self.desired_speed = params['desired_speed']
        self.max_ego_spawn_times = params['max_ego_spawn_times']

        self.max_past_step = params['max_past_step']
        self.max_waypt = params['max_waypt']  # waypoint数量--地图上一个点，道路信息Transform
        self.obs_range = params['obs_range']  # 观察距离
        self.lidar_bin = params['lidar_bin']  # 雷达看到的距离
        self.d_behind = params['d_behind']
        self.obs_size = int(self.obs_range / self.lidar_bin)  # 观察距离/雷达看到的距离
        self.display_route = params['display_route']  # 是否渲染路线

        # 可以保留的
        # self.display_size = params['display_size']  # rendering screen size
        self.number_of_vehicles = params['number_of_vehicles']  # 车子数量
        self.number_of_walkers = params['number_of_walkers']  # 行人数量
        self.out_lane_thres = params['out_lane_thres']  # 车道数量
        # if params['pixor']:
        #     self.pixor = params['pixor']
        #     self.pixor_size = params['pixor_size']
        # else:
        #     self.pixor = False

        self.code_mode = params['code_mode']
        self.spectator = None

        self.store_coordinate = params['store_coordinate']  # 是否存储坐标

        '''
        定义动作空间和观测空间
        '''
        # 定义动作空间
        self.discrete = params['discrete']
        self.discrete_act = [params['discrete_acc'],
                             params['discrete_steer']]  # acc[-3.0, 0.0, 3.0], steer[-0.2, 0.0, 0.2]
        self.n_acc = len(self.discrete_act[0])
        self.n_steer = len(self.discrete_act[1])
        # 如果动作空间是离散的
        if self.discrete:
            self.action_space = spaces.Discrete(self.n_acc * self.n_steer)  # 通过spaces.Discrete(）转换为gym中的离散动作空间
        else:
            # 连续动作，gym：spaces.Box 描述的为一个n维的实数空间 [公式] ，可以指定上下限，也可以不指定上下限。
            # 'continuous_accel_range': [-3.0, 3.0]
            # 'continuous_steer_range': [-0.3, 0.3],
            self.action_space = spaces.Box(np.array([params['continuous_accel_range'][0],
                                                     params['continuous_steer_range'][0]]),
                                           np.array([params['continuous_accel_range'][1],
                                                     params['continuous_steer_range'][1]]),
                                           dtype=np.float32)  # acc, steer
        # 定义观测空间
        # self.observation_space = spaces.Box(low=-50.0, high=50.0, shape=(21,), dtype=np.float32)
        #         print(self.observation_space)
        observation_space_dict = {
            'camera': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            # 'lidar': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            # 'birdeye': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            # 'state': spaces.Box(np.array([-2, -1, -5, 0]), np.array([2, 1, 30, 1]), dtype=np.float32),
            'state': spaces.Box(low=-50.0, high=50.0, shape=(21,), dtype=np.float32)
        }
        # if self.pixor:
        #     observation_space_dict.update({
        #         'roadmap': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
        #         'vh_clas': spaces.Box(low=0, high=1, shape=(self.pixor_size, self.pixor_size, 1), dtype=np.float32),
        #         'vh_regr': spaces.Box(low=-5, high=5, shape=(self.pixor_size, self.pixor_size, 6), dtype=np.float32),
        #         'pixor_state': spaces.Box(np.array([-1000, -1000, -1, -1, -5]), np.array([1000, 1000, 1, 1, 20]),
        #                                   dtype=np.float32)
        #     })
        self.observation_space = spaces.Dict(observation_space_dict)

        '''
        连接Carla,并根据任务模式设置ego车辆的起点和终点
        '''
        print('connect carla client')
        self._make_carla_client('localhost', self.port)
        print('connected carla client')
        # 根据任务设置起点和终点
        self.starts, self.dests = train_coordinates(self.task_mode)
        #         print("起点是：",self.starts,"\n终点是:" self.dests)
        self.route_deterministic_id = 0

        '''
        得到能够用于生成车辆和行人的位置spawn_points
        '''
        self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())

        self.walker_spawn_points = []
        # 随机生成行人的位置 spawn_point
        for i in range(self.number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                self.walker_spawn_points.append(spawn_point)
        '''
        # 创建ego车辆和传感器的blueprint
        '''
        # 1.ego车辆
        # 'ego_vehicle_filter': 'vehicle.lincoln*',表示车辆的外观
        self.ego_bp = self._create_vehicle_bluepprint(params['ego_vehicle_filter'], color='49,8,8')
        # print(self.ego_bp)

        # 2.Collision sensor,碰撞
        self.collision_hist = []  # The collision history
        self.collision_hist_l = 1  # collision history length
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

        # 3.lidar传感器
        self.lidar_data = None
        self.lidar_height = 2.1
        self.lidar_trans = carla.Transform(carla.Location(x=0.0, z=self.lidar_height))
        self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        self.lidar_bp.set_attribute('channels', '32')
        self.lidar_bp.set_attribute('range', '5000')

        # 4.Camera 传感器
        self.camera_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
        self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
        self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
        self.camera_bp.set_attribute('fov', '110')
        # 设置传感器捕获的周期
        self.camera_bp.set_attribute('sensor_tick', '0.02')
        '''
        设置一些基本配置项
        '''
        # Set fixed simulation step for synchronous mode，设置时间间隔
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.dt

        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0

        # Initialize the renderer
        # self._init_renderer()

        self.state_info = {}  # 存储状态信息
        # self.actors = []  # 存储actor
        self.distances = [1., 5., 10.]

    def reset(self):
        # 1.清空传感器
        self.collision_sensor = None
        self.obstacle_sensor = None
        self.lidar_sensor = None
        self.camera_sensor = None

        # 2.清除所有的行人、路人、传感器
        # while self.actors:
        #     actor = self.actors.pop()
        #     if actor is not None and actor.is_alive:
        #         actor.destroy()
        self._clear_all_actors(['sensor.other.collision', 'sensor.lidar.ray_cast', 'sensor.camera.rgb', 'vehicle.*',
                                'controller.ai.walker', 'walker.*', 'sensor.other.obstacle'])
        #         self._clear_all_actors(['sensor.other.collision', 'vehicle.*', 'controller.ai.walker', 'walker.*'])
        # 3.取消同步模式
        self._set_synchronous_mode(False)

        # 4.在随机位置上生成周围车辆
        # random.shuffle(self.vehicle_spawn_points)
        # count = self.number_of_vehicles
        # if count > 0:
        #     for spawn_point in self.vehicle_spawn_points:
        #         if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4]):
        #             count -= 1
        #         if count <= 0:
        #             break
        # while count > 0:
        #     if self._try_spawn_random_vehicle_at(random.choice(self.vehicle_spawn_points), number_of_wheels=[4]):
        #         count -= 1
        # print("成功生成周围车辆！")

        # 5.Spawn pedestrians，在随机位置上生成行人
        # random.shuffle(self.walker_spawn_points)
        # count = self.number_of_walkers
        # if count > 0:
        #     for spawn_point in self.walker_spawn_points:
        #         if self._try_spawn_random_walker_at(spawn_point):
        #             count -= 1
        #         if count <= 0:
        #             break
        # while count > 0:
        #     if self._try_spawn_random_walker_at(random.choice(self.walker_spawn_points)):
        #         count -= 1
        # print("成功生成周围行人！")

        # 得到周围车辆、行人的多边形边界-->可以用于判断ego车是否会与周围车辆重合
        self.vehicle_polygons = []
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        self.walker_polygons = []
        walker_poly_dict = self._get_actor_polygons('walker.*')
        self.walker_polygons.append(walker_poly_dict)

        # 根据任务，得到车辆的起点和终点，以及走哪一条路
        if self.task_mode == 'Straight':
            self.route_id = 0
        elif self.task_mode == 'Curve':
            self.route_id = 1  # np.random.randint(2, 4)
        elif self.task_mode == 'U_curve':
            self.route_id = 0
        elif self.task_mode == 'Long' or self.task_mode == 'Lane' or self.task_mode == 'Lane_test':
            if self.code_mode == 'train':
                # self.route_id = np.random.randint(0, 4)
                self.route_id = 0
            elif self.code_mode == 'test':
                self.route_id = 0
                # self.route_id = self.route_deterministic_id
                # self.route_deterministic_id = (self.route_deterministic_id + 1) % 4
        elif self.task_mode == 'TrafficLight':
            self.route_id = 0
        self.start = self.starts[self.route_id]
        self.dest = self.dests[self.route_id]
        self.current_wpt = np.array((self.start[0], self.start[1], self.start[5]))  # (x,y,yaw)
        # 6.Spawn the ego vehicle 根据任务在指定的位置上生成ego车辆
        ego_spawn_times = 0
        while True:
            if ego_spawn_times > self.max_ego_spawn_times:
                self.reset()
            transform = self._set_carla_transform(self.start)  # 从起点生成ego车辆
            if self.code_mode == 'train':
                # 如果处于训练模式下，则在指定路段的随机位置生成ego车辆
                 transform = self._get_random_position_between(start=self.start, dest=self.dest, transform=transform)
            else:
                transform = self.get_position(self.start, 60)  # Lane 修正
            if self._try_spawn_ego_vehicle_at(transform): # and self._try_spawn_random_vehicle_at(self.get_position(self.start, 30)):
                break
            else:
                ego_spawn_times += 1
                time.sleep(0.1)
            print("成功生成ego车辆！")
        spec_loc = carla.Location(self.start[0], self.start[1], self.start[2] + 5)
        # spec_rot = carla.Rotation(self.start[3], self.start[4], self.start[5])
        spec_rot = self.ego.get_transform().rotation
        spec_tf = carla.Transform(spec_loc, spec_rot)
        forward_vector = spec_tf.get_forward_vector()
        spec_tf.location += forward_vector * (-5)
        self.spectator.set_transform(spec_tf)

        # 7.Add collision sensor，添加检测碰撞传感器,并监听碰撞事件存储碰撞强度
        self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
        # self.actors.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: get_collision_hist(event))

        self.obstacle_sensor = ObstacleDetector(self.ego)

        # self.actors.append(self.obstacle_sensor.sensor)

        def get_collision_hist(event):
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
            self.collision_hist.append(intensity)
            if len(self.collision_hist) > self.collision_hist_l:
                self.collision_hist.pop(0)

        self.collision_hist = []
        print("成功生成检测碰撞传感器！")

        # 8.Add lidar sensor,添加检测lidar传感器，存储data
        # self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego)
        # self.lidar_sensor.listen(lambda data: get_lidar_data(data))
        #
        # def get_lidar_data(data):
        #     self.lidar_data = data
        #
        # print("成功生成lidar传感器！")

        # 9.Add camera sensor，添加camera传感器，并存储img
        self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)
        self.camera_sensor.listen(lambda data: get_camera_img(data))

        def get_camera_img(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.camera_img = array

        print("成功生成camera传感器！")

        # 10.Update timesteps
        self.time_step = 1
        self.reset_step += 1
        #         print(self.time_step,self.reset_step)

        # 11.Enable sync mode 同步模式：服务器会等待客户端命令
        self.settings.synchronous_mode = True
        self.world.apply_settings(self.settings)

        # 设置汽车的初始速度
        yaw = (self.ego.get_transform().rotation.yaw) * np.pi / 180.0
        init_speed = carla.Vector3D(
            x=self.desired_speed * np.cos(yaw),
            y=self.desired_speed * np.sin(yaw))
        self.ego.set_target_velocity(init_speed)
        self.world.tick()
        # 重设上一步的action序列
        self.last_action = np.array([0.0, 0.0])

        self.isCollided = False
        self.isTimeOut = False
        # self.isSuccess = False
        self.isOutOfLane = False
        self.isSpecialSpeed = False
        self.reachDest = False
        self.enter_junction = False

        self.obstacle_sensor = ObstacleDetector(self.ego)

        # 12.设置car的路线
        self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
        self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

        # x Set ego information for render
        # self.birdeye_render.set_hero(self.ego, self.ego.id)

        obs = self._get_obs()
        #         print('当前状态是：', obs)
        return obs  # gym返回

    def step(self, action):
        # 计算加速度和方向盘角度
        if self.discrete:
            acc = self.discrete_act[0][action // self.acc]
            steer = self.discrete_act[1][action % self.n_steer]
        else:
            acc = action[0]
            steer = action[1]
        # 将加速度转化为刹车和油门---其中throttle [0.0, 1.0]，steer[-1,1],brake[0,1]之间的标量
        if acc > 0:
            throttle = np.clip(acc / 3, 0, 1)  # 这里除3，要看情况
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-acc / 8, 0, 1)  # 这里除8，要看情况

        # Apply control
        act = carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))
        self.ego.apply_control(act)
        if self.time_step > min(300, self.max_time_episode / 3):
            ego_transform = self.ego.get_transform()
            self.spectator.set_transform(
                carla.Transform(ego_transform.location + carla.Location(z=50),
                                carla.Rotation(-90)))

        self.world.tick()

        # Append actors polygon list，更新车辆和行人的位置信息
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        while len(self.vehicle_polygons) > self.max_past_step:
            self.vehicle_polygons.pop(0)
        walker_poly_dict = self._get_actor_polygons('walker.*')
        self.walker_polygons.append(walker_poly_dict)
        while len(self.walker_polygons) > self.max_past_step:
            self.walker_polygons.pop(0)

        # route planner，规划路线
        self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

        # state information
        info = {
            'waypoints': self.waypoints,
            'vehicle_front': self.vehicle_front,
            'crashed': self.isCollided,
            'offroad': self.isOutOfLane,
        }

        # Update timesteps
        self.time_step += 1
        self.total_step += 1
        #         print("time_step：", self.time_step, "total_step:", self.total_step)
        obs = self._get_obs()
        if self.store_coordinate:
            info['coordinate'] = self._get_ego_pos()
        self._out_of_lane()
        reward = self._get_reward()
        terminal = self._terminal()
        #         print("obs:", obs, "\nreward:", reward, "\nterminal:", terminal)
        return obs, reward, terminal, copy.deepcopy(info)

    #         return (obs, reward, terminal, copy.deepcopy(info))

    def render(self):
        pass

    def close(self):
        self._clear_all_actors(['sensor.other.collision', 'vehicle.*', 'controller.ai.walker', 'walker.*'])

    def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
        """生成一个特定类型的车辆blueprint
        Args:
          actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.
        Returns:
          bp: the blueprint object of carla.
        """
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [x for x in blueprints if
                                                     int(x.get_attribute('number_of_wheels')) == nw]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp

    def _init_renderer(self):
        """Initialize the birdeye view renderer.
        """
        pygame.init()
        self.display = pygame.display.set_mode(
            (self.display_size * 3, self.display_size),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        pixels_per_meter = self.display_size / self.obs_range
        pixels_ahead_vehicle = (self.obs_range / 2 - self.d_behind) * pixels_per_meter
        birdeye_params = {
            'screen_size': [self.display_size, self.display_size],
            'pixels_per_meter': pixels_per_meter,
            'pixels_ahead_vehicle': pixels_ahead_vehicle
        }
        self.birdeye_render = BirdeyeRender(self.world, birdeye_params)

    def _set_synchronous_mode(self, synchronous=True):
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)

    def _try_spawn_random_walker_at(self, transform):
        # 在指定位置上生成行人，参数分别是位置transform
        # 返回值：True表示成功生成，否则为False
        walker_bp = random.choice(self.world.get_blueprint_library().filter('walker.*'))
        # set as not invencible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        walker_actor = self.world.try_spawn_actor(walker_bp, transform)
        # 如果能够在指定位置上生成行人，那么为行人设置行为
        if walker_actor is not None:
            # self.actors.append(walker_actor)
            walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
            walker_controller_actor = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker_actor)
            # start walker
            walker_controller_actor.start()
            # set walk to random point
            if (self.world.get_random_location_from_navigation()):
                walker_controller_actor.go_to_location(self.world.get_random_location_from_navigation())
            # random max speed
            walker_controller_actor.set_max_speed(1 + random.random())  # max speed between 1 and 2 (default is 1.4 m/s)
            return True
        return False

    def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
        # 在指定位置上生成车辆，参数分别是位置transform，轮子数量（4表示是汽车）
        # 返回值：True表示成功生成，否则为False
        blueprint = self._create_vehicle_bluepprint('vehicle.*', number_of_wheels=number_of_wheels)
        blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self.world.try_spawn_actor(blueprint, transform)
        # 有可能该位置无法生成车辆
        if vehicle is not None:
            # self.actors.append(vehicle)
            # vehicle.set_autopilot()
            return True
        return False

    def _try_spawn_ego_vehicle_at(self, transform):
        # 尝试在指定位置生成ego车辆，要求为不能够与其他车辆重合
        # 返回值：True表示成功生成，否则为False
        vehicle = None
        # Check if ego position overlaps with surrounding vehicles
        overlap = False
        for idx, poly in self.vehicle_polygons[-1].items():
            poly_center = np.mean(poly, axis=0)
            ego_center = np.array([transform.location.x, transform.location.y])
            dis = np.linalg.norm(poly_center - ego_center)
            if dis > 8:
                continue
            else:
                overlap = True
                break
        if not overlap:
            vehicle = self.world.try_spawn_actor(self.ego_bp, transform)
            # if vehicle is not None:
            #     self.actors.append(vehicle)
        if vehicle is not None:
            self.ego = vehicle
            return True
        return False

    def _get_actor_polygons(self, filt):
        # 得到actor的多边形的边界，其中filt用于判断actor的类型（车辆or行人）
        actor_poly_dict = {}
        for actor in self.world.get_actors().filter(filt):
            # Get x, y and yaw of the actor
            trans = actor.get_transform()
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi
            # Get length and width
            bb = actor.bounding_box
            l = bb.extent.x
            w = bb.extent.y
            # Get bounding box polygon in the actor's local coordinate
            poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
            # Get rotation matrix to transform to global coordinate
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            # Get global bounding box polygon
            poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)
            actor_poly_dict[actor.id] = poly
        return actor_poly_dict

    def _clear_all_actors(self, actor_filters):
        """Clear specific actors."""
        if self.obstacle_sensor is not None:
            self.obstacle_sensor.sensor.destroy()
        for actor_filter in actor_filters:
            # self.client.apply_batch([carla.command.DestroyActor(actor) for actor in self.world.get_actors().filter(actor_filter)])
            for actor in self.world.get_actors().filter(actor_filter):
                try:
                    if actor.is_alive:
                        if actor.type_id == 'controller.ai.walker' or actor.type_id == 'sensor.camera.rgb' or actor.type_id == 'sensor.other.collision':
                            actor.stop()
                        actor.destroy()
                finally:
                    pass

    def get_state(self):
        curr_velocity = self.ego.get_velocity()
        curr_transform = self.ego.get_transform()
        speed = math.sqrt(curr_velocity.x ** 2 + curr_velocity.y ** 2 + curr_velocity.z ** 2)
        curr_waypoint = self.map.get_waypoint(curr_transform.location)
        if curr_waypoint.is_junction and curr_waypoint.get_junction().id == 189:
            self.enter_junction = True
            # print(curr_waypoint.get_junction().id)
        if curr_waypoint is None:
            angle = 0
        else:
            way_rotation = curr_waypoint.transform.rotation
            angle = curr_transform.rotation.yaw - way_rotation.yaw
        vector = carla.Vector3D(0, 0, 0)
        vector.x = curr_transform.location.x - curr_waypoint.transform.location.x
        vector.y = curr_transform.location.y - curr_waypoint.transform.location.y
        offset = curr_transform.location.distance(curr_waypoint.transform.location)
        distance = self.obstacle_sensor.distance
        light_state: carla.TrafficLightState = self.ego.get_traffic_light_state()
        traffic_light = self.ego.get_traffic_light()
        if traffic_light is None:
            sl_distance = 50
        else:
            # print(f'found traffic light, state:{str(traffic_light.get_state())} --- {datetime.datetime.now().timestamp()}')
            waypoint = self.map.get_waypoint(curr_transform.location)
            sl_distance = 74 - waypoint.s
        state = State()
        state.speed = speed
        state.distance = distance
        state.angle = angle
        state.light_state = str(light_state) if traffic_light is not None else 'None'
        state.sl_distance = sl_distance
        state.offset = offset
        return state

    def _get_obs(self):
        ## Birdeye rendering
        # self.birdeye_render.vehicle_polygons = self.vehicle_polygons
        # self.birdeye_render.walker_polygons = self.walker_polygons
        # self.birdeye_render.waypoints = self.waypoints

        # birdeye view with roadmap and actors
        # birdeye_render_types = ['roadmap', 'actors']
        # if self.display_route:
        #     birdeye_render_types.append('waypoints')
        # self.birdeye_render.render(self.display, birdeye_render_types)
        # birdeye = pygame.surfarray.array3d(self.display)
        # birdeye = birdeye[0:self.display_size, :, :]
        # birdeye = display_to_rgb(birdeye, self.obs_size)

        # Roadmap
        # if self.pixor:
        #     roadmap_render_types = ['roadmap']
        #     if self.display_route:
        #         roadmap_render_types.append('waypoints')
        #     self.birdeye_render.render(self.display, roadmap_render_types)
        #     roadmap = pygame.surfarray.array3d(self.display)
        #     roadmap = roadmap[0:self.display_size, :, :]
        #     roadmap = display_to_rgb(roadmap, self.obs_size)
        #     # Add ego vehicle
        #     for i in range(self.obs_size):
        #         for j in range(self.obs_size):
        #             if abs(birdeye[i, j, 0] - 255) < 20 and abs(birdeye[i, j, 1] - 0) < 20 and abs(
        #                     birdeye[i, j, 0] - 255) < 20:
        #                 roadmap[i, j, :] = birdeye[i, j, :]

        # Display birdeye image
        # birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
        # self.display.blit(birdeye_surface, (0, 0))

        ## Lidar image generation
        # point_cloud = []
        # # Get point cloud data
        # for location in self.lidar_data:
        #     point_cloud.append([location.point.x, location.point.y, -location.point.z])
        # point_cloud = np.array(point_cloud)
        # # Separate the 3D space to bins for point cloud, x and y is set according to self.lidar_bin,
        # # and z is set to be two bins.
        # y_bins = np.arange(-(self.obs_range - self.d_behind), self.d_behind + self.lidar_bin, self.lidar_bin)
        # x_bins = np.arange(-self.obs_range / 2, self.obs_range / 2 + self.lidar_bin, self.lidar_bin)
        # z_bins = [-self.lidar_height - 1, -self.lidar_height + 0.25, 1]
        # # Get lidar image according to the bins
        # lidar, _ = np.histogramdd(point_cloud, bins=(x_bins, y_bins, z_bins))
        # lidar[:, :, 0] = np.array(lidar[:, :, 0] > 0, dtype=np.uint8)
        # lidar[:, :, 1] = np.array(lidar[:, :, 1] > 0, dtype=np.uint8)
        # # Add the waypoints to lidar image
        # if self.display_route:
        #     wayptimg = (birdeye[:, :, 0] <= 10) * (birdeye[:, :, 1] <= 10) * (birdeye[:, :, 2] >= 240)
        # else:
        #     wayptimg = birdeye[:, :, 0] < 0  # Equal to a zero matrix
        # wayptimg = np.expand_dims(wayptimg, axis=2)
        # wayptimg = np.fliplr(np.rot90(wayptimg, 3))
        #
        # # Get the final lidar image
        # lidar = np.concatenate((lidar, wayptimg), axis=2)
        # lidar = np.flip(lidar, axis=1)
        # lidar = np.rot90(lidar, 1)
        # lidar = lidar * 255
        #
        # # Display lidar image
        # lidar_surface = rgb_to_display_surface(lidar, self.display_size)
        # self.display.blit(lidar_surface, (self.display_size, 0))

        ## Display camera image
        camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
        # camera_surface = rgb_to_display_surface(camera, self.display_size)
        # self.display.blit(camera_surface, (self.display_size * 2, 0))

        # Display on pygame
        # pygame.display.flip()

        # 从Carla传感器中获取观测数据
        ego_x, ego_y = self._get_ego_pos()
        self.current_wpt = self._get_waypoint_xyz()
        delta_yaw, wpt_yaw, ego_yaw = self._get_delta_yaw()
        road_heading = np.array([np.cos(wpt_yaw / 180 * np.pi), np.sin(wpt_yaw / 180 * np.pi)])
        ego_heading = np.float32(ego_yaw / 180.0 * np.pi)
        ego_heading_vec = np.array((np.cos(ego_heading), np.sin(ego_heading)))
        future_angles = self._get_future_wpt_angle(distances=self.distances)

        velocity = self.ego.get_velocity()
        accel = self.ego.get_acceleration()
        dyaw_dt = self.ego.get_angular_velocity().z
        v_t_absolute = np.array([velocity.x, velocity.y])
        a_t_absolute = np.array([accel.x, accel.y])

        v_t = _vec_decompose(v_t_absolute, ego_heading_vec)
        a_t = _vec_decompose(a_t_absolute, ego_heading_vec)

        pos_err_vec = np.array((ego_x, ego_y)) - self.current_wpt[0:2]
        self.state_info['velocity_t'] = v_t
        self.state_info['acceleration_t'] = a_t
        self.state_info['delta_yaw_t'] = delta_yaw
        self.state_info['dyaw_dt_t'] = dyaw_dt
        self.state_info['lateral_dist_t'] = np.linalg.norm(pos_err_vec) * \
                                            np.sign(pos_err_vec[0] * road_heading[1] - \
                                                    pos_err_vec[1] * road_heading[0])
        self.state_info['action_t_1'] = self.last_action
        self.state_info['angles_t'] = future_angles

        velocity_t = self.state_info['velocity_t']
        accel_t = self.state_info['acceleration_t']

        delta_yaw_t = np.array(self.state_info['delta_yaw_t']).reshape(
            (1,)) / 2.0
        dyaw_dt_t = np.array(self.state_info['dyaw_dt_t']).reshape((1,)) / 5.0

        lateral_dist_t = self.state_info['lateral_dist_t'].reshape(
            (1,)) * 10.0
        action_last = self.state_info['action_t_1'] * 10.0

        future_angles = self.state_info['angles_t'] / 2.0

        self.state_info['next_state'] = self.get_state()
        next_state = self.state_info['next_state']
        speed = np.array(next_state.speed).reshape((1,))
        angle = np.array(next_state.angle).reshape((1,))
        offset = np.array(next_state.offset).reshape((1,))
        e_speed = np.array(next_state.e_speed).reshape((1,))
        distance = np.array(next_state.distance).reshape((1,))
        e_distance = np.array(next_state.e_distance).reshape((1,))
        safe_distance = np.array(next_state.safe_distance).reshape((1,))

        if next_state.light_state == 'Green':
            light_state = np.array(1).reshape((1,))
        elif next_state.light_state == 'red':
            light_state = np.array(-1).reshape((1,))
        else:
            light_state = np.array(0).reshape((1,))
        sl_distance = np.array(next_state.sl_distance).reshape((1,))
        # print(velocity_t, "------", accel_t, "------", delta_yaw_t, "------", dyaw_dt_t, "------", lateral_dist_t, "------",
        #     action_last, "------", future_angles, "------", speed, "------", angle)
        info_vec = np.concatenate([
            velocity_t, accel_t, delta_yaw_t, dyaw_dt_t, lateral_dist_t,
            action_last, future_angles, speed, angle, offset, e_speed, distance,
            e_distance, safe_distance, light_state, sl_distance
        ],
            axis=0)
        info_vec = info_vec.squeeze()

        # if self.pixor:
        #     ## Vehicle classification and regression maps (requires further normalization)
        #     vh_clas = np.zeros((self.pixor_size, self.pixor_size))
        #     vh_regr = np.zeros((self.pixor_size, self.pixor_size, 6))
        #
        #     # Generate the PIXOR image. Note in CARLA it is using left-hand coordinate
        #     # Get the 6-dim geom parametrization in PIXOR, here we use pixel coordinate
        #     for actor in self.world.get_actors().filter('vehicle.*'):
        #         x, y, yaw, l, w = get_info(actor)
        #         x_local, y_local, yaw_local = get_local_pose((x, y, yaw), (ego_x, ego_y, ego_yaw))
        #         if actor.id != self.ego.id:
        #             if abs(y_local) < self.obs_range / 2 + 1 and x_local < self.obs_range - self.d_behind + 1 and x_local > -self.d_behind - 1:
        #                 x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel = get_pixel_info(
        #                     local_info=(x_local, y_local, yaw_local, l, w),
        #                     d_behind=self.d_behind, obs_range=self.obs_range, image_size=self.pixor_size)
        #                 cos_t = np.cos(yaw_pixel)
        #                 sin_t = np.sin(yaw_pixel)
        #                 logw = np.log(w_pixel)
        #                 logl = np.log(l_pixel)
        #                 pixels = get_pixels_inside_vehicle(
        #                     pixel_info=(x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel),
        #                     pixel_grid=self.pixel_grid)
        #                 for pixel in pixels:
        #                     vh_clas[pixel[0], pixel[1]] = 1
        #                     dx = x_pixel - pixel[0]
        #                     dy = y_pixel - pixel[1]
        #                     vh_regr[pixel[0], pixel[1], :] = np.array(
        #                         [cos_t, sin_t, dx, dy, logw, logl])
        #
        #     # Flip the image matrix so that the origin is at the left-bottom
        #     vh_clas = np.flip(vh_clas, axis=0)
        #     vh_regr = np.flip(vh_regr, axis=0)
        #
        #     # Pixor state, [x, y, cos(yaw), sin(yaw), speed]
        #     pixor_state = [ego_x, ego_y, np.cos(ego_yaw), np.sin(ego_yaw), speed]

        obs = {
            'camera': camera.astype(np.uint8),
            # 'lidar': lidar.astype(np.uint8),
            # 'birdeye': birdeye.astype(np.uint8),
            'state': np.float32(info_vec),
        }
        # if self.pixor:
        #     obs.update({
        #         'roadmap': roadmap.astype(np.uint8),
        #         'vh_clas': np.expand_dims(vh_clas, -1).astype(np.float32),
        #         'vh_regr': vh_regr.astype(np.float32),
        #         'pixor_state': pixor_state,
        #     })

        return obs

    def _get_reward(self):

        state = self.get_state()
        r1 = - (k1 * math.sin(math.radians(state.angle)) + k2 * abs(state.offset))
        r2 = - (k3 * abs(state.distance - state.e_distance))
        # curr_waypoint = self.map.get_waypoint(self.player.get_transform().location)
        if self.isCollided or self.isOutOfLane or self.obstacle_sensor.distance < safe_distance:
            r_done = -500.0
            return r_done
        elif self.enter_junction:
            r3 = 100
            # is_end = True
        else:
            r3 = - (k2 * abs(state.distance - state.safe_distance))
        """Calculate the step reward."""
        r_step = 10.0
        r_reach = 0
        # 若发生碰撞或者超出车道
        # if self.isCollided or self.isOutOfLane:  # or self.isSpecialSpeed
        #     r_done = -500.0
        #     return r_done

        if self.reachDest:
            r_reach = 300.0
        ###################
        # # reward for speed tracking
        v = self.ego.get_velocity()
        speed = np.sqrt(v.x ** 2 + v.y ** 2)
        # delta_speed = -abs(speed - self.desired_speed)
        # r_speed = -delta_speed ** 2 / 5.0
        r_speed = 0
        if speed < 1:
            r_speed = -20.0
        #
        # # reward for steering:
        # delta_yaw, _, _ = self._get_delta_yaw()
        # r_steer = -100 * (delta_yaw * np.pi / 180) ** 2
        #
        # # reward for action smoothness
        # # r_action_regularized = -5 * np.linalg.norm(action) ** 2
        #
        # # cost for lateral acceleration
        # # r_lat = - abs(self.ego.get_control().steer) * lspeed_lon ** 2
        #
        # # reward for lateral distance to the center of road
        # lateral_dist = self.state_info['lateral_dist_t']
        # r_lateral = -10.0 * lateral_dist ** 2
        ###################

        r4 = - (k1 * math.sin(math.radians(state.angle)) + k2 * abs(state.offset))
        light_state = str(self.ego.get_traffic_light_state())
        # light = self.ego.get_traffic_light()
        curr_transform = self.ego.get_transform()
        curr_waypoint = self.map.get_waypoint(curr_transform.location)
        next_waypoint = curr_waypoint.next(10)[0]
        # print(f'light state:{light_state}, is junction:{next_waypoint.is_junction}')
        if next_waypoint.is_junction \
                and next_waypoint.get_junction().id == 189 \
                and light_state == 'Green' and abs(state.speed) < 0.001:
            r5 = -100
            # is_end = True
            # print('stop green')
        elif state.light_state == 'Red' and ((state.speed ** 2) / (2 * state.sl_distance)) > 2.778:
            r5 = -100
            # is_end = True
            # print('acc red')
        elif not curr_waypoint.is_junction and self.enter_junction:
            r5 = 100
            # is_end = True
        else:
            r5 = 0

        # if state.light_state is not None:
        #     return r4 + r5 - r_step + r_reach
        # else:
        #     return r1 + r2 + r3 - r_step + r_reach
        # return r1 + r2 + r3 + r4 + r5 + r_step + r_reach
        # return r_speed + r_steer + r_lateral + r_step + r_reach
        return r1 + r2 + r3 + r4 + r5 + r_step + r_reach + r_speed

    def _get_future_wpt_angle(self, distances):
        angles = []
        current_wpt = self.map.get_waypoint(location=self.ego.get_location())
        if not current_wpt:
            self.logger.error('Fail to find a waypoint')
            current_road_heading = self.current_wpt[3]
        else:
            current_road_heading = current_wpt.transform.rotation.yaw

        for d in distances:
            wpt_heading = current_wpt.next(d)[0].transform.rotation.yaw
            delta_heading = delta_angle_between(current_road_heading,
                                                wpt_heading)
            angles.append(delta_heading)

        return np.array(angles, dtype=np.float32)

    def _make_carla_client(self, host, port):
        while True:
            try:
                self.logger.info("connecting to Carla server...")
                self.client = carla.Client(host, port)
                #                 print(self.task_mode)
                self.client.set_timeout(10.0)
                # 根据当前的任务模式设置地图
                if self.task_mode == 'Straight':
                    self.world = self.client.load_world('Town01')
                elif self.task_mode == 'Curve':
                    self.world = self.client.load_world('Town05')
                elif self.task_mode == 'Long':
                    self.world = self.client.load_world('Town01')
                elif self.task_mode == 'Lane':
                    self.world = self.client.load_world('Town05')
                elif self.task_mode == 'U_curve':
                    self.world = self.client.load_world('Town03')
                elif self.task_mode == 'Lane_test':
                    self.world = self.client.load_world('Town03')
                elif self.task_mode == 'TrafficLight':
                    self.world = self.client.get_world()
                self.spectator = self.world.get_spectator()
                self.map = self.world.get_map()  # 得到地图
                self.world.set_weather(carla.WeatherParameters.ClearNoon)  # 设置天气
                self.logger.info("Carla server port {} connected!".format(port))
                break
            except Exception as exc:
                print(traceback.format_exc())
                self.logger.error('Fail to connect to carla-server...sleeping for 2')
                time.sleep(2)

    # 设置transform
    def _set_carla_transform(self, pose):
        transform = carla.Transform()
        transform.location.x = pose[0]
        transform.location.y = pose[1]
        transform.location.z = pose[2]
        transform.rotation.pitch = pose[3]
        transform.rotation.roll = pose[4]
        transform.rotation.yaw = pose[5]
        return transform

    # 得到ego车辆当前的位置
    def _get_ego_pos(self):
        ego_trans = self.ego.get_transform()
        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        return ego_x, ego_y

    # 得到当前车辆所在位置的waypoint
    def _get_waypoint_xyz(self):
        waypoint = self.map.get_waypoint(location=self.ego.get_location())
        if waypoint:
            return np.array(
                (waypoint.transform.location.x, waypoint.transform.location.y,
                 waypoint.transform.rotation.yaw))
        else:
            return self.current_wpt

    def _get_delta_yaw(self):
        current_wpt = self.map.get_waypoint(location=self.ego.get_location())
        if not current_wpt:
            self.logger.error('Fail to find a waypoint')
            wpt_yaw = self.current_wpt[2] % 360
        else:
            wpt_yaw = current_wpt.transform.rotation.yaw % 360
        ego_yaw = self.ego.get_transform().rotation.yaw % 360
        delta_yaw = ego_yaw - wpt_yaw
        if 180 <= delta_yaw and delta_yaw <= 360:
            delta_yaw -= 360
        elif -360 <= delta_yaw and delta_yaw <= -180:
            delta_yaw += 360

        return delta_yaw, wpt_yaw, ego_yaw

    def _get_random_position_between(self, start, dest, transform):
        if self.task_mode == 'Straight' or self.task_mode == 'TrafficLight':
            ratio = float(np.random.rand() * 30)
        elif self.task_mode == 'Curve':
            ratio = float(np.random.rand() * 45)
        elif self.task_mode == 'Long':
            ratio = float(np.random.rand() * 60)
        elif self.task_mode == 'Lane':
            ratio = float((1/3 + np.random.rand()/3) * 60)
            # ratio = float(np.random.rand() * 60)
        else:
            ratio = 0
        transform = self.get_position(start, ratio)
        return transform

    def get_position(self, start, distance):
        start_location = carla.Location(x=start[0], y=start[1], z=0.22)
        transform = self.map.get_waypoint(location=start_location).next(distance)[0].transform
        transform.location.z = start[2]
        return transform

    def _out_of_lane(self):
        if abs(self.state_info['lateral_dist_t']) > 1.2 + self.out_lane_thres:
            self.isOutOfLane = True
        else:
            self.isOutOfLane = False

    def _terminal(self):
        ego_x, ego_y = get_pos(self.ego)

        # If collides,发生碰撞
        if len(self.collision_hist) > 0:
            self.logger.debug(
                'Collision happened! Episode cost %d steps in route %d.' %
                (self.time_step, self.route_id))
            self.isCollided = True
            return True

        # If reach maximum timestep，到达最大次数
        if self.time_step > self.max_time_episode:
            self.logger.debug('Time out! Episode cost %d steps in route %d.' %
                              (self.time_step, self.route_id))
            self.isTimeOut = True
            return True

        # If out of lane，超出车道
        if self.isOutOfLane:
            print("lane invasion happened! Episode Done.")
            if self.state_info['lateral_dist_t'] > 0:
                self.logger.debug(
                    'Left Lane invasion! Episode cost %d steps in route %d.' %
                    (self.time_step, self.route_id))
            else:
                self.logger.debug(
                    'Right Lane invasion! Episode cost %d steps in route %d.' %
                    (self.time_step, self.route_id))
            return True

        # If at destination，到达目的地
        if self.dests is not None:  # If at destination
            for dest in self.dests:
                if np.sqrt((ego_x - dest[0]) ** 2 + (ego_y - dest[1]) ** 2) < 4:
                    self.reachDest = True
                    return True
        return False

    # def close(self):
    #     if self.obstacle_sensor is not None:
    #         self.obstacle_sensor.sensor.destroy()
    #     while self.actors:
    #         (self.actors.pop()).destroy()

import copy
import numpy as np
import random
import time
from skimage.transform import resize

import gym
from gym import spaces
from gym.utils import seeding

from carla_env.misc import *
from carla_env.route_planner import RoutePlanner

#导入Carla
import glob
import os
import sys
try:
    sys.path.append(glob.glob('./carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
'''
{'obs_size': (160, 100), 'max_past_step': 1, 'dt': 0.025, 'ego_vehicle_filter': 'vehicle.lincoln*', 'port': 2021, 'task_mode': 'Lane', 'code_mode': 'train', 'max_time_episode': 250, 'desired_speed': 15, 'max_ego_spawn_times': 100, 'number_of_vehicles': 3, 'number_of_walkers': 5, 'max_waypt': 20, 'obs_range': 32, 'lidar_bin': 0.125, 'd_behind': 12, 'out_lane_thres': 2.0, 'display_route': False, 'discrete': False, 'discrete_acc': [-3.0, 0.0, 3.0], 'discrete_steer': [-0.2, 0.0, 0.2], 'continuous_accel_range': [-3.0, 3.0], 'continuous_steer_range': [-0.3, 0.3]},
'''


class CarlaEnv(gym.Env):
     def __init__(self,params):
        self.obs_size = params['obs_size']  # rendering screen size
        self.max_past_step = params['max_past_step']
        self.port = params['port']
        self.number_of_vehicles = params['number_of_vehicles']
        self.number_of_walkers = params['number_of_walkers']
        self.dt = params['dt'] #时间间隔
        self.task_mode = params['task_mode']
        self.max_time_episode = params['max_time_episode']
        self.max_waypt = params['max_waypt'] #waypoint数量--地图上一个点，道路信息Transform
        self.obs_range = params['obs_range'] #观察距离
        self.lidar_bin = params['lidar_bin'] #雷达看到的距离
        self.d_behind = params['d_behind'] 
        self.obs_size = int(self.obs_range/self.lidar_bin) #观察距离/雷达看到的距离
        self.out_lane_thres = params['out_lane_thres'] #车道数量
        self.desired_speed = params['desired_speed']
        self.max_ego_spawn_times = params['max_ego_spawn_times']
        self.display_route = params['display_route'] #是否渲染路线
        
        #目的地
        if params['task_mode'] == 'roundabout':
          self.dests = [[4.46, -61.46, 0], [-49.53, -2.89, 0], [-6.48, 55.47, 0], [35.96, 3.33, 0]]
        else:
          self.dests = None
        
        #定义动作空间
        self.discrete = params['discrete']
        self.discrete_act = [params['discrete_acc'], params['discrete_steer']] # acc, steer
        self.n_acc = len(self.discrete_act[0])
        self.n_steer = len(self.discrete_act[1])
        #如果动作空间是离散的
        if self.discrete:
          self.action_space = spaces.Discrete(self.n_acc*self.n_steer) #通过spaces.Discrete(）转换为gym中的离散动作空间
        else:
          #连续动作，gym：spaces.Box 描述的为一个n维的实数空间 [公式] ，可以指定上下限，也可以不指定上下限。
          #'continuous_accel_range': [-3.0, 3.0]
          #'continuous_steer_range': [-0.3, 0.3],
          self.action_space = spaces.Box(np.array([params['continuous_accel_range'][0], 
          params['continuous_steer_range'][0]]), np.array([params['continuous_accel_range'][1],
          params['continuous_steer_range'][1]]), dtype=np.float32)  # acc, steer
        #定义观测空间
        observation_space_dict = {
#           'camera': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
#           'lidar': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
          'state': spaces.Box(np.array([-2, -1, -5, 0]), np.array([2, 1, 30, 1]), dtype=np.float32)
         }
        self.observation_space = spaces.Dict(observation_space_dict)
        #print(self.observation_space)
        
        # 连接Carla并获得world
        print('connecting to Carla server...')
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        self.world = client.load_world('Town03')
        print('Carla server connected!')
        
        # Set weather,设置天气
        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
        #print(self.vehicle_spawn_points)
        self.walker_spawn_points = []
        self.walker_spawn_points = []
        #随机生成行人的位置 spawn_point
        for i in range(self.number_of_walkers):
          spawn_point = carla.Transform()
          loc = self.world.get_random_location_from_navigation()
          if (loc != None):
            spawn_point.location = loc
            self.walker_spawn_points.append(spawn_point)
        #print(self.walker_spawn_points)
        
        #创建blueprint
        # 1.ego车辆
        #'ego_vehicle_filter': 'vehicle.lincoln*',表示车辆的外观
        self.ego_bp = self._create_vehicle_bluepprint(params['ego_vehicle_filter'], color='49,8,8')
#         print(self.ego_bp)
        
        # 2.Collision sensor,碰撞
        self.collision_hist = [] # The collision history
        self.collision_hist_l = 1 # collision history length
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
        
        # Set fixed simulation step for synchronous mode，设置时间间隔
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.dt

        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0
     
     def reset(self): 
        # 1.清空传感器  
        self.collision_sensor = None
        self.lidar_sensor = None
        self.camera_sensor = None
        
        # 2.清除所有的行人、路人、传感器
        self._clear_all_actors(['sensor.other.collision', 'sensor.lidar.ray_cast', 'sensor.camera.rgb', 'vehicle.*', 'controller.ai.walker', 'walker.*'])
        # 3.取消同步模式
        self._set_synchronous_mode(False)
        
        # 4.在随机位置上生成周围车辆
        random.shuffle(self.vehicle_spawn_points)
        count = self.number_of_vehicles
        if count > 0:
          for spawn_point in self.vehicle_spawn_points:
            if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4]):
              count -= 1
            if count <= 0:
              break
        while count > 0:
          if self._try_spawn_random_vehicle_at(random.choice(self.vehicle_spawn_points), number_of_wheels=[4]):
            count -= 1
        print("成功生成周围车辆！")  
        
        # 5.Spawn pedestrians，在随机位置上生成行人
        random.shuffle(self.walker_spawn_points)
        count = self.number_of_walkers
        if count > 0:
          for spawn_point in self.walker_spawn_points:
            if self._try_spawn_random_walker_at(spawn_point):
              count -= 1
            if count <= 0:
              break
        while count > 0:
          if self._try_spawn_random_walker_at(random.choice(self.walker_spawn_points)):
            count -= 1
        print("成功生成周围行人！")  
        
        # 得到周围车辆、行人的多边形边界-->可以用于判断ego车是否会与周围车辆重合
        self.vehicle_polygons = []
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        self.walker_polygons = []
        walker_poly_dict = self._get_actor_polygons('walker.*')
        self.walker_polygons.append(walker_poly_dict)

        # 6.Spawn the ego vehicle 随机位置上生成ego车辆
        ego_spawn_times = 0
        while True:
          if ego_spawn_times > self.max_ego_spawn_times:
            self.reset()
          if self.task_mode == 'random':
            transform = random.choice(self.vehicle_spawn_points)# 设置ego的起始位置
          if self.task_mode == 'roundabout':
            self.start=[52.1+np.random.uniform(-5,5),-4.2, 178.66] # random
            # self.start=[52.1,-4.2, 178.66] # static
            transform = set_carla_transform(self.start)
          if self._try_spawn_ego_vehicle_at(transform):
            break
          else:
            ego_spawn_times += 1
            time.sleep(0.1)
        print("成功生成ego车辆！")  
        
        # 7.Add collision sensor，添加检测碰撞传感器,并监听碰撞事件存储碰撞强度
        self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
        self.collision_sensor.listen(lambda event: get_collision_hist(event))
        
        def get_collision_hist(event):
          impulse = event.normal_impulse
          intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
          self.collision_hist.append(intensity)
          if len(self.collision_hist)>self.collision_hist_l:
            self.collision_hist.pop(0)
        self.collision_hist = []
        print("成功生成检测碰撞传感器！")
        
        # 8.Add lidar sensor,添加检测lidar传感器，存储data
        self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego)
        self.lidar_sensor.listen(lambda data: get_lidar_data(data))
        def get_lidar_data(data):
          self.lidar_data = data
        print("成功生成lidar传感器！") 
        
        # 9.Add camera sensor，添加camera传感器，并存储img
        self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)
        self.camera_sensor.listen(lambda data: get_camera_img(data))
        def get_camera_img(data):
          array = np.frombuffer(data.raw_data, dtype = np.dtype("uint8"))
          array = np.reshape(array, (data.height, data.width, 4))
          array = array[:, :, :3]
          array = array[:, :, ::-1]
          self.camera_img = array
        print("成功生成camera传感器！") 
        
        # 10.Update timesteps
        self.time_step=0
        self.reset_step+=1
#         print(self.time_step,self.reset_step)
        
        # 11.Enable sync mode 同步模式：服务器会等待客户端命令
        self.settings.synchronous_mode = True
        self.world.apply_settings(self.settings)

        # 12.设置car的路线
        self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
        self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

        #x Set ego information for render
        #self.birdeye_render.set_hero(self.ego, self.ego.id)
        obs = self._get_obs()
        print('当前状态是：')
        print(obs)
        return obs #gym返回
        
   
     def step(self,action):
        #计算加速度和方向盘角度
        if self.discrete:
          acc = self.discrete_act[0][action//self.n_steer]
          steer = self.discrete_act[1][action%self.n_steer]
        else:
          acc = action[0]
          steer = action[1]
        # 将加速度转化为刹车和油门---其中throttle [0.0, 1.0]，steer[-1,1],brake[0,1]之间的标量
        if acc > 0:
          throttle = np.clip(acc/3,0,1) ##这里除3，要看情况
          brake = 0
        else:
          throttle = 0
          brake = np.clip(-acc/8,0,1) ##这里除8，要看情况

        # Apply control
        act = carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))
        self.ego.apply_control(act)

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
          'vehicle_front': self.vehicle_front
        }

        # Update timesteps
        self.time_step += 1
        self.total_step += 1
        print("time_step：",self.time_step,"total_step:",self.total_step)
        obs = self._get_obs()
        reward = self._get_reward()
        terminal = self._terminal()
        print("obs:",obs,"\nreward:",reward,"\nterminal:",terminal)
        return (obs, reward, terminal, copy.deepcopy(info))     
    
     def close(self):
        pass
    
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
          blueprint_library = blueprint_library + [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == nw]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
          if not color:
            color = random.choice(bp.get_attribute('color').recommended_values)
          bp.set_attribute('color', color)
        return bp
    
     def _set_synchronous_mode(self, synchronous = True):
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)
    
     def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
        #在指定位置上生成车辆，参数分别是位置transform，轮子数量（4表示是汽车）
        #返回值：True表示成功生成，否则为False
        blueprint = self._create_vehicle_bluepprint('vehicle.*', number_of_wheels=number_of_wheels)
        blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self.world.try_spawn_actor(blueprint, transform)
        #有可能该位置无法生成车辆
        if vehicle is not None:
          vehicle.set_autopilot()
          return True
        return False
   
     def _try_spawn_random_walker_at(self, transform):
        #在指定位置上生成行人，参数分别是位置transform
        #返回值：True表示成功生成，否则为False
        walker_bp = random.choice(self.world.get_blueprint_library().filter('walker.*'))
        # set as not invencible
        if walker_bp.has_attribute('is_invincible'):
          walker_bp.set_attribute('is_invincible', 'false')
        walker_actor = self.world.try_spawn_actor(walker_bp, transform)
        #如果能够在指定位置上生成行人，那么为行人设置行为
        if walker_actor is not None:
          walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
          walker_controller_actor = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker_actor)
          # start walker
          walker_controller_actor.start()
          # set walk to random point
          walker_controller_actor.go_to_location(self.world.get_random_location_from_navigation())
          # random max speed
          walker_controller_actor.set_max_speed(1 + random.random())    # max speed between 1 and 2 (default is 1.4 m/s)
          return True
        return False
   
     def _try_spawn_ego_vehicle_at(self, transform):
        #尝试在指定位置生成ego车辆，要求为不能够与其他车辆重合
        #返回值：True表示成功生成，否则为False
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
        if vehicle is not None:
          self.ego=vehicle
          return True
        return False
    
     def _get_actor_polygons(self, filt):
        #得到actor的多边形的边界，其中filt用于判断actor的类型（车辆or行人）
        actor_poly_dict={}
        for actor in self.world.get_actors().filter(filt):
          # Get x, y and yaw of the actor
          trans=actor.get_transform()
          x=trans.location.x
          y=trans.location.y
          yaw=trans.rotation.yaw/180*np.pi
          # Get length and width
          bb=actor.bounding_box
          l=bb.extent.x
          w=bb.extent.y
          # Get bounding box polygon in the actor's local coordinate
          poly_local=np.array([[l,w],[l,-w],[-l,-w],[-l,w]]).transpose()
          # Get rotation matrix to transform to global coordinate
          R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
          # Get global bounding box polygon
          poly=np.matmul(R,poly_local).transpose()+np.repeat([[x,y]],4,axis=0)
          actor_poly_dict[actor.id]=poly
        return actor_poly_dict    
    
     def _clear_all_actors(self, actor_filters):
        """Clear specific actors."""
        for actor_filter in actor_filters:
          for actor in self.world.get_actors().filter(actor_filter):
            if actor.is_alive:
              if actor.type_id == 'controller.ai.walker':
                actor.stop()
              actor.destroy()
     
     def _get_obs(self):
        #从Carla传感器中获取观测数据
        ego_trans = self.ego.get_transform()
        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        ego_yaw = ego_trans.rotation.yaw/180*np.pi
        # 计算当前最近的waypoint离ego车辆的距离以及方向
        lateral_dis, w = get_preview_lane_dis(self.waypoints, ego_x, ego_y)
        # z轴旋转变化的角度
        delta_yaw = np.arcsin(np.cross(w, 
          np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))
        v = self.ego.get_velocity()
        #速度
        speed = np.sqrt(v.x**2 + v.y**2)
        #【ego车与最近waypoint之间的距离，z轴旋转变化的角度，车辆速度，前方是否有车辆(0-无，1-有)】
        state = np.array([lateral_dis, - delta_yaw, speed, self.vehicle_front])
        obs = {
#           'camera':camera.astype(np.uint8),
#           'lidar':lidar.astype(np.uint8),
#           'birdeye':birdeye.astype(np.uint8),
          'state': state,
         }
        return obs
    
     def _get_reward(self):
        """Calculate the step reward."""
        # reward for speed tracking
        v = self.ego.get_velocity()
        speed = np.sqrt(v.x**2 + v.y**2)
        r_speed = -abs(speed - self.desired_speed)

        # reward for collision
        r_collision = 0
        if len(self.collision_hist) > 0:
          r_collision = -1

        # reward for steering:
        r_steer = -self.ego.get_control().steer**2

        # reward for out of lane
        ego_x, ego_y = get_pos(self.ego)
        dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
        r_out = 0
        if abs(dis) > self.out_lane_thres:
          r_out = -1

        # longitudinal speed
        lspeed = np.array([v.x, v.y])
        lspeed_lon = np.dot(lspeed, w)

        # cost for too fast
        r_fast = 0
        if lspeed_lon > self.desired_speed:
          r_fast = -1

        # cost for lateral acceleration
        r_lat = - abs(self.ego.get_control().steer) * lspeed_lon**2

        r = 200*r_collision + 1*lspeed_lon + 10*r_fast + 1*r_out + r_steer*5 + 0.2*r_lat - 0.1

        return r
        
  
     def _terminal(self):
        """Calculate whether to terminate the current episode."""
        # Get ego state
        ego_x, ego_y = get_pos(self.ego)

        # If collides
        if len(self.collision_hist)>0: 
          return True

        # If reach maximum timestep
        if self.time_step>self.max_time_episode:
          return True

        # If at destination
        if self.dests is not None: # If at destination
          for dest in self.dests:
            if np.sqrt((ego_x-dest[0])**2+(ego_y-dest[1])**2)<4:
              return True

        # If out of lane
        dis, _ = get_lane_dis(self.waypoints, ego_x, ego_y)
        if abs(dis) > self.out_lane_thres:
          return True

        return False
        
     
 
            
            

    
        
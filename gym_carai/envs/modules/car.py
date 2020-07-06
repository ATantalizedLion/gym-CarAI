import pyglet
import numpy as np

from gym_carai.envs.modules.util import center_image, Line, vector_length, Circle
from pyglet.window import key


class Bumper(Line):
    def __init__(self, pos, debug_batch):
        super().__init__(pos)
        self.set_image('RedBar.png')
        self.sprite.batch = debug_batch


class Sensor(Line):
    def __init__(self, pos, debug_batch, sensor_range):
        super().__init__(pos, height=2)
        self.set_image('BlueBar.png')
        self.sprite.batch = debug_batch
        self.colimg = pyglet.resource.image('marker.png')
        center_image(self.colimg)
        self.collision_marker = pyglet.sprite.Sprite(img=self.colimg, x=-10, y=-10, batch=debug_batch)
        self.sensor_range = sensor_range

class Car:
    def __init__(self, initial_position=(100, 100, 0), car_length=64, main_batch=[], debug_batch=[], mode=[]):
        self.debug_batch = debug_batch
        self.image = pyglet.resource.image("car.png")
        center_image(self.image)
        self.initPos = initial_position
        self.x = initial_position[0]
        self.y = initial_position[1]
        self.xc = self.x
        self.yc = self.y
        self.c = np.array([self.xc, self.yc])
        self.sprite = pyglet.sprite.Sprite(img=self.image, x=self.x, y=self.y)
        self.sprite.scale = car_length / self.image.height
        self.width = self.sprite.width
        self.height = self.sprite.height
        self.rotation = initial_position[2]
        self.rotation_rad = np.deg2rad(self.rotation)
        self.sprite.rotation = self.rotation
        self.vel = 60.0
        self.acc = 0.0

        self.mode = mode
        self.sensorDistance = 1920

        # related to controls
        self.key_handler = key.KeyStateHandler()
        self.rotate_speed = 200.0
        self.acc_speed = 300.0

        # set up bumpers
        front_corner1 = self.c + np.cos(self.rotation_rad) * np.array([-self.width / 2, self.height / 2])
        front_corner2 = self.c + np.cos(self.rotation_rad) * np.array([self.width / 2, self.height / 2])
        rear_corner1 = self.c - np.cos(self.rotation_rad) * np.array([self.width / 2, self.height / 2])
        rear_corner2 = self.c - np.cos(self.rotation_rad) * np.array([-self.width / 2, self.height / 2])
        self.Bumper = Bumper([front_corner1[0], front_corner1[1], front_corner2[0], front_corner2[1]], self.debug_batch)
        self.SideL = Bumper([front_corner1[0], front_corner1[1], rear_corner1[0], rear_corner1[1]], self.debug_batch)
        self.SideR = Bumper([front_corner2[0], front_corner2[1], rear_corner2[0], rear_corner2[1]], self.debug_batch)
        self.Rear = Bumper([rear_corner1[0], rear_corner1[1], rear_corner2[0], rear_corner2[1]], self.debug_batch)

        # set up distance sensors
        fc = self.c + np.array([0.0, self.height / 2]) - np.array([-self.height / 2, 0.0])
        rc = self.c - np.array([0.0, self.height / 2]) + np.array([-self.height / 2, 0.0])
        sl = self.c - np.array([0.0, self.width / 2]) - np.array([-self.width / 2, 0.0])
        sr = self.c + np.array([0.0, self.width / 2]) + np.array([-self.width / 2, 0.0])

        self.FrontDistanceSensor = Sensor([fc[0], fc[1], fc[0], fc[1] + self.sensorDistance], self.debug_batch,
                                          self.sensorDistance)
        self.RearDistanceSensor = Sensor([rc[0], rc[1], rc[0], rc[1] - self.sensorDistance], self.debug_batch,
                                         self.sensorDistance)
        self.LeftDistanceSensor = Sensor([sl[0], sl[1], sl[0] - self.sensorDistance, sl[1]], self.debug_batch,
                                         self.sensorDistance)
        self.RightDistanceSensor = Sensor([sr[0], sr[1], sr[0] + self.sensorDistance, sr[1]], self.debug_batch,
                                          self.sensorDistance)

    def update(self, dt, action):
        """"
        action [0] = -1 to 1, steering
        action [1] = -1 to 1, gas
        action [2] = brake (bool)
        """
        if self.mode == 'simple':
            rot = action[0]-1
            self.rotation += rot * self.rotate_speed * dt
            self.vel = 90
        else:
            if abs(action[1]) > 1:
                action[1] = action[1]/abs(action[1])
            self.rotation += action[0] * self.rotate_speed * dt
            self.vel += self.acc_speed * dt * action[1]
            if self.vel > 0.0:
                self.vel += - min(2.0 * self.vel, 10.0 * self.acc_speed) * dt * action[2]
            else:
                self.vel += min(2.0 * self.vel, 10.0 * self.acc_speed) * dt * action[2]

        self.rotation_rad = np.deg2rad(self.rotation)
        cos = np.cos(self.rotation_rad)
        sin = np.sin(self.rotation_rad)
        self.vel += self.acc * dt

        self.x += self.vel * np.sin(self.rotation_rad) * dt
        self.y += self.vel * np.cos(self.rotation_rad) * dt

        self.sprite.x = self.x
        self.sprite.y = self.y
        self.sprite.rotation = self.rotation

        self.xc = self.x
        self.yc = self.y
        self.c = np.array([self.xc, self.yc])

        # Calculate bumper positions based on rotation and center position
        fc = self.c + cos * np.array([0.0, self.height / 2]) \
             - sin * np.array([-self.height / 2, 0.0])
        rc = self.c - cos * np.array([0.0, self.height / 2]) \
             + sin * np.array([-self.height / 2, 0.0])
        sl = self.c - sin * np.array([0.0, self.width / 2]) \
             - cos * np.array([-self.width / 2, 0.0])
        sr = self.c + sin * np.array([0.0, self.width / 2]) \
             + cos * np.array([-self.width / 2, 0.0])
        self.Bumper.update_position([fc[0], fc[1], self.rotation])
        self.Rear.update_position([rc[0], rc[1], self.rotation])
        self.SideL.update_position([sl[0], sl[1], self.rotation - 90])
        self.SideR.update_position([sr[0], sr[1], self.rotation - 90])

        # Calculate line based sensor position
        self.FrontDistanceSensor.update_position([fc[0], fc[1], self.rotation - 90])
        self.RearDistanceSensor.update_position([rc[0], rc[1], self.rotation + 90])
        self.RightDistanceSensor.update_position([sl[0], sl[1], self.rotation])
        self.LeftDistanceSensor.update_position([sr[0], sr[1], self.rotation + 180])

        # Calculate line based sensor position
        self.FrontDistanceSensor.update_position_x1y1([fc[0], fc[1], self.rotation - 90])
        self.RearDistanceSensor.update_position_x1y1([rc[0], rc[1], self.rotation + 90])
        self.RightDistanceSensor.update_position_x1y1([sl[0], sl[1], self.rotation])
        self.LeftDistanceSensor.update_position_x1y1([sr[0], sr[1], self.rotation + 180])

    def reset(self):
        self.x = self.initPos[0]
        self.y = self.initPos[1]
        self.xc = self.x
        self.yc = self.y
        self.c = np.array([self.xc, self.yc])
        self.rotation = self.initPos[2]
        self.rotation_rad = np.deg2rad(self.rotation)
        self.vel = 0.0
        self.acc = 0.0

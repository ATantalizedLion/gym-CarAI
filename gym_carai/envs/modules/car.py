import pyglet
import numpy as np

from v1.game.util import center_image, Line
from pyglet.window import key


class Bumper(Line):
    def __init__(self, pos):
        super().__init__(pos)
        self.set_image('BlueBar.png')


class Car:
    def __init__(self, initial_position=(100, 100), car_length=64):
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
        self.rotation = 0.001
        self.rotation_rad = np.deg2rad(self.rotation)
        self.vel = 0.0
        self.acc = 0.0

        # related to controls
        self.key_handler = key.KeyStateHandler()
        self.rotate_speed = 260.0
        self.acc_speed = 300.0

        front_corner1 = self.c + np.cos(self.rotation_rad) * np.array([-self.width/2, self.height/2])
        front_corner2 = self.c + np.cos(self.rotation_rad) * np.array([self.width/2, self.height/2])
        rear_corner1 = self.c - np.cos(self.rotation_rad) * np.array([self.width/2, self.height/2])
        rear_corner2 = self.c - np.cos(self.rotation_rad) * np.array([-self.width/2, self.height/2])
        self.Bumper = Bumper([front_corner1[0], front_corner1[1], front_corner2[0], front_corner2[1]])
        self.Side1 = Bumper([front_corner1[0], front_corner1[1], rear_corner1[0], rear_corner1[1]])
        self.Side2 = Bumper([front_corner2[0], front_corner2[1], rear_corner2[0], rear_corner2[1]])
        self.Rear = Bumper([rear_corner1[0], rear_corner1[1], rear_corner2[0], rear_corner2[1]])

    def update(self, dt):
        if self.key_handler[key.LEFT]:
            self.rotation -= self.rotate_speed * dt
        if self.key_handler[key.RIGHT]:
            self.rotation += self.rotate_speed * dt
        if self.key_handler[key.DOWN]:
            if self.vel > 0:
                self.vel += -10*self.acc_speed * dt
            else:
                self.vel = 0
        if self.key_handler[key.UP]:
            self.vel += self.acc_speed * dt

        self.rotation_rad = np.deg2rad(self.rotation)
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
        fc = self.c + np.cos(self.rotation_rad) * np.array([0.0, self.height/2]) - np.sin(self.rotation_rad) * np.array([-self.height/2, 0.0])
        rc = self.c - np.cos(self.rotation_rad) * np.array([0.0, self.height/2]) + np.sin(self.rotation_rad) * np.array([-self.height/2, 0.0])
        s1 = self.c - np.sin(self.rotation_rad) * np.array([0.0, self.width/2]) - np.cos(self.rotation_rad) * np.array([-self.width/2, 0.0])
        s2 = self.c + np.sin(self.rotation_rad) * np.array([0.0, self.width/2]) + np.cos(self.rotation_rad) * np.array([-self.width/2, 0.0])
        self.Bumper.update_position([fc[0], fc[1], self.rotation])
        self.Rear.update_position([rc[0], rc[1], self.rotation])
        self.Side1.update_position([s1[0], s1[1], self.rotation-90])
        self.Side2.update_position([s2[0], s2[1], self.rotation-90])

    def reset(self):
        self.x = self.initPos[0]
        self.y = self.initPos[1]
        self.xc = self.x
        self.yc = self.y
        self.c = np.array([self.xc, self.yc])
        self.rotation = 0.001
        self.rotation_rad = np.deg2rad(self.rotation)
        self.vel = 0.0
        self.acc = 0.0

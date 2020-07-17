import pyglet
import numpy as np
from gym_carai.envs.modules.util import center_image, LineObject
from pyglet.window import key

# TODO: base sensor on offset from car centre.
class Bumper(LineObject):
    def __init__(self, pos, debug_batch):
        super().__init__(pos)
        self.create_sprite(debug_batch, color=(0, 255, 255))

    def update_position(self, pos):
        # pos is either 3 long: x,y,theta, or 4 long, x1,y1,x2,y2
        if len(pos) == 3:
            self.x, self.y, self.rotation = pos
            self.rotation_rad = np.deg2rad(self.rotation)
            if self.sprite is not None:
                self.sprite.update_position_rot(-self.rotation, self.x, self.y)
            self.x1 = self.x + np.cos(self.rotation_rad) * -self.width / 2
            self.x2 = self.x + np.cos(self.rotation_rad) * self.width / 2
            self.y1 = self.y + np.sin(self.rotation_rad) * self.width / 2
            self.y2 = self.y - np.sin(self.rotation_rad) * self.width / 2


class Sensor(LineObject):
    def __init__(self, debug_batch, sensor_range, sensor_orientation, offset):
        self.sensor_orientation = sensor_orientation
        rad_orientation = np.deg2rad(sensor_orientation)
        pos = [0, 0, sensor_range*np.sin(rad_orientation), sensor_range*np.cos(rad_orientation)]
        super().__init__(pos, height=2)
        self.create_sprite(debug_batch, color=(0, 0, 255))
        self.colimg = pyglet.resource.image('marker.png')
        center_image(self.colimg)
        self.collision_marker = pyglet.sprite.Sprite(img=self.colimg, x=-10, y=-10, batch=debug_batch)
        self.sensor_range = sensor_range
        self.offset = offset  # vector from car centre to attachment point of sensor, in car coordinates.

class Car:
    def __init__(self, initial_position=(100, 100, 0), car_length=64, main_batch=[], debug_batch=[], mode='simple'):
        self.debug_batch = debug_batch
        self.image = pyglet.resource.image("car.png")
        center_image(self.image)
        self.initPos = initial_position
        self.x = initial_position[0]
        self.y = initial_position[1]
        self.c = np.array([self.x, self.y])
        self.sprite = pyglet.sprite.Sprite(img=self.image, x=self.x, y=self.y, batch=main_batch)
        self.sprite.scale = car_length / self.image.height
        self.width = self.sprite.width
        self.height = self.sprite.height
        self.rotation = initial_position[2]
        self.rotation_rad = np.deg2rad(self.rotation)
        self.sprite.rotation = self.rotation
        self.mode = mode
        self.sensorRange = 1000

        # related to controls
        self.key_handler = key.KeyStateHandler()
        if mode == 'simple':
            self.turn_speed = 180.0
            self.vel = 90
        elif mode == 'less-simple':
            self.turn_speed = 180.0
            self.vel = 90
        else:
            # for non simple mode
            self.rotation_speed = 0
            self.vel_x = 0
            self.vel_y = 0

            # applied 'forces'
            self.acc_power = 400.0
            self.turn_speed = 720.0
            self.brake_force = 4
            self.angular_brake_force = 4

            # decay factors
            self.drifting_factor = 3.5
            self.drag = 0.9
            self.angular_drag = 0.85

        # set up bumpers, distance sensors - only size is relevant, proper orientation will be set in the first step.
        self.Bumper = Bumper([-self.width / 2, 0, self.width / 2, 0], self.debug_batch)
        self.SideL = Bumper([0, self.height / 2, 0, -self.height / 2], self.debug_batch)
        self.SideR = Bumper([0, self.height / 2, 0, -self.height / 2], self.debug_batch)
        self.Rear = Bumper([-self.width / 2, 0, self.width / 2, 0], self.debug_batch)

        # right + left side sensor
        move_fact = 0.4  # -0.5 is back, 0.5 is front
        fsl = [-self.width/2, self.height * move_fact]
        fsr = [self.width/2, self.height * move_fact]
        fc = [0, self.height/2]
        rc = [0, -self.height/2]
        flc = [-self.width/2, self.height/2]
        frc = [self.width/2, self.height/2]

        # set up distance sensors
        if mode == 'simple':
            RightDistanceSensor = Sensor(self.debug_batch, self.sensorRange, 90, fsr)
            self.sensors = [RightDistanceSensor]
        elif mode == 'less-simple':
            # FrontDistanceSensorL = Sensor(self.debug_batch, self.sensorRange, 0, flc)
            FrontDistanceSensor = Sensor(self.debug_batch, self.sensorRange, 0, fc)
            # FrontDistanceSensorR = Sensor(self.debug_batch, self.sensorRange, 0, frc)
            # RearDistanceSensor = Sensor(self.debug_batch, self.sensorRange, 180, rc)
            LeftDistanceSensor = Sensor(self.debug_batch, self.sensorRange, -90, fsl)
            RightDistanceSensor = Sensor(self.debug_batch, self.sensorRange, 90, fsr)
            # RightCornerAngled1 = Sensor(self.debug_batch, self.sensorRange, 35, frc)
            # LeftCornerAngled1 = Sensor(self.debug_batch, self.sensorRange, -35, flc)
            # RightCornerAngled2 = Sensor(self.debug_batch, self.sensorRange, 65, frc)
            # LeftCornerAngled2 = Sensor(self.debug_batch, self.sensorRange, -65, flc)
            self.sensors = [FrontDistanceSensor, # FrontDistanceSensorR, FrontDistanceSensorL,
                            LeftDistanceSensor, RightDistanceSensor]  # ,  # RearDistanceSensor,
                            # RightCornerAngled1, LeftCornerAngled1, RightCornerAngled2, LeftCornerAngled2]
        else:
            # FrontDistanceSensorL = Sensor(self.debug_batch, self.sensorRange, 0, flc)
            FrontDistanceSensor = Sensor(self.debug_batch, self.sensorRange, 0, fc)
            # FrontDistanceSensorR = Sensor(self.debug_batch, self.sensorRange, 0, frc)
            # RearDistanceSensor = Sensor(self.debug_batch, self.sensorRange, 180, rc)
            LeftDistanceSensor = Sensor(self.debug_batch, self.sensorRange, -90, fsl)
            RightDistanceSensor = Sensor(self.debug_batch, self.sensorRange, 90, fsr)
            RightCornerAngled1 = Sensor(self.debug_batch, self.sensorRange, 35, frc)
            LeftCornerAngled1 = Sensor(self.debug_batch, self.sensorRange, -35, flc)
            RightCornerAngled2 = Sensor(self.debug_batch, self.sensorRange, 65, frc)
            LeftCornerAngled2 = Sensor(self.debug_batch, self.sensorRange, -65, flc)
            self.sensors = [FrontDistanceSensor, # FrontDistanceSensorR, FrontDistanceSensorL,
                            LeftDistanceSensor, RightDistanceSensor,  # RearDistanceSensor,
                            RightCornerAngled1, LeftCornerAngled1, RightCornerAngled2, LeftCornerAngled2]

    def update(self, dt, action):
        """"
        action [0] = -1 to 1, steering
        action [1] = -1 to 1, gas
        action [2] = brake (bool)
        """
        if self.mode == 'simple' or self.mode == 'less-simple':
            self.rotation += action[0] * self.turn_speed * dt

            self.rotation_rad = np.deg2rad(self.rotation)
            cos = np.cos(self.rotation_rad)
            sin = np.sin(self.rotation_rad)

            self.x += self.vel * sin * dt
            self.y += self.vel * cos * dt
        else:
            # Currently quite sluggish, this was in an attempt to make it easier for the AI.
            self.rotation_speed += action[0] * self.turn_speed * dt
            self.rotation += self.rotation_speed * dt
            self.rotation_rad = np.deg2rad(self.rotation)
            cos = np.cos(self.rotation_rad)
            sin = np.sin(self.rotation_rad)

            self.vel_x += self.acc_power * dt * action[1] * sin
            self.vel_y += self.acc_power * dt * action[1] * cos
            self.x += self.vel_x * dt
            self.y += self.vel_y * dt

            self.vel_x -= self.vel_x * self.drag * dt
            self.vel_y -= self.vel_y * self.drag * dt

            x_car, y_car = global_to_car(self.vel_x, self.vel_y, self.rotation_rad)
            x_car -= x_car * self.drifting_factor * dt
            self.vel_x, self.vel_y = car_to_global(x_car, y_car, self.rotation_rad)

            self.rotation_speed -= (self.angular_drag + self.angular_brake_force) * self.rotation_speed * dt
            # self.rotation += action[0] * self.turn_speed * dt
            # self.vel = self.vel + action[1] * self.acc_power * dt
            #
            # self.rotation_rad = np.deg2rad(self.rotation)
            # cos = np.cos(self.rotation_rad)
            # sin = np.sin(self.rotation_rad)
            #
            # self.x += self.vel * sin * dt
            # self.y += self.vel * cos * dt

        self.sprite.x = self.x
        self.sprite.y = self.y
        self.sprite.rotation = self.rotation

        self.c = np.array([self.x, self.y])

        # Calculate bumper positions based on rotation and center position
        fc = self.c + cos * np.array([0.0, self.height/2]) - sin * np.array([-self.height/2, 0.0])  # front centre
        rc = self.c - cos * np.array([0.0, self.height/2]) + sin * np.array([-self.height/2, 0.0])  # rear centre
        sl = self.c - sin * np.array([0.0, self.width/2]) - cos * np.array([-self.width/2, 0.0])  # left side centre
        sr = self.c + sin * np.array([0.0, self.width/2]) + cos * np.array([-self.width/2, 0.0])  # right side centre

        self.Bumper.update_position([fc[0], fc[1], self.rotation])
        self.Rear.update_position([rc[0], rc[1], self.rotation])
        self.SideL.update_position([sl[0], sl[1], (self.rotation - 90)])
        self.SideR.update_position([sr[0], sr[1], (self.rotation - 90)])

        # Calculate line based sensor position
        for sensor in self.sensors:
            sensor_x, sensor_y = car_to_global(sensor.offset[0], sensor.offset[1], self.rotation_rad)
            sensor.update_position_x1y1([self.x + sensor_x, self.y + sensor_y, self.rotation+sensor.sensor_orientation])

    def reset(self):
        self.x, self.y, self.rotation = self.initPos
        self.c = np.array([self.x, self.y])
        self.rotation_rad = np.deg2rad(self.rotation)
        self.acc = 0.0
        self.rotation_speed = 0
        self.vel_x = 0
        self.vel_y = 0
        if self.mode == 'simple':
            self.vel = 90
        elif self.mode == 'less-simple':
            self.vel = 90
        else:
            self.vel = 0


def global_to_car(x, y, theta):
    """ converts an x and y vector in global coordinate system to a forward/right pointing vector """
    # global velocities in car coordinates
    cos = np.cos(theta)
    sin = np.sin(theta)
    car_x = cos * x - sin * y
    car_y = cos * y + sin * x
    return car_x, car_y


def car_to_global(x, y, theta):
    """ converts an x and y vector in car coordinate system to a global x/y vector
    Theta is the rotation of the car"""
    # car velocities in global coordinates
    cos = np.cos(-theta)
    sin = np.sin(-theta)
    global_x = cos * x - sin * y
    global_y = cos * y + sin * x
    return global_x, global_y

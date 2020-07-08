import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pyglet
import numpy as np
from gym_carai.envs.modules.track import generate_track
from gym_carai.envs.modules.car import Car
from gym_carai.envs.modules.util import line_overlapping, vector_length
from gym_carai.envs.modules.viewer import Viewer

pyglet.options['debug_gl'] = False  # performance increase
window_h_size = 1920
window_v_size = 1080
debug = 0  # renders all bumpers, sensors and collision markers.


class SimpleCarAIEnv(gym.Env):
    metadata = {'render.modes': ['human', 'human-vsync', 'rgb_array', 'manual'], 'video.frames_per_second': 60}

    def __init__(self):
        pyglet.resource.path = ['gym_carai/envs/resources']
        pyglet.resource.reindex()
        score_label_font_size = 36
        self.viewer = None
        self.Terminate = False
        self.manual = None
        self.keys = None
        self.vsync = False

        # 3 batches, one for car, one for obstacles, one for debug features
        self.main_batch = pyglet.graphics.Batch()
        self.track_batch = pyglet.graphics.Batch()
        self.debug_batch = pyglet.graphics.Batch()

        # Labels for score and level, scaled based upon score label font size
        self.score = 0
        self.t = 0
        self.episode = 0

        self.done = 0
        self.reward = 0
        self.JStar = 0
        self.observations = np.array([[0, 0]])

        self.score_label = None
        self.track_label = None
        self.time_label = None
        self.episode_label = None

        self.action_space = spaces.Box(np.array([-1]), np.array([+1]))  # steering only, -1 to +1 on one action

        self.track_name = 'simpleSquareTrack'

        # define functions
        self.walls, self.checkpoints, car_position = generate_track(
            'gym_carai/envs/resources/' + self.track_name + '.csv')
        self.current_checkpoint = 1

        for item in self.walls:
            item.sprite.batch = self.track_batch
        for item in self.checkpoints:
            item.sprite.batch = self.track_batch

        self.car_obj = Car(car_position, debug_batch=self.debug_batch, mode='simple')
        self.car_obj.sprite.batch = self.main_batch
        self.car_bumpers = [self.car_obj.Bumper, self.car_obj.SideL, self.car_obj.SideR, self.car_obj.Rear]

        # all objects requiring .update()
        self.envObjects = [self.car_obj]

        # implemented in this env:
        # self.sensors = [self.car_obj.FrontDistanceSensor, self.car_obj.RightDistanceSensor,
        #                 self.car_obj.RearDistanceSensor, self.car_obj.LeftDistanceSensor]
        self.sensors = [self.car_obj.RightDistanceSensor, self.car_obj.LeftDistanceSensor, self.car_obj.FrontDistanceSensor]

        self.observation_space = spaces.Box(np.zeros(len(self.sensors)),
                                            600*np.ones(len(self.sensors)))
        self.observations = np.array([np.zeros(len(self.sensors))])

    def step(self, action, dt):
        """"observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)"""
        self.reward = 0
        if self.manual:
            if self.keys[pyglet.window.key.LEFT] == 1:
                if self.keys[pyglet.window.key.RIGHT] == 1:
                    action[0] = 0
                else:
                    action[0] = -1
            elif self.keys[pyglet.window.key.RIGHT] == 1:
                action[0] = 1
            else:
                action[0] = 0
            # if self.keys[pyglet.window.key.UP] == 1:
            #     action[1] = 1
            # else:
            #     action[1] = 0
            # if self.keys[pyglet.window.key.DOWN] == 1:
            #     action[2] = 1
            # else:
            #     action[2] = 0

        for obj in self.envObjects:
            obj.update(dt, action)

        for obj in self.walls:
            for car_bumper in self.car_bumpers:
                tf, _, _, _, _, _, _ = line_overlapping(car_bumper.line(), obj.line())
                if tf:
                    self.done = 1

        for obj in self.checkpoints:
            for car_bumper in self.car_bumpers:
                if obj.id == self.current_checkpoint:
                    tf, _, _, _, _, _, _ = line_overlapping(car_bumper.line(), obj.line())
                    if tf:
                        self.current_checkpoint += 1
                        if self.current_checkpoint > len(self.checkpoints):
                            self.current_checkpoint -= len(self.checkpoints)
                        self.score += obj.score
                        self.reward += obj.score
                        self.score_label.text = "Current Score: " + str(self.score)

        current_sensor_number = 0
        for sensor in self.sensors:
            min_distance = sensor.sensor_range
            dist = min_distance
            col_loc = [-50, -50]  # give some value to prevent crash
            for obj in self.walls:
                tf, p, q, t, r, u, s = line_overlapping(sensor.line(), obj.line())
                if tf:
                    dist = vector_length(p, p + t * r)
                    if dist < min_distance:
                        min_distance = dist
                        col_loc = p + t * r
            if dist < 900000:
                sensor.collision_marker.x = col_loc[0]
                sensor.collision_marker.y = col_loc[1]
            else:
                # no collision found, draw off-screen
                sensor.collision_marker.x = -50
                sensor.collision_marker.y = -50
            self.observations[0][current_sensor_number] = min_distance
            current_sensor_number+=1
        self.t += dt
        if self.time_label:
            self.time_label.text = "Current Episode Time: " + str(round(self.t))
        if self.viewer:
            if not self.Terminate:
                self.Terminate = self.viewer.Terminate

        if self.reward == 0:
            self.reward = -0.05  # reward for current state if not at a checkpoint

        if self.done:
            self.reward = -50 - 100/self.t  # penalty for hitting wall, higher penalty if wall is hit early

        self.JStar = 0
        return self.observations, self.reward, self.done, {'t': self.t, 'JStar': self.JStar}, self.Terminate  # , 'JStar': self.JStar

    def reset(self):
        self.score = 0
        self.current_checkpoint = 1
        self.t = 0
        self.reward = 0
        self.done = 0
        for obj in self.envObjects:
            obj.reset()
        if self.viewer:
            self.episode_label.text = "Current episode:" + str(self.episode)
            self.score_label.text = "Current Score: " + str(self.score)
        self.episode += 1

    def render(self, mode='human'):
        assert mode in ['human', 'human-vsync', 'rgb_array', 'manual']
        if mode == 'human-vsync':
            self.vsync = True
        if mode == 'manual':
            self.manual = 1
        if self.viewer is None:
            self.viewer = Viewer(window_h_size, window_v_size, self.manual, self.vsync)
            [self.score_label, self.track_label, self.time_label, self.episode_label] = \
                self.viewer.labels(self.main_batch, 36, self.score, self.track_name, self.t, self.episode)
            # redefine draw event
            pyglet.gl.glClearColor(1, 1, 1, 1)  # white background
            self.viewer.toDraw = [self.track_batch, self.main_batch]
            if debug:
                self.viewer.toDraw.append(self.debug_batch)
            if self.manual:
                self.keys = pyglet.window.key.KeyStateHandler()
        if self.viewer.is_open:
            if mode == 'human' or mode == 'manual':
                self.viewer.render()
            elif mode == 'rgb_array':
                rgb = self.viewer.render(True)
                return rgb
        if mode == 'manual':
            self.viewer.window.push_handlers(self.keys)

    def close(self):
        if self.viewer:
            self.viewer.close()

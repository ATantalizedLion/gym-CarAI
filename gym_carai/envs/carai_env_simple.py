import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pyglet
import numpy as np

from gym_carai.envs.modules.track import generate_track
from gym_carai.envs.modules.car import Car
from gym_carai.envs.modules.util import line_overlapping, vector_length
from gym_carai.envs.modules.viewer import Viewer

window_h_size = 1920
window_v_size = 1080
FPS = 60
debug = 0

class SimpleCarAIEnv(gym.Env):
    metadata = {'render.modes': ['human'], 'video.frames_per_second': FPS}

    def __init__(self):
        pyglet.resource.path = ['gym_carai/envs/resources']
        pyglet.resource.reindex()
        score_label_font_size = 36
        self.dt = 1 / FPS
        self.viewer = None

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
        self.observations = None

        self.score_label = None
        self.track_label = None
        self.time_label = None
        self.episode_label = None

        self.action_space = spaces.Box(np.array([-1], dtype=np.float64),
                                       np.array([+1], dtype=np.float64),
                                       dtype=np.float64)  # steering only
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

        self.sensors = [self.car_obj.FrontDistanceSensor, self.car_obj.RightDistanceSensor,
                        self.car_obj.RearDistanceSensor, self.car_obj.LeftDistanceSensor]

    def step(self, action):
        """"observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)"""
        for obj in self.envObjects:
            obj.update(self.dt, action)

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
                        self.score_label.text = "Current Score: " + str(self.score)

        self.observations = []
        for sensor in self.sensors:
            min_distance = sensor.sensor_range
            dist = 900000
            col_loc = []
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
                self.observations.append(min_distance)
            else:
                # no collision found, draw off-screen
                sensor.collision_marker.update_position([-50, -50])
                self.observations.append(min_distance)  # front left rear right
        self.reward = self.score
        if self.time_label:
            self.t += self.dt
            self.time_label.text = "Current Episode Time: " + str(round(self.t))
        return self.observations, self.reward, self.done, 'noinfo', self.viewer.Terminate

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
        self.episode += 1

    def render(self, mode='human'):
        assert mode in ['human', 'rgb_array']
        if self.viewer is None:
            self.viewer = Viewer(window_h_size, window_v_size, False)
            [self.score_label, self.track_label, self.time_label, self.episode_label] = \
                self.viewer.labels(self.main_batch, 36, self.score, self.track_name, self.t, self.episode)
            # redefine draw event
            pyglet.gl.glClearColor(1, 1, 1, 1)  # white background
            self.viewer.toDraw = [self.track_batch, self.main_batch]
            if debug:
                self.viewer.toDraw.append(self.debug_batch)
        if self.viewer.is_open:
            if mode == 'human':
                self.viewer.render()
            elif mode == 'rgb_array':
                rgb = self.viewer.render(True)
                return rgb

    def close(self):
        if self.viewer:
            self.viewer.close()
